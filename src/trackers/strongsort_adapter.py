# ABOUTME: Adapter for StrongSORT tracking backend with enhanced re-id and Kalman filtering
# ABOUTME: Handles import isolation (avoids collision with DeepSORT's identically-named modules)

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import FaceDetection, TrackedFace

_STRONG_SORT_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor" / "strong_sort"


def _inject_opts():
    """Inject the opts module with StrongSORT configuration before importing."""
    opts_module = types.ModuleType("opts")
    opts_module.opt = types.SimpleNamespace(
        NSA=True,
        EMA=True,
        EMA_alpha=0.9,
        MC=True,
        MC_lambda=0.98,
        woC=True,
        ECC=False,
    )
    sys.modules["opts"] = opts_module


def _inject_sklearn_compat():
    """Inject a compatibility shim for the deprecated sklearn linear_assignment."""
    # StrongSORT imports from sklearn.utils.linear_assignment_ which was removed in sklearn 0.24+
    if "sklearn.utils.linear_assignment_" not in sys.modules:
        compat_module = types.ModuleType("sklearn.utils.linear_assignment_")

        def _linear_assignment(cost_matrix):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return np.column_stack((row_ind, col_ind))

        compat_module.linear_assignment = _linear_assignment
        sys.modules["sklearn.utils.linear_assignment_"] = compat_module


def _load_strongsort_module(name: str):
    """Load a module from StrongSORT's deep_sort package, isolated from DeepSORT's.

    We load StrongSORT's deep_sort modules under the key 'ss_deep_sort.*' in sys.modules
    to avoid collisions with the original DeepSORT (nwojke/deep_sort).
    """
    ss_key = f"ss_deep_sort.{name}"
    if ss_key in sys.modules:
        return sys.modules[ss_key]

    # First ensure the parent package exists
    parent_key = "ss_deep_sort"
    if parent_key not in sys.modules:
        parent = types.ModuleType(parent_key)
        parent.__path__ = [str(_STRONG_SORT_ROOT / "deep_sort")]
        parent.__package__ = parent_key
        sys.modules[parent_key] = parent

    file_path = _STRONG_SORT_ROOT / "deep_sort" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(ss_key, file_path)
    module = importlib.util.module_from_spec(spec)

    # Temporarily swap all deep_sort.* entries so StrongSORT's internal imports
    # resolve to our ss_deep_sort modules instead of DeepSORT's cached ones
    saved = {}
    for key in list(sys.modules.keys()):
        if key == "deep_sort" or key.startswith("deep_sort."):
            saved[key] = sys.modules.pop(key)

    # Map deep_sort -> ss_deep_sort and all its already-loaded submodules
    sys.modules["deep_sort"] = sys.modules[parent_key]
    for key, mod in list(sys.modules.items()):
        if key.startswith("ss_deep_sort."):
            subname = key[len("ss_deep_sort."):]
            sys.modules[f"deep_sort.{subname}"] = mod

    sys.modules[ss_key] = module
    spec.loader.exec_module(module)

    # Restore original deep_sort modules
    for key in list(sys.modules.keys()):
        if key == "deep_sort" or key.startswith("deep_sort."):
            sys.modules.pop(key, None)
    sys.modules.update(saved)

    # Also register as a submodule
    setattr(sys.modules[parent_key], name, module)
    return module


# Inject compatibility before loading
_inject_opts()
_inject_sklearn_compat()

# Load StrongSORT modules in dependency order
_ss_kalman = _load_strongsort_module("kalman_filter")
_ss_detection = _load_strongsort_module("detection")
_ss_nn_matching = _load_strongsort_module("nn_matching")
_ss_iou_matching = _load_strongsort_module("iou_matching")
_ss_linear_assignment = _load_strongsort_module("linear_assignment")
_ss_track = _load_strongsort_module("track")
_ss_tracker = _load_strongsort_module("tracker")

SSDetection = _ss_detection.Detection
SSNearestNeighborDistanceMetric = _ss_nn_matching.NearestNeighborDistanceMetric
SSTracker = _ss_tracker.Tracker


def _xyxy_to_tlwh(bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)


class StrongSortAdapter:
    """Adapter for StrongSORT tracker."""

    def __init__(
        self,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 1,  # EMA mode uses budget=1
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
    ):
        self._max_cosine_distance = max_cosine_distance
        self._nn_budget = nn_budget
        self._max_iou_distance = max_iou_distance
        self._max_age = max_age
        self._n_init = n_init
        self._tracker = self._create_tracker()

    def _create_tracker(self) -> SSTracker:
        metric = SSNearestNeighborDistanceMetric(
            "cosine", self._max_cosine_distance, self._nn_budget
        )
        return SSTracker(
            metric,
            max_iou_distance=self._max_iou_distance,
            max_age=self._max_age,
            n_init=self._n_init,
        )

    def reset(self) -> None:
        self._tracker = self._create_tracker()

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        ss_detections = []
        for d in detections:
            if d.embedding is None:
                raise ValueError("StrongSORT requires embeddings for each detection")
            tlwh = _xyxy_to_tlwh(d.bbox_xyxy)
            ss_detections.append(SSDetection(tlwh, d.confidence, d.embedding))

        self._tracker.predict()
        self._tracker.update(ss_detections)

        results = []
        for track in self._tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            results.append(
                TrackedFace(
                    bbox_xyxy=bbox.astype(np.float32),
                    track_id=track.track_id,
                    confidence=1.0,
                )
            )
        return results

    @property
    def name(self) -> str:
        return "StrongSORT"

    @property
    def needs_embeddings(self) -> bool:
        return True
