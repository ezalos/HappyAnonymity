# ABOUTME: Adapter for the DeepSORT (Deep Simple Online and Realtime Tracking) backend
# ABOUTME: Wraps nwojke/deep_sort with ArcFace embeddings for face re-identification

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..types import FaceDetection, TrackedFace

# Add vendor/deep_sort to path
_deepsort_dir = str(Path(__file__).resolve().parent.parent.parent / "vendor" / "deep_sort")
if _deepsort_dir not in sys.path:
    sys.path.insert(0, _deepsort_dir)

from deep_sort.detection import Detection  # noqa: E402
from deep_sort.nn_matching import NearestNeighborDistanceMetric  # noqa: E402
from deep_sort.tracker import Tracker  # noqa: E402


def _xyxy_to_tlwh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [top_left_x, top_left_y, width, height]."""
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)


class DeepSortAdapter:
    """Adapter for DeepSORT tracker."""

    def __init__(
        self,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
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

    def _create_tracker(self) -> Tracker:
        metric = NearestNeighborDistanceMetric(
            "cosine", self._max_cosine_distance, self._nn_budget
        )
        return Tracker(
            metric,
            max_iou_distance=self._max_iou_distance,
            max_age=self._max_age,
            n_init=self._n_init,
        )

    def reset(self) -> None:
        self._tracker = self._create_tracker()

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        ds_detections = []
        for d in detections:
            if d.embedding is None:
                raise ValueError("DeepSORT requires embeddings for each detection")
            tlwh = _xyxy_to_tlwh(d.bbox_xyxy)
            ds_detections.append(Detection(tlwh, d.confidence, d.embedding))

        self._tracker.predict()
        self._tracker.update(ds_detections)

        results = []
        for track in self._tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
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
        return "DeepSORT"

    @property
    def needs_embeddings(self) -> bool:
        return True
