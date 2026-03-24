# ABOUTME: Adapter for the SORT (Simple Online and Realtime Tracking) backend
# ABOUTME: Wraps abewley/sort with monkey-patched imports for headless operation

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ..types import FaceDetection, TrackedFace


def _patch_sort_imports():
    """Set matplotlib to headless backend before sort.py tries to use TkAgg."""
    import matplotlib
    matplotlib.use("Agg")  # Headless backend — must be called before sort.py does use('TkAgg')

    # sort.py calls matplotlib.use('TkAgg') at import time, which will fail in headless.
    # We monkey-patch matplotlib.use to be a no-op after setting our backend.
    matplotlib.use = lambda *a, **kw: None


# Patch before importing SORT
_patch_sort_imports()

# Add vendor/sort to path so we can import sort.py
_sort_dir = str(Path(__file__).resolve().parent.parent.parent / "vendor" / "sort")
if _sort_dir not in sys.path:
    sys.path.insert(0, _sort_dir)

from sort import Sort  # noqa: E402


class SortAdapter:
    """Adapter for SORT tracker."""

    def __init__(self, max_age: int = 5, min_hits: int = 1, iou_threshold: float = 0.3):
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._tracker = Sort(
            max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold
        )

    def reset(self) -> None:
        self._tracker = Sort(
            max_age=self._max_age,
            min_hits=self._min_hits,
            iou_threshold=self._iou_threshold,
        )

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        if not detections:
            dets = np.empty((0, 5))
        else:
            dets = np.array(
                [[*d.bbox_xyxy, d.confidence] for d in detections], dtype=np.float64
            )

        results = self._tracker.update(dets)

        return [
            TrackedFace(
                bbox_xyxy=r[:4].astype(np.float32),
                track_id=int(r[4]),
                confidence=1.0,
            )
            for r in results
        ]

    @property
    def name(self) -> str:
        return "SORT"

    @property
    def needs_embeddings(self) -> bool:
        return False
