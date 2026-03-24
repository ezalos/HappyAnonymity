# ABOUTME: Shared data types for the face anonymization pipeline
# ABOUTME: Defines FaceDetection (detector output) and TrackedFace (tracker output)

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceDetection:
    """A detected face before tracking."""

    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    embedding: np.ndarray | None = None  # 512-dim ArcFace, None for SORT


@dataclass
class TrackedFace:
    """A tracked face with persistent identity."""

    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2]
    track_id: int
    confidence: float


@dataclass
class FrameMetrics:
    """Per-frame metrics for comparison."""

    frame_idx: int
    tracker_name: str
    num_detections: int
    num_tracks: int
    active_track_ids: list[int] = field(default_factory=list)
    process_time_ms: float = 0.0
