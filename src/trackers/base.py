# ABOUTME: Abstract interface (Protocol) for tracker adapters
# ABOUTME: All tracker backends (SORT, DeepSORT, StrongSORT) implement this interface

from __future__ import annotations

from typing import Protocol

from ..types import FaceDetection, TrackedFace


class TrackerAdapter(Protocol):
    """Unified interface for all tracking backends."""

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        ...

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        """Process one frame's detections and return tracked faces."""
        ...

    @property
    def name(self) -> str:
        """Human-readable tracker name."""
        ...

    @property
    def needs_embeddings(self) -> bool:
        """Whether this tracker requires face embeddings."""
        ...
