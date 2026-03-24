# ABOUTME: Video reading and writing utilities using OpenCV
# ABOUTME: VideoReader iterates frames; VideoWriter outputs MP4 with H.264 codec

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Iterates over frames of a video file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __iter__(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        return frame

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


class VideoWriter:
    """Writes frames to an MP4 video file."""

    def __init__(self, path: str | Path, fps: float, resolution: tuple[int, int]):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, resolution)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __del__(self):
        if hasattr(self, "_writer"):
            self._writer.release()
