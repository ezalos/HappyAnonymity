# ABOUTME: Core video processing pipeline for face anonymization
# ABOUTME: Orchestrates detect → track → blur loop with metrics collection

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from .blurring import blur_faces
from .detector import FaceDetector
from .trackers.base import TrackerAdapter
from .types import FrameMetrics
from .video_io import VideoReader, VideoWriter


class AnonymizationPipeline:
    """Processes a video: detects faces, tracks them, and blurs them."""

    def __init__(
        self,
        detector: FaceDetector,
        tracker: TrackerAdapter,
        blur_kernel: int = 99,
    ):
        self.detector = detector
        self.tracker = tracker
        self.blur_kernel = blur_kernel

    def process_video(
        self,
        input_path: str,
        output_path: str,
        draw_boxes: bool = False,
    ) -> list[FrameMetrics]:
        """Process a video: detect, track, and blur faces.

        Args:
            input_path: Path to input video.
            output_path: Path to output video.
            draw_boxes: If True, draw bounding boxes with track IDs on the output.

        Returns:
            Per-frame metrics.
        """
        reader = VideoReader(input_path)
        writer = VideoWriter(output_path, reader.fps, reader.resolution)
        self.tracker.reset()

        metrics: list[FrameMetrics] = []

        for frame_idx, frame in enumerate(
            tqdm(reader, total=reader.frame_count, desc=f"Processing [{self.tracker.name}]")
        ):
            t_start = time.perf_counter()

            # Detect faces
            detections = self.detector.detect(frame)

            # Track
            tracked = self.tracker.update(detections)

            t_elapsed = (time.perf_counter() - t_start) * 1000

            # Blur faces
            bboxes = [t.bbox_xyxy for t in tracked]
            result = blur_faces(frame, bboxes, self.blur_kernel)

            # Draw bounding boxes with track IDs
            if draw_boxes:
                result = _draw_tracked_faces(result, tracked)

            writer.write(result)

            metrics.append(
                FrameMetrics(
                    frame_idx=frame_idx,
                    tracker_name=self.tracker.name,
                    num_detections=len(detections),
                    num_tracks=len(tracked),
                    active_track_ids=[t.track_id for t in tracked],
                    process_time_ms=t_elapsed,
                )
            )

        writer.release()
        return metrics


def detection_only(
    detector: FaceDetector,
    input_path: str,
    output_path: str,
) -> None:
    """Detection-only mode: draw boxes without tracking or blurring."""
    import cv2

    reader = VideoReader(input_path)
    writer = VideoWriter(output_path, reader.fps, reader.resolution)

    for frame in tqdm(reader, total=reader.frame_count, desc="Detecting"):
        detections = detector.detect(frame)
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result,
                f"{det.confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        writer.write(result)

    writer.release()


def _draw_tracked_faces(frame: np.ndarray, tracked: list, color_seed: int = 42) -> np.ndarray:
    """Draw bounding boxes with track IDs on frame."""
    import cv2

    rng = np.random.RandomState(color_seed)
    colors = {}

    for t in tracked:
        if t.track_id not in colors:
            colors[t.track_id] = tuple(int(c) for c in rng.randint(50, 255, 3))
        color = colors[t.track_id]
        x1, y1, x2, y2 = t.bbox_xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID:{t.track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame
