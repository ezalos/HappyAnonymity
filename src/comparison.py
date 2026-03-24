# ABOUTME: Comparison utilities for evaluating tracker performance
# ABOUTME: Generates 2x2 grid video, metrics CSV, and summary table across trackers

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .types import FrameMetrics
from .video_io import VideoReader, VideoWriter


def create_comparison_video(
    original_path: str,
    tracker_videos: dict[str, str],
    output_path: str,
) -> None:
    """Create a 2x2 grid comparison video.

    Layout:
    ┌──────────┬──────────┐
    │ Original │  SORT    │
    ├──────────┼──────────┤
    │ DeepSORT │StrongSORT│
    └──────────┴──────────┘
    """
    # Open all video readers
    original = VideoReader(original_path)
    readers = {name: VideoReader(path) for name, path in tracker_videos.items()}
    tracker_names = list(tracker_videos.keys())

    h, w = original.height, original.width
    grid_w, grid_h = w * 2, h * 2
    writer = VideoWriter(output_path, original.fps, (grid_w, grid_h))

    label_order = ["Original"] + tracker_names

    for orig_frame in tqdm(original, total=original.frame_count, desc="Creating comparison"):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        frames = {"Original": orig_frame}
        for name, reader in readers.items():
            ret, frame = reader._cap.read()
            frames[name] = frame if ret else np.zeros_like(orig_frame)

        # Place in grid
        positions = [(0, 0), (0, w), (h, 0), (h, w)]
        for i, label in enumerate(label_order):
            if i >= 4:
                break
            y, x = positions[i]
            frame = frames.get(label, np.zeros_like(orig_frame))
            grid[y : y + h, x : x + w] = frame

            # Label background then text
            cv2.rectangle(
                grid,
                (x + 5, y + 5),
                (x + 15 + len(label) * 20, y + 40),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                grid,
                label,
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(grid)

    writer.release()


def write_metrics_csv(
    all_metrics: dict[str, list[FrameMetrics]],
    output_path: str,
) -> None:
    """Write per-frame metrics to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx", "tracker", "num_detections", "num_tracks",
            "track_ids", "process_time_ms",
        ])
        for tracker_name, metrics in all_metrics.items():
            for m in metrics:
                writer.writerow([
                    m.frame_idx,
                    m.tracker_name,
                    m.num_detections,
                    m.num_tracks,
                    ";".join(str(tid) for tid in m.active_track_ids),
                    f"{m.process_time_ms:.2f}",
                ])


def print_summary_table(all_metrics: dict[str, list[FrameMetrics]]) -> None:
    """Print a summary comparison table to stdout."""
    print("\n" + "=" * 70)
    print("TRACKER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Tracker':<12} | {'Avg FPS':>8} | {'Total Tracks':>12} | {'Avg Length':>10} | {'Fragments':>10}")
    print("-" * 70)

    for tracker_name, metrics in all_metrics.items():
        if not metrics:
            continue

        # Average FPS
        total_time_s = sum(m.process_time_ms for m in metrics) / 1000.0
        avg_fps = len(metrics) / total_time_s if total_time_s > 0 else 0

        # Track statistics
        all_track_ids: set[int] = set()
        track_first_seen: dict[int, int] = {}
        track_last_seen: dict[int, int] = {}

        for m in metrics:
            for tid in m.active_track_ids:
                all_track_ids.add(tid)
                if tid not in track_first_seen:
                    track_first_seen[tid] = m.frame_idx
                track_last_seen[tid] = m.frame_idx

        total_tracks = len(all_track_ids)

        # Average track length (in frames)
        if total_tracks > 0:
            track_lengths = [
                track_last_seen[tid] - track_first_seen[tid] + 1
                for tid in all_track_ids
            ]
            avg_length = sum(track_lengths) / len(track_lengths)
        else:
            avg_length = 0

        # Estimate fragmentation: count track IDs that appear in overlapping
        # spatial regions (simple heuristic)
        fragments = max(0, total_tracks - _estimate_unique_persons(metrics))

        print(f"{tracker_name:<12} | {avg_fps:>8.1f} | {total_tracks:>12} | {avg_length:>10.1f} | {fragments:>10}")

    print("=" * 70)


def _estimate_unique_persons(metrics: list[FrameMetrics]) -> int:
    """Rough estimate of unique persons based on max simultaneous tracks."""
    if not metrics:
        return 0
    return max(m.num_tracks for m in metrics)
