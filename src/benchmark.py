# ABOUTME: Benchmarking module for measuring throughput and VRAM usage per tracker
# ABOUTME: Times detection, tracking, and blurring separately; measures peak GPU memory

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from .blurring import blur_faces
from .detector import FaceDetector
from .trackers.base import TrackerAdapter
from .types import FaceDetection
from .video_io import VideoReader


@dataclass
class BenchmarkResult:
    """Benchmark results for a single tracker run."""

    tracker_name: str
    total_frames: int
    total_time_s: float
    detection_time_s: float
    tracking_time_s: float
    blurring_time_s: float
    peak_vram_mb: float  # 0 if no GPU
    baseline_vram_mb: float  # VRAM before processing
    per_frame_times_ms: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.total_frames / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def avg_frame_ms(self) -> float:
        return (self.total_time_s / self.total_frames * 1000) if self.total_frames > 0 else 0

    @property
    def avg_detection_ms(self) -> float:
        return (self.detection_time_s / self.total_frames * 1000) if self.total_frames > 0 else 0

    @property
    def avg_tracking_ms(self) -> float:
        return (self.tracking_time_s / self.total_frames * 1000) if self.total_frames > 0 else 0

    @property
    def avg_blurring_ms(self) -> float:
        return (self.blurring_time_s / self.total_frames * 1000) if self.total_frames > 0 else 0

    @property
    def vram_used_mb(self) -> float:
        return max(0, self.peak_vram_mb - self.baseline_vram_mb)


def get_vram_mb() -> float:
    """Get current GPU memory usage in MB. Returns 0 if no GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return 0.0


def get_gpu_name() -> str:
    """Get GPU name. Returns 'N/A' if no GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetName(handle)
    except Exception:
        return "N/A"


def get_total_vram_mb() -> float:
    """Get total GPU memory in MB."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total / (1024 * 1024)
    except Exception:
        return 0.0


def benchmark_tracker(
    detector: FaceDetector,
    tracker: TrackerAdapter,
    input_path: str,
    blur_kernel: int = 99,
    max_frames: int | None = None,
) -> BenchmarkResult:
    """Benchmark a single tracker on a video.

    Measures detection, tracking, and blurring time separately,
    plus peak VRAM usage.
    """
    reader = VideoReader(input_path)
    tracker.reset()

    n_frames = min(reader.frame_count, max_frames) if max_frames else reader.frame_count

    # Force garbage collection and measure baseline VRAM
    gc.collect()
    baseline_vram = get_vram_mb()
    peak_vram = baseline_vram

    total_det_time = 0.0
    total_track_time = 0.0
    total_blur_time = 0.0
    per_frame_times = []

    for i, frame in enumerate(
        tqdm(reader, total=n_frames, desc=f"Benchmarking [{tracker.name}]")
    ):
        if max_frames and i >= max_frames:
            break

        t_frame_start = time.perf_counter()

        # Detection
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()
        total_det_time += t1 - t0

        # Tracking
        t0 = time.perf_counter()
        tracked = tracker.update(detections)
        t1 = time.perf_counter()
        total_track_time += t1 - t0

        # Blurring
        t0 = time.perf_counter()
        bboxes = [t.bbox_xyxy for t in tracked]
        blur_faces(frame, bboxes, blur_kernel)
        t1 = time.perf_counter()
        total_blur_time += t1 - t0

        frame_time = (time.perf_counter() - t_frame_start) * 1000
        per_frame_times.append(frame_time)

        # Sample VRAM periodically (every 10 frames to avoid overhead)
        if i % 10 == 0:
            current_vram = get_vram_mb()
            peak_vram = max(peak_vram, current_vram)

    # Final VRAM sample
    peak_vram = max(peak_vram, get_vram_mb())

    total_time = total_det_time + total_track_time + total_blur_time

    return BenchmarkResult(
        tracker_name=tracker.name,
        total_frames=min(n_frames, len(per_frame_times)),
        total_time_s=total_time,
        detection_time_s=total_det_time,
        tracking_time_s=total_track_time,
        blurring_time_s=total_blur_time,
        peak_vram_mb=peak_vram,
        baseline_vram_mb=baseline_vram,
        per_frame_times_ms=per_frame_times,
    )


def print_benchmark_table(results: list[BenchmarkResult], device: str) -> None:
    """Print a formatted benchmark comparison table."""
    gpu_name = get_gpu_name()
    total_vram = get_total_vram_mb()

    print()
    print("=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"Device: {device.upper()}" + (f" ({gpu_name}, {total_vram:.0f} MB)" if device == "cuda" else ""))
    if results:
        print(f"Frames: {results[0].total_frames}")
    print()

    # Throughput table
    print("--- Throughput ---")
    print(f"{'Tracker':<12} | {'FPS':>8} | {'ms/frame':>9} | {'Detect':>9} | {'Track':>9} | {'Blur':>9}")
    print("-" * 76)
    for r in results:
        print(
            f"{r.tracker_name:<12} | {r.fps:>8.1f} | {r.avg_frame_ms:>8.1f}ms | "
            f"{r.avg_detection_ms:>8.1f}ms | {r.avg_tracking_ms:>8.1f}ms | {r.avg_blurring_ms:>8.1f}ms"
        )

    # VRAM table
    if any(r.peak_vram_mb > 0 for r in results):
        print()
        print("--- VRAM Usage ---")
        print(f"{'Tracker':<12} | {'Baseline':>10} | {'Peak':>10} | {'Delta':>10}")
        print("-" * 52)
        for r in results:
            print(
                f"{r.tracker_name:<12} | {r.baseline_vram_mb:>8.0f}MB | "
                f"{r.peak_vram_mb:>8.0f}MB | {r.vram_used_mb:>8.0f}MB"
            )

    # Time breakdown as percentages
    print()
    print("--- Time Breakdown ---")
    print(f"{'Tracker':<12} | {'Detection':>10} | {'Tracking':>10} | {'Blurring':>10}")
    print("-" * 52)
    for r in results:
        if r.total_time_s > 0:
            det_pct = r.detection_time_s / r.total_time_s * 100
            trk_pct = r.tracking_time_s / r.total_time_s * 100
            blur_pct = r.blurring_time_s / r.total_time_s * 100
            print(
                f"{r.tracker_name:<12} | {det_pct:>9.1f}% | {trk_pct:>9.1f}% | {blur_pct:>9.1f}%"
            )

    print("=" * 90)
