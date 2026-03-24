# ABOUTME: CLI entry point for HappyAnonymity face anonymization system
# ABOUTME: Provides process, compare, and detect subcommands

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="happyanon",
        description="Video face anonymization with SORT/DeepSORT/StrongSORT comparison",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- process --
    proc = subparsers.add_parser("process", help="Anonymize a video with a specific tracker")
    proc.add_argument("--input", required=True, help="Input video path")
    proc.add_argument("--output", required=True, help="Output video path")
    proc.add_argument(
        "--tracker",
        required=True,
        choices=["sort", "deepsort", "strongsort"],
        help="Tracking backend",
    )
    proc.add_argument("--conf-threshold", type=float, default=0.5)
    proc.add_argument("--blur-kernel", type=int, default=99)
    proc.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    proc.add_argument("--model-dir", default="models")
    proc.add_argument("--draw-boxes", action="store_true", help="Draw track ID boxes on output")

    # -- compare --
    comp = subparsers.add_parser("compare", help="Run all trackers and generate comparison")
    comp.add_argument("--input", required=True, help="Input video path")
    comp.add_argument("--output-dir", required=True, help="Output directory for comparison")
    comp.add_argument(
        "--trackers",
        default="sort,deepsort,strongsort",
        help="Comma-separated list of trackers",
    )
    comp.add_argument("--conf-threshold", type=float, default=0.5)
    comp.add_argument("--blur-kernel", type=int, default=99)
    comp.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    comp.add_argument("--model-dir", default="models")

    # -- detect --
    det = subparsers.add_parser("detect", help="Detection only, no tracking or blur")
    det.add_argument("--input", required=True, help="Input video path")
    det.add_argument("--output", required=True, help="Output video path")
    det.add_argument("--conf-threshold", type=float, default=0.5)
    det.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    det.add_argument("--model-dir", default="models")

    # -- benchmark --
    bench = subparsers.add_parser("benchmark", help="Benchmark throughput and VRAM per tracker")
    bench.add_argument("--input", required=True, help="Input video path")
    bench.add_argument(
        "--trackers",
        default="sort,deepsort,strongsort",
        help="Comma-separated list of trackers",
    )
    bench.add_argument("--max-frames", type=int, default=None, help="Limit frames (default: all)")
    bench.add_argument("--conf-threshold", type=float, default=0.5)
    bench.add_argument("--blur-kernel", type=int, default=99)
    bench.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    bench.add_argument("--model-dir", default="models")

    return parser


def cmd_process(args: argparse.Namespace) -> None:
    from .detector import FaceDetector
    from .pipeline import AnonymizationPipeline
    from .trackers import create_tracker

    tracker = create_tracker(args.tracker)
    needs_emb = tracker.needs_embeddings

    print(f"Initializing detector (embeddings={'yes' if needs_emb else 'no'})...")
    detector = FaceDetector(
        extract_embeddings=needs_emb,
        conf_threshold=args.conf_threshold,
        device=args.device,
        model_dir=args.model_dir,
    )

    pipeline = AnonymizationPipeline(
        detector=detector,
        tracker=tracker,
        blur_kernel=args.blur_kernel,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    metrics = pipeline.process_video(args.input, args.output, draw_boxes=args.draw_boxes)

    total_time = sum(m.process_time_ms for m in metrics) / 1000.0
    fps = len(metrics) / total_time if total_time > 0 else 0
    print(f"Done: {len(metrics)} frames, {fps:.1f} FPS, saved to {args.output}")


def cmd_compare(args: argparse.Namespace) -> None:
    from .comparison import create_comparison_video, print_summary_table, write_metrics_csv
    from .detector import FaceDetector
    from .pipeline import AnonymizationPipeline
    from .trackers import create_tracker

    tracker_names = [t.strip() for t in args.trackers.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input).stem

    all_metrics: dict[str, list] = {}
    tracker_videos: dict[str, str] = {}

    # We need embeddings if any tracker requires them
    needs_emb = any(create_tracker(n).needs_embeddings for n in tracker_names)

    print(f"Initializing detector (embeddings={'yes' if needs_emb else 'no'})...")
    detector = FaceDetector(
        extract_embeddings=needs_emb,
        conf_threshold=args.conf_threshold,
        device=args.device,
        model_dir=args.model_dir,
    )

    for tracker_name in tracker_names:
        tracker = create_tracker(tracker_name)
        output_path = str(output_dir / f"{input_stem}_{tracker_name}.mp4")

        pipeline = AnonymizationPipeline(
            detector=detector,
            tracker=tracker,
            blur_kernel=args.blur_kernel,
        )

        print(f"\n--- Running {tracker.name} ---")
        metrics = pipeline.process_video(args.input, output_path, draw_boxes=True)
        all_metrics[tracker.name] = metrics
        tracker_videos[tracker.name] = output_path

    # Generate comparison video
    grid_path = str(output_dir / f"{input_stem}_comparison.mp4")
    print(f"\nCreating comparison grid video...")
    create_comparison_video(args.input, tracker_videos, grid_path)

    # Write metrics CSV
    csv_path = str(output_dir / f"{input_stem}_metrics.csv")
    write_metrics_csv(all_metrics, csv_path)

    # Print summary
    print_summary_table(all_metrics)

    print(f"\nOutputs:")
    print(f"  Grid video:  {grid_path}")
    print(f"  Metrics CSV: {csv_path}")
    for name, path in tracker_videos.items():
        print(f"  {name}:      {path}")


def cmd_detect(args: argparse.Namespace) -> None:
    from .detector import FaceDetector
    from .pipeline import detection_only

    print("Initializing detector...")
    detector = FaceDetector(
        extract_embeddings=False,
        conf_threshold=args.conf_threshold,
        device=args.device,
        model_dir=args.model_dir,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detection_only(detector, args.input, args.output)
    print(f"Done: saved to {args.output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import benchmark_tracker, print_benchmark_table
    from .detector import FaceDetector
    from .trackers import create_tracker

    tracker_names = [t.strip() for t in args.trackers.split(",")]
    results = []

    for tracker_name in tracker_names:
        tracker = create_tracker(tracker_name)
        needs_emb = tracker.needs_embeddings

        print(f"\nInitializing detector for {tracker.name} (embeddings={'yes' if needs_emb else 'no'})...")
        detector = FaceDetector(
            extract_embeddings=needs_emb,
            conf_threshold=args.conf_threshold,
            device=args.device,
            model_dir=args.model_dir,
        )

        result = benchmark_tracker(
            detector=detector,
            tracker=tracker,
            input_path=args.input,
            blur_kernel=args.blur_kernel,
            max_frames=args.max_frames,
        )
        results.append(result)

    print_benchmark_table(results, args.device)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "process": cmd_process,
        "compare": cmd_compare,
        "detect": cmd_detect,
        "benchmark": cmd_benchmark,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
