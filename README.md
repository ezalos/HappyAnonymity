# HappyAnonymity

Video face anonymization system that blurs faces using 3 different tracking backends for comparison.

## Tracking Backends

| Backend | Method | Re-ID | Source |
|---------|--------|-------|--------|
| **SORT** | Kalman filter + Hungarian algorithm | None (geometry only) | [abewley/sort](https://github.com/abewley/sort) |
| **DeepSORT** | SORT + appearance matching cascade | ArcFace 512-dim | [nwojke/deep_sort](https://github.com/nwojke/deep_sort) |
| **StrongSORT** | DeepSORT + NSA Kalman, EMA features, MC cost | ArcFace 512-dim | [dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT) |

All three share the same face detector (**RetinaFace**) via [InsightFace](https://github.com/deepinsight/insightface), so only the tracking step varies — making comparison fair.

## Pipeline

```
Video Frame → RetinaFace (detect faces) → ArcFace (extract embeddings)
           → Tracker (SORT | DeepSORT | StrongSORT)
           → Gaussian blur on tracked face regions
           → Output video
```

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ezalos/HappyAnonymity.git
cd HappyAnonymity

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Download face detection/recognition models (~275MB)
uv run python scripts/download_models.py
```

## Usage

### Process a single video

```bash
uv run python -m src process \
    --input data/input/walking_nyc.mp4 \
    --output data/output/walking_nyc_sort.mp4 \
    --tracker sort \
    --device cuda  # or cpu
```

### Compare all 3 trackers

```bash
uv run python -m src compare \
    --input data/input/walking_nyc.mp4 \
    --output-dir data/output/comparison/ \
    --device cuda
```

This generates:
- 3 anonymized videos (one per tracker, with bounding boxes + track IDs)
- A **2x2 comparison grid video** (original + 3 trackers side by side)
- A **metrics CSV** with per-frame tracking stats
- A **summary table** printed to stdout

### Detection only (debug)

```bash
uv run python -m src detect \
    --input data/input/walking_nyc.mp4 \
    --output data/output/detections.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tracker` | required | `sort`, `deepsort`, or `strongsort` |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--conf-threshold` | `0.5` | Face detection confidence threshold |
| `--blur-kernel` | `99` | Gaussian blur kernel size (must be odd) |
| `--draw-boxes` | off | Draw bounding boxes with track IDs |
| `--model-dir` | `models` | Directory for InsightFace model weights |

## Docker

```bash
# Build
docker compose build

# Process
docker compose run --rm anonymize process \
    --input data/input/walking_nyc.mp4 \
    --output data/output/result.mp4 \
    --tracker deepsort

# Compare
docker compose run --rm anonymize compare \
    --input data/input/walking_nyc.mp4 \
    --output-dir data/output/comparison/
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

## Example Results

Comparison on `walking_nyc.mp4` (10s street footage, CPU):

```
Tracker      |  Avg FPS | Total Tracks | Avg Length |  Fragments
----------------------------------------------------------------------
SORT         |      0.9 |           76 |       42.5 |         61
DeepSORT     |      0.9 |           57 |       56.9 |         42
StrongSORT   |      0.9 |           62 |       52.6 |         47
```

- **Fewer total tracks** = better re-identification (same person keeps same ID)
- **Longer avg track length** = more stable tracking through occlusions
- **Fewer fragments** = fewer identity switches

## Architecture

```
src/
├── cli.py              # CLI entry point (process/compare/detect)
├── pipeline.py         # Core detect → track → blur loop
├── detector.py         # RetinaFace + ArcFace via InsightFace
├── video_io.py         # OpenCV video read/write
├── blurring.py         # Gaussian blur on bounding box regions
├── comparison.py       # Grid video, metrics CSV, summary table
├── types.py            # FaceDetection, TrackedFace dataclasses
└── trackers/
    ├── base.py             # TrackerAdapter Protocol
    ├── sort_adapter.py     # SORT adapter
    ├── deepsort_adapter.py # DeepSORT adapter
    └── strongsort_adapter.py # StrongSORT adapter (with import isolation)
```

The adapter pattern normalizes the 3 trackers behind a unified `TrackerAdapter` interface. The pipeline code is backend-agnostic.

## License

Tracker backends have their own licenses:
- SORT: GPL-3.0
- DeepSORT: GPL-3.0
- StrongSORT: MIT
