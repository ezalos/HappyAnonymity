# HappyAnonymity - Face Anonymization System Design

## Context

Build a video face anonymization system that compares 3 tracking backends (SORT, DeepSORT, StrongSORT) using a shared face detector (RetinaFace) and re-id model (ArcFace). The system must run in Docker with GPU support, process videos via CLI, and produce comparison outputs proving correctness.

The repo starts empty with 2 WebM street footage videos.

## Architecture

### Pipeline

```
Video Frame → RetinaFace (face detection) → ArcFace (embedding extraction, optional)
           → Tracker (SORT | DeepSORT | StrongSORT)
           → Gaussian Blur on tracked face regions
           → Output Video
```

The face detector and embedder are shared across all trackers. Only the tracking step varies, making comparison fair.

### Adapter Pattern

A `TrackerAdapter` Protocol defines the interface:

```python
class TrackerAdapter(Protocol):
    def reset(self) -> None: ...
    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]: ...
    @property
    def name(self) -> str: ...
```

Each adapter normalizes its backend's API:
- **SortAdapter**: converts `FaceDetection` → `(N,5)` numpy `[x1,y1,x2,y2,score]`, calls `Sort.update()`, converts output `[x1,y1,x2,y2,track_id]` → `TrackedFace`
- **DeepSortAdapter**: converts `bbox_xyxy` → `tlwh`, creates `Detection(tlwh, confidence, embedding)`, calls `tracker.predict()` + `tracker.update()`, reads `tracker.tracks`
- **StrongSortAdapter**: same as DeepSORT adapter but uses StrongSORT's enhanced `deep_sort/` module with NSA Kalman, EMA features, MC cost

### Shared Data Types

```python
@dataclass
class FaceDetection:
    bbox_xyxy: np.ndarray    # [x1, y1, x2, y2]
    confidence: float
    embedding: np.ndarray | None  # 512-dim ArcFace, None for SORT

@dataclass
class TrackedFace:
    bbox_xyxy: np.ndarray    # [x1, y1, x2, y2]
    track_id: int
    confidence: float
```

## Backend Integration

### Git Submodules

The 3 tracker repos are added as git submodules under `vendor/`:

```
vendor/sort/        → https://github.com/abewley/sort
vendor/deep_sort/   → https://github.com/nwojke/deep_sort
vendor/strong_sort/  → https://github.com/dyhBUPT/StrongSORT
```

### Patches

**`patches/sort_headless.patch`**: Remove `matplotlib` and `skimage` top-level imports from `sort.py` (they crash in headless Docker, only used in `__main__` demo block).

**`patches/strongsort_compat.patch`**:
- Replace `from sklearn.utils.linear_assignment_ import linear_assignment` with `scipy.optimize.linear_sum_assignment` (deprecated in sklearn 0.24+)
- Replace `np.int` with `int` (removed in numpy 1.24)

Patches applied during Docker build.

### StrongSORT `opt` Global

StrongSORT reads config from a global `opt` namespace (argparse in `opts.py`). We inject it via monkey-patching before import:

```python
import sys, types
opts_module = types.ModuleType('opts')
opts_module.opt = types.SimpleNamespace(
    NSA=True, EMA=True, EMA_alpha=0.9,
    MC=True, MC_lambda=0.98, woC=True, ECC=False,
)
sys.modules['opts'] = opts_module
```

ECC disabled (requires pre-computed camera motion JSON per video).

## Face Detection & Embedding

InsightFace's `FaceAnalysis` handles both:
- **Detection-only mode** (for SORT): `allowed_modules=['detection']`
- **Detection + recognition mode** (for DeepSORT/StrongSORT): `allowed_modules=['detection', 'recognition']`

This avoids double-detecting. ArcFace produces 512-dim embeddings. DeepSORT's cosine metric works with any dimension (the original 128-dim was specific to mars-small128).

Configuration:
- `det_size=(640, 640)`, `conf_threshold=0.5`
- ONNX Runtime with CUDA execution provider

## Package Management: uv

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management.

- `pyproject.toml` declares dependencies and project metadata
- `uv.lock` pins exact versions (committed to git)
- `uv run` executes commands in the managed environment
- `uv sync` installs all dependencies

Key dependencies:
- `opencv-python-headless` (no X11 deps)
- `onnxruntime-gpu` (InsightFace inference)
- `numpy`, `scipy`, `filterpy`, `lap`, `insightface`, `tqdm`

## Docker

### Dockerfile

Base: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`

Install `uv` in the container, then use `uv sync` to install all dependencies from the lockfile. This ensures reproducible builds.

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git patch curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy vendored trackers + patches
COPY vendor/ ./vendor/
COPY patches/ ./patches/
RUN cd vendor/sort && patch -p1 < /app/patches/sort_headless.patch
RUN cd vendor/strong_sort && patch -p1 < /app/patches/strongsort_compat.patch

# Download models
COPY scripts/ ./scripts/
RUN uv run python scripts/download_models.py --model-dir /app/models

# Copy application code
COPY src/ ./src/

ENTRYPOINT ["uv", "run", "python", "-m", "src.cli"]
```

### docker-compose.yml

```yaml
services:
  anonymize:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    entrypoint: ["uv", "run", "python", "-m", "src.cli"]
```

## CLI Interface

```bash
# Process single video with one tracker
docker compose run anonymize process \
    --input data/input/video.webm \
    --output data/output/video_sort.mp4 \
    --tracker sort

# Compare all 3 trackers
docker compose run anonymize compare \
    --input data/input/video.webm \
    --output-dir data/output/comparison/

# Detection only (debug)
docker compose run anonymize detect \
    --input data/input/video.webm \
    --output data/output/detections.mp4
```

Options: `--tracker {sort,deepsort,strongsort}`, `--det-size`, `--conf-threshold`, `--blur-kernel`, `--device {cuda,cpu}`

## Comparison & Verification

### Visual: 2x2 Grid Video

```
┌──────────────┬──────────────┐
│   Original   │     SORT     │
├──────────────┼──────────────┤
│   DeepSORT   │  StrongSORT  │
└──────────────┴──────────────┘
```

Each quadrant shows bounding boxes with track IDs (color-coded) + blur applied. Tracker name overlaid.

### Metrics CSV

Per-frame: `frame_idx, tracker, num_detections, num_active_tracks, track_ids, processing_time_ms`

### Summary Table

```
Tracker     | Avg FPS | Tracks | Avg Length | Fragments (est.)
------------|---------|--------|------------|------------------
SORT        |   45.2  |   127  |    12.3    |       34
DeepSORT    |   28.7  |    89  |    23.1    |       12
StrongSORT  |   24.1  |    82  |    27.8    |        8
```

Metrics: total unique track IDs (fewer = better re-id), average track lifespan (longer = more stable), estimated fragmentation (ID reuse in same spatial region).

## Project Structure

```
HappyAnonymity/
├── data/
│   ├── input/          # Source videos (moved here from root)
│   └── output/         # Anonymized videos + comparisons
├── models/             # InsightFace model weights (gitignored)
├── vendor/             # Git submodules
│   ├── sort/
│   ├── deep_sort/
│   └── strong_sort/
├── src/
│   ├── __init__.py
│   ├── cli.py          # CLI entry point (argparse)
│   ├── pipeline.py     # Video processing loop
│   ├── detector.py     # RetinaFace + ArcFace wrapper
│   ├── blurring.py     # Gaussian blur on bbox regions
│   ├── video_io.py     # OpenCV video read/write
│   ├── types.py        # FaceDetection, TrackedFace dataclasses
│   ├── comparison.py   # Grid video + metrics + summary table
│   └── trackers/
│       ├── __init__.py
│       ├── base.py     # TrackerAdapter Protocol
│       ├── sort_adapter.py
│       ├── deepsort_adapter.py
│       └── strongsort_adapter.py
├── patches/
│   ├── sort_headless.patch
│   └── strongsort_compat.patch
├── scripts/
│   └── download_models.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml      # uv project definition + dependencies
├── uv.lock             # Pinned dependency lockfile
├── Makefile
├── .gitmodules
└── .gitignore
```

## Implementation Phases

### Phase 1: Foundation
1. Move videos to `data/input/`, set up `.gitignore`
2. `uv init` + configure `pyproject.toml` with all dependencies, `uv sync` to generate lockfile
3. Add 3 git submodules under `vendor/`
4. Create and test patches
5. Write `Dockerfile` and `docker-compose.yml`
6. Write `scripts/download_models.py`

### Phase 2: Core Components
7. `src/types.py` - dataclasses
8. `src/detector.py` - RetinaFace + ArcFace wrapper
9. `src/video_io.py` - OpenCV video read/write
10. `src/blurring.py` - Gaussian blur
11. Verify: detect faces on a single frame

### Phase 3: Tracker Adapters
12. `src/trackers/base.py` - Protocol
13. `src/trackers/sort_adapter.py` - simplest, test first
14. `src/trackers/deepsort_adapter.py`
15. `src/trackers/strongsort_adapter.py` (including `opt` injection)

### Phase 4: Pipeline + CLI
16. `src/pipeline.py` - orchestration
17. `src/cli.py` - `process` command
18. End-to-end test: one video with SORT

### Phase 5: Comparison + Verification
19. `src/comparison.py` - grid video, metrics CSV, summary table
20. Add `compare` command to CLI
21. Run full comparison on both videos
22. Generate and review metrics

### Phase 6: Polish
23. Add `detect` debug command
24. `Makefile` convenience targets
25. Final Docker build from clean state
