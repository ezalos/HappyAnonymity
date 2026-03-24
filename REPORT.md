# Benchmark Report

**GPU:** NVIDIA GeForce RTX 4090 (24,564 MB VRAM)
**Face Detector:** RetinaFace (InsightFace buffalo_l, det_size=640x640)
**Re-ID Model:** ArcFace w600k_r50 (512-dim embeddings, used by DeepSORT/StrongSORT)

---

## Test Videos

| Video | Resolution | Duration | Frames | Conditions |
|-------|-----------|----------|--------|------------|
| `walking_nyc.mp4` | 1080x1920 | 10.4s | 623 | Daylight, well-lit street |
| `people_walking.mp4` | 1920x1080 | 116.6s | 3,493 | Dark / low-light, challenging |

---

## Throughput

### walking_nyc.mp4 (daylight, 623 frames)

| Tracker | FPS | ms/frame | Detection | Tracking | Blurring |
|------------|------:|----------:|-----------:|----------:|----------:|
| SORT | **36.4** | 27.5 ms | 7.2 ms | **1.1 ms** | 19.2 ms |
| DeepSORT | 12.7 | 79.0 ms | 20.6 ms | 22.8 ms | 35.6 ms |
| StrongSORT | 11.5 | 86.7 ms | 21.3 ms | 29.8 ms | 35.7 ms |

### people_walking.mp4 (dark, 200 frames sampled)

| Tracker | FPS | ms/frame | Detection | Tracking | Blurring |
|------------|------:|----------:|-----------:|----------:|----------:|
| SORT | **41.4** | 24.2 ms | 9.1 ms | **1.0 ms** | 14.0 ms |
| DeepSORT | 20.9 | 47.8 ms | 20.5 ms | 2.5 ms | 24.8 ms |
| StrongSORT | 21.4 | 46.8 ms | 20.4 ms | 2.3 ms | 24.2 ms |

SORT is **2-3x faster** than DeepSORT/StrongSORT. The speed difference comes from:

1. **No ArcFace inference** -- SORT uses geometry-only tracking, so the detector runs in detection-only mode (7-9 ms vs 20 ms). DeepSORT and StrongSORT need ArcFace to extract 512-dim face embeddings.
2. **Minimal tracking overhead** -- SORT's tracking costs ~1 ms (Kalman predict + IoU-based Hungarian matching). DeepSORT/StrongSORT add cosine distance computation and matching cascades (2-30 ms depending on scene density).

The dark video is actually faster: fewer faces detected per frame = less tracking and blurring work.

## VRAM Usage

| Tracker | Baseline | Peak | Delta |
|------------|----------:|------:|-------:|
| SORT | 9,007 MB | 9,111 MB | **104 MB** |
| DeepSORT | 9,267 MB | 9,403 MB | **136 MB** |
| StrongSORT | 9,291 MB | 9,404 MB | **113 MB** |

VRAM is consistent across videos (models are the same). The differences are small:

- **RetinaFace** (det_10g.onnx, ~16 MB) is the main GPU model for all three.
- **ArcFace** (w600k_r50.onnx, ~166 MB) adds ~30 MB of GPU overhead for DeepSORT/StrongSORT.
- Tracking algorithms run entirely on CPU -- zero VRAM for tracking logic.

Note: baseline VRAM is ~9 GB due to other processes. The pipeline itself uses only **104-136 MB**.

## Tracking Quality

### walking_nyc.mp4 (daylight, 623 frames)

| Tracker | Total Track IDs | Avg Track Length | Max Simultaneous | Fragments |
|------------|----------------:|-----------------:|-----------------:|----------:|
| SORT | 76 | 42.5 frames | 15 | 61 |
| DeepSORT | **57** | **56.9 frames** | 15 | **42** |
| StrongSORT | 62 | 52.6 frames | 15 | 47 |

### people_walking.mp4 (dark, 3,493 frames)

| Tracker | Total Track IDs | Avg Track Length | Max Simultaneous | Fragments |
|------------|----------------:|-----------------:|-----------------:|----------:|
| SORT | 340 | 31.4 frames | 9 | 331 |
| DeepSORT | **277** | **40.2 frames** | 9 | **268** |
| StrongSORT | 346 | 31.7 frames | 9 | 337 |

- **Total Track IDs**: Lower is better (same person keeps same ID). DeepSORT wins on both videos.
- **Avg Track Length**: Higher is better (tracks persist through occlusions). DeepSORT leads: 56.9 frames (daylight) and 40.2 frames (dark).
- **Fragments**: Estimated identity switches. DeepSORT has the fewest.

**Dark video degrades all trackers** -- avg track length drops from 42-57 frames to 31-40 frames. Face detection is less reliable in low light, causing more fragmented detections that the trackers can't recover from. DeepSORT degrades the most gracefully thanks to its appearance-based matching cascade.

StrongSORT **underperforms** on both datasets. Its enhancements (NSA Kalman, EMA features, MC cost) were tuned for full-body person re-identification on MOT benchmarks. For **face tracking** with ArcFace embeddings (which are already highly discriminative), DeepSORT's simpler cosine matching works better.

## Time Breakdown

### walking_nyc.mp4

| Tracker | Detection | Tracking | Blurring |
|------------|----------:|----------:|---------:|
| SORT | 26.3% | 3.9% | **69.8%** |
| DeepSORT | 26.0% | 28.9% | 45.1% |
| StrongSORT | 24.5% | **34.4%** | 41.1% |

### people_walking.mp4

| Tracker | Detection | Tracking | Blurring |
|------------|----------:|----------:|---------:|
| SORT | 37.8% | 4.1% | **58.1%** |
| DeepSORT | 42.9% | 5.2% | 51.9% |
| StrongSORT | 43.5% | 4.8% | 51.7% |

Key observations:

- **Blurring dominates SORT** (58-70%) because there's no ArcFace overhead.
- **The dark video shifts load to detection** (38-44% vs 24-26%) because RetinaFace works harder on low-contrast faces.
- **Tracking is cheap in the dark video** (2-5 ms) because fewer simultaneous faces means smaller cost matrices for Hungarian matching.

Optimization opportunities:
- **Blurring**: Move to GPU (cv2.cuda or custom kernel) for ~2x speedup.
- **Detection**: Use a lighter model (RetinaFace mobilenet) or reduce det_size.
- **Tracking**: Already fast; diminishing returns.

## Conclusion

| | SORT | DeepSORT | StrongSORT |
|---|---|---|---|
| **Best for** | Real-time, low-latency | Quality anonymization | Research/benchmarking |
| **FPS (daylight)** | 36.4 | 12.7 | 11.5 |
| **FPS (dark)** | 41.4 | 20.9 | 21.4 |
| **VRAM** | 104 MB | 136 MB | 113 MB |
| **Track quality (daylight)** | Adequate | **Best** | Good |
| **Track quality (dark)** | Poor | **Best** | Poor |
| **Complexity** | Trivial | Moderate | High |

**Recommendation**: **DeepSORT** offers the best balance of tracking quality and throughput for face anonymization. It degrades most gracefully in challenging conditions (low light). SORT is viable when real-time speed is critical and occasional missed faces are acceptable. StrongSORT's complexity doesn't pay off for face tracking with ArcFace embeddings.
