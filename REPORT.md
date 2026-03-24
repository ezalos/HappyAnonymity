# Benchmark Report

**Video:** `walking_nyc.mp4` (1080x1920, 60 FPS, 623 frames, 10.4s)
**GPU:** NVIDIA GeForce RTX 4090 (24,564 MB VRAM)
**Face Detector:** RetinaFace (InsightFace buffalo_l, det_size=640x640)
**Re-ID Model:** ArcFace w600k_r50 (512-dim embeddings, used by DeepSORT/StrongSORT)

---

## Throughput

| Tracker | FPS | ms/frame | Detection | Tracking | Blurring |
|------------|------:|----------:|-----------:|----------:|----------:|
| SORT | **36.4** | 27.5 ms | 7.2 ms | **1.1 ms** | 19.2 ms |
| DeepSORT | 12.7 | 79.0 ms | 20.6 ms | 22.8 ms | 35.6 ms |
| StrongSORT | 11.5 | 86.7 ms | 21.3 ms | 29.8 ms | 35.7 ms |

SORT is **3x faster** than DeepSORT/StrongSORT. The speed difference comes from two factors:

1. **No ArcFace inference** -- SORT uses geometry-only tracking (Kalman filter + Hungarian algorithm), so the detector runs in detection-only mode (7.2 ms vs 20.6 ms). DeepSORT and StrongSORT need the ArcFace recognition model to extract 512-dim face embeddings for re-identification.

2. **Minimal tracking overhead** -- SORT's tracking step costs 1.1 ms (pure numpy: Kalman predict + IoU-based Hungarian matching). DeepSORT adds cosine distance computation on appearance features and a matching cascade (22.8 ms). StrongSORT adds NSA Kalman filtering, EMA feature updates, and combined motion+appearance cost (29.8 ms).

Blurring is the largest time component for SORT (70% of frame time). It's CPU-bound Gaussian blur on 1080x1920 frames at kernel_size=99. For DeepSORT/StrongSORT, detection and tracking dominate instead.

## VRAM Usage

| Tracker | Baseline | Peak | Delta |
|------------|----------:|------:|-------:|
| SORT | 9,007 MB | 9,111 MB | **104 MB** |
| DeepSORT | 9,267 MB | 9,403 MB | **136 MB** |
| StrongSORT | 9,291 MB | 9,404 MB | **113 MB** |

VRAM differences are small because:

- **RetinaFace** (det_10g.onnx, ~16 MB) is the main GPU model for all three.
- **ArcFace** (w600k_r50.onnx, ~166 MB) is loaded additionally for DeepSORT/StrongSORT, adding ~30 MB of GPU memory overhead (the model is partially offloaded by ONNX Runtime).
- The tracking algorithms themselves (SORT, DeepSORT, StrongSORT) run entirely on CPU -- zero VRAM for tracking logic.

Note: baseline VRAM is ~9 GB due to other processes on the GPU. The face anonymization pipeline itself uses only 104-136 MB.

## Tracking Quality

| Tracker | Total Track IDs | Avg Track Length | Max Simultaneous | Fragments |
|------------|----------------:|-----------------:|-----------------:|----------:|
| SORT | 76 | 42.5 frames | 15 | 61 |
| DeepSORT | **57** | **56.9 frames** | 15 | **42** |
| StrongSORT | 62 | 52.6 frames | 15 | 47 |

- **Total Track IDs**: Lower is better -- means the tracker assigns fewer new IDs to the same person (better re-identification). DeepSORT wins with 57 vs SORT's 76.
- **Avg Track Length**: Higher is better -- means tracks persist longer through occlusions. DeepSORT maintains tracks for 56.9 frames on average vs SORT's 42.5.
- **Fragments**: Estimated number of identity switches. SORT has 61 fragments, DeepSORT only 42.

DeepSORT outperforms StrongSORT on this dataset because StrongSORT's enhancements (NSA Kalman, EMA features, MC cost, woC matching) were tuned for **full-body person re-identification** on MOT benchmarks. Here, we're tracking **faces** with ArcFace embeddings, which are already highly discriminative -- DeepSORT's simpler cosine matching cascade handles them well without the additional complexity.

## Time Breakdown

| Tracker | Detection | Tracking | Blurring |
|------------|----------:|----------:|---------:|
| SORT | 26.3% | 3.9% | **69.8%** |
| DeepSORT | 26.0% | 28.9% | 45.1% |
| StrongSORT | 24.5% | **34.4%** | 41.1% |

For all trackers, **detection + blurring** dominates. The tracking algorithm itself is never the main bottleneck. Optimization opportunities:

- **Blurring**: Move to GPU (cv2.cuda or custom kernel) for ~2x speedup.
- **Detection**: Use a lighter model (RetinaFace mobilenet) or reduce det_size.
- **Tracking**: Already fast; diminishing returns on optimization.

## Conclusion

| | SORT | DeepSORT | StrongSORT |
|---|---|---|---|
| **Best for** | Real-time, low-latency | Quality anonymization | Research/benchmarking |
| **FPS** | 36.4 | 12.7 | 11.5 |
| **VRAM** | 104 MB | 136 MB | 113 MB |
| **Track quality** | Adequate | Best | Good |
| **Complexity** | Trivial | Moderate | High |

**Recommendation**: DeepSORT offers the best balance of tracking quality and throughput for face anonymization. SORT is viable when real-time processing is required and occasional missed faces are acceptable. StrongSORT's additional complexity doesn't pay off for face tracking with ArcFace embeddings.
