# Video Processing Benchmark

This document is the durable record for CPU performance work on the video analysis pipeline.

## Benchmark Workload

- Video: `Emme-first-1000-frames.mp4`
- Video SHA-256: `b6480d95992a4c49323147fe9dcf86c621926358da0eb0c5078329b8e6570d64`
- Codec: VP9
- Resolution: 1280x720
- Frame rate: 30 FPS
- Frames: 1,000
- Duration: 33.333 seconds
- Models: ball tracking, action recognition, player detection, and jersey number detection

Model fingerprints:

| Model | SHA-256 |
| --- | --- |
| `action_recognition_yv11m.pt` | `05e55372458a191b9849df9d81295c918eb5bae8e0cbe4f5daf5a8984e62b85f` |
| `jersey_number_detection.pt` | `106f1fb4e888003bf9918e7bfa9d3cd35116198dfc2dba7ce0ef3b73a9b79f58` |
| `player_detection_yv8.pt` | `b9b77646cffdcad8b27360e68ac5f982fd1b6cabc9b24f4e91b39278c7f69d46` |
| `VballNetV1_seq9_grayscale_148_h288_w512.onnx` | `81ee27fe68d3e9b1e991e3d17ef529af85708609c482d3b6397bfbd9becda884` |

## Environment

- Date: 2026-08-09
- Machine: `Mac17,7`
- CPU cores: 18 physical, 18 logical
- Memory: 64 GiB
- Python: 3.11.15
- OpenCV: 5.0.0
- PyTorch: 2.13.0
- ONNX Runtime: 1.28.0
- Ultralytics: 8.4.117
- PyTorch intra-op threads after imports: 1; Ultralytics changes this to 8 when its first predictor initializes
- PyTorch inter-op threads: 18
- ONNX providers: CoreML, Azure, CPU

## Baseline

Code state:

- Git commit: `6fda786bd13dea3986f5d16343a86b9dcf59c487`
- The working tree already contained an uncommitted change in `ai_core/processor.py`: the Norfair tracker distance function was `mean_euclidean` instead of the committed `euclidean`.
- The backend detected MPS, but the YOLO prediction calls did not receive the selected device and therefore ran on CPU.

Command:

```bash
/usr/bin/time -lp ./venv/bin/python -c "from ai_core.processor import VolleyballAnalyzer; a=VolleyballAnalyzer(ball_model_path='models/VballNetV1_seq9_grayscale_148_h288_w512.onnx', action_model_path='models/action_recognition_yv11m.pt', player_model_path='models/player_detection_yv8.pt', jersey_number_model_path='models/jersey_number_detection.pt'); a.analyze_video('Emme-first-1000-frames.mp4', 'data/results/Emme-first-1000-frames-baseline.json')"
```

Performance:

| Metric | Baseline |
| --- | ---: |
| Analysis time | 232.79 s |
| Wall time | 234.47 s |
| Throughput | 4.30 frames/s |
| User CPU time | 590.54 s |
| System CPU time | 59.35 s |
| Average CPU use | 2.77 cores (15.4% of 18 cores) |
| Peak resident memory | 1.43 GB |

Average CPU use is `(user time + system time) / wall time`.

Output summary:

| Metric | Baseline |
| --- | ---: |
| Player boxes | 5,247 |
| Frames with tracked players | 997 |
| Ball detections after filtering | 665 |
| Ball trajectory points after interpolation | 997 |
| Action frame detections | 0 |
| Merged actions | 0 |
| Plays | 3 |

- Result: `data/results/Emme-first-1000-frames-baseline.json`
- Result SHA-256: `2e5b4b452f43029829750f571b61f911095f009f9042fbaf511a3bcd7afb5c08`

## Initial Findings

The current frame loop performs these stages serially:

1. Player YOLO inference and player tracking, including conditional jersey YOLO inference.
2. Ball ONNX inference.
3. Action YOLO inference.
4. Per-frame result aggregation.

Ranked hypotheses:

1. PyTorch CPU inference is constrained by its one-thread intra-op setting, leaving both YOLO models under-parallelized.
2. Player, ball, and action inference are independent within a frame but are serialized.
3. The selected `mps` device is not passed to Ultralytics prediction calls, despite the acceleration message.
4. Two jersey-model ROI inferences cause variable per-frame latency and late-run timing spikes.

## Experiments

### Stage Profile

The first 200 frames were copied without re-encoding and processed through the full pipeline. The stages account for 39.30 of 39.38 analysis seconds:

| Stage | Time | Share |
| --- | ---: | ---: |
| Player YOLO | 13.56 s | 34.4% |
| Tracking and jersey YOLO | 9.02 s | 22.9% |
| Ball ONNX | 3.78 s | 9.6% |
| Action YOLO | 12.94 s | 32.9% |

YOLO inference accounts for about 90% of processing time when jersey detection is included.

### PyTorch Thread Count

Ultralytics sets PyTorch to 8 intra-op threads when a predictor initializes. Tests after warming all three YOLO predictors produced these 200-frame analysis times:

| Threads | Analysis time | Result |
| ---: | ---: | --- |
| 1 | 58.60 s | 48.8% slower than the initial profile |
| 4 | 39.06-39.98 s | Approximately unchanged |
| 8 | 39.38 s | Existing Ultralytics default |
| 18 | 41.77 s | 6.1% slower than 8 threads |

Using all 18 cores inside each inference is slower. The model operations are too small to amortize the extra synchronization and memory pressure.

### Concurrent Stages

Running player YOLO, ball ONNX, and action YOLO concurrently reduced the 200-frame processing loop from 39.38 to 30.32 seconds and raised average use from 2.83 to 4.32 cores. The experiment was rejected because raw player detections changed from 833 to 829 and tracked boxes changed from 943 to 938. Concurrent PyTorch predictions did not preserve output determinism.

### YOLO Batching

Isolated player detection on the same 200 frames:

| Batch size | Time | Output versus batch 1 |
| ---: | ---: | --- |
| 1 | 13.14 s | Reference |
| 2 | 13.69 s | Exact |
| 4 | 9.16 s | Exact |
| 8 | 8.36 s | Exact |
| 16 | 39.58 s | Different and much slower |

Isolated action detection fell from 12.93 to 8.29 seconds at batch size 8 with identical output.

The production change batches only the independent full-frame player and action models. Ball inference, tracking, jersey detection, and result aggregation remain in frame order.

### Integrated 200-Frame Check

| Batch size | Analysis | Wall | Average cores | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 39.49 s | 41.03 s | 2.87 | 1.44 GB |
| 4 | 36.29 s | 37.87 s | 3.55 | 2.07 GB |
| 8 | 34.84 s | 36.42 s | 4.43 | 3.02 GB |

Batch 8 was 11.8% faster by analysis time. Its complete output was identical to batch 1 after removing only `analysis_time`.

## Optimized Full Benchmark

Command:

```bash
/usr/bin/time -lp ./venv/bin/python -c "from ai_core.processor import VolleyballAnalyzer; a=VolleyballAnalyzer(ball_model_path='models/VballNetV1_seq9_grayscale_148_h288_w512.onnx', action_model_path='models/action_recognition_yv11m.pt', player_model_path='models/player_detection_yv8.pt', jersey_number_model_path='models/jersey_number_detection.pt'); a.analyze_video('Emme-first-1000-frames.mp4', 'data/results/Emme-first-1000-frames-batch8.json')"
```

| Metric | Baseline | Batch 8 | Change |
| --- | ---: | ---: | ---: |
| Analysis time | 232.79 s | 226.12 s | 2.9% faster |
| Wall time | 234.47 s | 227.71 s | 2.9% faster |
| Throughput | 4.30 frames/s | 4.42 frames/s | 2.9% higher |
| User CPU time | 590.54 s | 866.71 s | 46.8% higher |
| System CPU time | 59.35 s | 47.67 s | 19.7% lower |
| Average CPU use | 2.77 cores | 4.02 cores | 45.1% higher |
| Peak RSS | 1.43 GB | 2.93 GB | 104.7% higher |

- Result: `data/results/Emme-first-1000-frames-batch8.json`
- Result SHA-256: `ca7603772b6ac47a6a5aeed7b8461155eddbee9c9a9e46f59453440bbd4e3a73`
- Correctness: every output field matches the baseline after removing only `analysis_time`.

A full batch-4 run took 247.58 seconds and was rejected. It was 6.4% slower than baseline despite using 3.44 cores on average.

The full batch-8 gain is smaller than its 200-frame gain. It is faster through approximately frame 600, after which jersey-heavy sections and sustained CPU/cache pressure consume most of the benefit. Higher utilization does not translate linearly to throughput for this workload.

## Follow-Up Findings

- Jersey scheduling says "every 5 frames" but checks `track_id % 5 == 0`. Tracks divisible by five run jersey inference every frame while other tracks never run it. Correcting this may change jersey output and should be benchmarked separately.
- This video produces no action detections, but the action model still consumes about one third of inference time. Skipping or sampling that model would change application behavior and was not done.
- The full processor test file currently has 63 passing and 5 failing tests. The new batching and device-forwarding tests pass; the five failures predate and are unrelated to this work (three stale NumPy mocks and two existing distance-match expectations).

## MPS Experiment

`VolleyballAnalyzer.get_optimal_device()` selected MPS, but the selected device was not passed into any Ultralytics prediction. Passing `self.device` to player, action, jersey, and YOLO ball-fallback inference activates the Apple GPU while leaving ONNX ball inference and stateful processing unchanged.

### 200 Frames

| Metric | CPU batch 8 | MPS batch 8 | Change |
| --- | ---: | ---: | ---: |
| Analysis time | 34.84 s | 12.52 s | 64.1% lower |
| Wall time | 36.42 s | 14.12 s | 61.2% lower |
| Peak RSS | 3.02 GB | 1.74 GB | 42.4% lower |

Detection counts, track identities, ball output, and action output were identical. MPS floating-point output differed from CPU by at most 0.000305 pixels for player bounding boxes and 0.00000251 for player confidence.

### Full Benchmark

| Metric | Original CPU | CPU batch 8 | MPS batch 8 |
| --- | ---: | ---: | ---: |
| Analysis time | 232.79 s | 226.12 s | 61.58 s |
| Wall time | 234.47 s | 227.71 s | 63.31 s |
| Throughput | 4.30 frames/s | 4.42 frames/s | 16.24 frames/s |
| Average CPU use | 2.77 cores | 4.02 cores | 4.00 cores |
| Peak RSS | 1.43 GB | 2.93 GB | 1.85 GB |

MPS is 3.78x faster than the original baseline and 3.67x faster than CPU batch 8. Output counts and all track identities are unchanged. The only output differences are bounded floating-point variations:

- Maximum player bounding-box delta: 0.000305 pixels
- Mean player bounding-box delta: 0.0000441 pixels
- Maximum player-confidence delta: 0.00000280
- All non-player-tracking output: identical after removing `analysis_time`

- Result: `data/results/Emme-first-1000-frames-mps-batch8.json`
- Result SHA-256: `918602a50f904ca06122c530feeb14878ce2bee02894a76e0b493ff212c4771a`

## Dependency Audit

The installed ML stack is already current according to the package index on 2026-08-09:

| Package | Python 3.11 environment | Python 3.14 environment |
| --- | ---: | ---: |
| PyTorch | 2.13.0 | 2.13.0 |
| torchvision | 0.28.0 | 0.28.0 |
| Ultralytics | 8.4.117 | 8.4.117 |
| OpenCV | 5.0.0.93 | 5.0.0.93 |
| ONNX Runtime | 1.28.0 | 1.28.0 |
| NumPy | 2.4.6 | 2.5.2 |
| Norfair | 2.1.1 | 2.1.1 |

`requirements.txt` does not reproduce either environment: it caps PyTorch below 2.9 and torchvision below 0.22. Restoring dependency reproducibility should be a separate change before testing library upgrades.

The web stack is substantially older:

| Package | Installed | Current |
| --- | ---: | ---: |
| FastAPI | 0.95.2 | 0.141.1 |
| Pydantic | 1.10.8 | 2.13.4 |
| Uvicorn | 0.22.0 | 0.52.1 |
| Celery | 5.2.7 | 5.6.3 |
| Redis client | 4.5.4 | 8.1.0 |
| Norfair | 2.1.1 | 2.3.0 |

Python 3.14 can run the AI core, but the backend cannot import because FastAPI 0.95.2 and Pydantic 1.10.8 are incompatible with Python 3.14. Warmed 200-frame MPS runs were effectively tied: 11.77 seconds on Python 3.11 and 11.83 seconds on Python 3.14. The interpreter upgrade does not provide a meaningful video-processing speedup because inference runs in PyTorch, Metal, ONNX Runtime, and OpenCV native code.

Recommendation:

1. Keep Python 3.11 for the MPS performance change.
2. In a separate modernization change, upgrade FastAPI and Pydantic together and add a reproducible lock file.
3. Validate the full API and worker suite before moving the supported runtime to Python 3.14.
4. Treat modernization as compatibility, security, and maintainability work rather than a video-performance optimization.
