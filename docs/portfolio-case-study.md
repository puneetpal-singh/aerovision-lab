# AeroVision Lab: project case study

## Engineering question

How much small-object accuracy can a compact aerial detector recover without increasing model size, and what latency cost does each inference strategy incur?

## Implementation

Built an independent experiment pipeline around Ultralytics YOLO11n: deterministic VisDrone subsets, annotation conversion and hash-based audit, hosted training, whole-image/tiled prediction, configurable detection caps, shared COCO-style scoring, fixed-confidence diagnostics, export checks and a local ONNX CPU demo. The project uses the upstream model and training engine rather than claiming a new neural architecture.

Development began with a 4 GB laptop GPU. After shutdowns under GPU load, training moved to Colab and evaluation to Kaggle. The final local demo runs exclusively on CPU. Training/evaluation notebooks embed source snapshots, verify input artifacts, save checkpoints and bundle portable results.

## Evidence and decisions

1. Small pilots established the pipeline before a full run. The 800-image, 60-epoch, batch-16 pilot improved over the initial 20-epoch batch-2 pilot, but this comparison changed two optimization factors and is not an isolated batch-size ablation.
2. Prediction-cap experiments showed only modest AP gains from increasing caps; a saturated low-confidence output did not mean that the cap was the principal accuracy bottleneck.
3. Full-data training completed in about 2 h 44 min on a T4. Increasing inference resolution on the saved model offered a useful gain without another training run.
4. On 548 validation images, whole-image 1280 achieved 21.73% AP at 16.4 ms median latency, versus 20.47% at 66.1 ms for the tested tiled setup. Tiling retained slightly higher small-object average recall.
5. The validation-selected 1280/0.30 configuration was frozen. On 1,610 held-out test-dev images, AP improved from 13.69% at 640 to 18.61% at 1280, and recall at confidence 0.30 improved from 33.21% to 46.68% at almost unchanged precision.
6. FP32 ONNX export and a lightweight CPU runtime were verified on a fixed image sample. This supports the demo while leaving full-set ONNX accuracy as an explicit unverified item.

## Practical limits

Scores remain modest for several rare/small-object classes. There is one training seed, no uncertainty analysis and a custom ignored-region approximation. Timings are within-run measurements rather than universal throughput claims. Test-dev is now inspected and cannot be reused as an untouched test for subsequent tuning.

## CV-ready wording

- Developed a reproducible aerial small-object detection pipeline with deterministic dataset preparation, hosted YOLO11n training, tiled inference, error analysis and CPU ONNX deployment.
- Evaluated a frozen model on 1,610 held-out images; increasing inference resolution from 640 to 1280 improved custom-protocol AP by 4.92 percentage points, with median T4 latency increasing from 12.3 to 17.1 ms.

## Interview preparation

Be ready to explain the preprocessing/NMS and coordinate transforms, why class-aware merging is required, why GPU memory use differs from GPU utilization, how validation settings were frozen, why aggregate F1 can hide weak classes, and why the reported scores are not official challenge scores. Explain failed hypotheses and resource constraints as well as successful results.
