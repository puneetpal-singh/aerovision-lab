# Held-out test-dev review

The predeclared full-data YOLO11n checkpoint was evaluated on 1,610 test-dev
images at 640 and 1280, without retraining or a test-set threshold search.
Checkpoint SHA-256: `9ea4e947efe74f75703ab2c5a32bf5a6af70124945a6ee9bd366c7cd83172840`.

| Metric | Whole 640 | Selected whole 1280 |
|---|---:|---:|
| AP, IoU 0.50:0.95 | 13.69% | 18.61% |
| AP50 | 24.02% | 32.36% |
| Small-object AP | 4.86% | 11.30% |
| Small-object average recall | 10.57% | 23.24% |
| Median latency | 12.34 ms | 17.12 ms |
| Precision at confidence 0.30 | 69.93% | 69.89% |
| Recall at confidence 0.30 | 33.21% | 46.68% |
| Micro F1 at confidence 0.30 | 45.03% | 55.98% |
| Macro F1 at confidence 0.30 | 28.73% | 36.72% |

The selected resolution improves AP by 4.92 percentage points and fixed-threshold
recall by 13.48 percentage points at almost unchanged aggregate precision. Median
latency rises by about 39%. Measurements are batch-one PyTorch inference on T4,
including preprocessing/postprocessing and transfer but excluding image-file
decoding. They are not complete application throughput measurements.

The 1280 model's AP declines from 21.73% on validation to 18.61% on test-dev
(3.12 percentage points). The resolution gain survives on held-out images.
All ten class AP values improve over the 640 baseline, although absolute results
remain uneven. At confidence 0.30, car recall is 77.4%, pedestrian recall 28.8%,
people recall 9.2%, bicycle recall 4.2%, and awning-tricycle recall 7.3%.

The fixed configuration is whole-image 1280, confidence 0.30 for operating-point
metrics, class-aware NMS IoU 0.7, prediction/evaluator limit 1000. AP integration
retains confidence 0.001. The earlier ONNX export check covered ten validation
images and three synthetic shapes, not full test-dev export parity.

Scoring uses pycocotools with the documented crowd approximation for ignored
regions. Fixed-threshold diagnostics use score-ordered greedy same-class matching
at IoU 0.5. These are not official VisDrone challenge scores. They come from one
training seed; statistical uncertainty and repeat-training variation are not
quantified. Exact-image hash checks do not rule out related scenes or near-duplicates.

Retain this checkpoint/configuration and these test results as the frozen version.
Any further model tuning after this review must disclose that test-dev has already
been inspected. Next work: package the reproducible experiments, model artifacts,
and a demo, with the above limitations visible.
