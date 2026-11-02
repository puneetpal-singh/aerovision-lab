# AeroVision Lab model card

## Frozen v1

YOLO11n, ten VisDrone classes. COCO-pretrained initialization, fine-tuned on all 6,471 VisDrone2019-DET training images for 60 epochs at 640 pixels, batch 16, AdamW, seed 42. The selected inference configuration is whole-image 1280, confidence 0.30 and class-aware NMS IoU 0.7. AP evaluation retains confidence 0.001 and uses prediction/evaluator limits of 1000.

The authoritative model/settings hashes are in `configs/frozen-v1.json`. The app rejects a different ONNX binary rather than silently displaying benchmark results for another model.

## Data and selection

Validation: 548 images. Held-out test-dev: 1,610 images. Confidence and inference resolution were selected on validation and frozen before test-dev evaluation. Exact image-content hashes showed no overlap between train/validation and test-dev; near-duplicate scenes were not assessed. One trained model/seed was used.

## Intended use

A portfolio/research demonstration of small-object detection and the accuracy-latency trade-off in aerial imagery. The local demo supports CPU inference, exploration of confidence settings, image annotation and JSON export. UI uploads are held in session memory and are not explicitly saved by the app.

## Performance

Held-out 1280 AP: 18.61%; AP50: 32.36%; small-object AP: 11.30%. At confidence 0.30 and diagnostic IoU 0.5, aggregate precision was 69.89% and recall 46.68%. Median T4 PyTorch latency was 17.12 ms under the recorded protocol; CPU demo timings are separate.

Custom COCO-style evaluation with a crowd approximation is used. These are not official VisDrone challenge results. The diagnostic matching procedure differs from COCO AP matching.

## Failure modes

Tiny and occluded objects, class confusion, crowded scenes and changes in viewpoint or scene distribution can produce misses and false positives. At the frozen confidence, test-dev recall was 77.4% for cars, 28.8% for pedestrians, 9.2% for the dataset's `people` class, and 4.2% for bicycles. Aggregate scores therefore do not represent uniform performance across classes. Do not treat box predictions as ground truth.

## Export and demo scope

The exported ONNX graph uses FP32, dynamic spatial dimensions and external preprocessing/NMS. Export verification passed on ten validation images and three synthetic shapes. The independent NumPy demo preprocessing/postprocessing matched the reference path for 518 detections on those ten images. Full-set ONNX accuracy and other runtimes/precisions have not been established.

The demo uses ONNX Runtime CPU with two inference threads. It requires no GPU. Annotated images and predictions are downloaded only on user action. The default server binds to localhost. No public hosting or external image service is configured.

## Version policy

This checkpoint and its recorded test result are frozen v1. Further tuning after inspecting test-dev requires a new development version and disclosure that this split is no longer an untouched evaluation set for that development.
