# AeroVision Lab

Small-object detection experiments on VisDrone: establish a reproducible baseline, then investigate resolution, tiled inference and deployment.

This is an independent implementation started from scratch. No training results have been produced yet.

## Stage 1: reproduce the task

1. Validate the environment and GPU with a short smoke run.
2. Prepare VisDrone with explicit class mapping, split manifests and annotation checks.
3. Train a YOLO11s baseline at 640 pixels. Save resolved settings, dependency versions, seed, hardware and checkpoint hashes.
4. Evaluate on validation data; reserve test-dev for final evaluation after selecting the experiment protocol.
5. Compare 640, 960 and 1280 input resolutions with a controlled training setup. Choose a batch size that fits all compared runs and record any deviation.

The first goal is to reproduce the task and establish our own measurements. Matching another implementation's exact numbers is not assumed.

## Stage 2: improve

- Compare whole-image and overlapping tiled inference using consistent scoring and explicit suppression settings.
- Investigate errors near tile boundaries, overlap and duplicate merging.
- Report small-object recall, per-class AP, false positives, memory and latency alongside overall mAP.
- Verify exported model predictions and evaluate deployment performance.

Every improvement must be compared with a saved baseline on the same data and documented protocol.

## Status

Planning scaffold only. Dataset conversion, training and evaluation will be implemented independently. No GPU rental or training job has started.
