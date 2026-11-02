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

## Run locally

Use Python 3.12. Create a project-local environment, install a CUDA-enabled PyTorch
wheel appropriate to the GPU (see https://pytorch.org/get-started/locally/), then:

```powershell
python -m pip install -e ".[dev]"
python -m aerovision.cli doctor
python -m pytest
python -m aerovision.cli prepare --train 800 --val 200
python -m aerovision.cli train --data data/mini-t800-v200-s42/dataset.yaml --epochs 2 --batch 2 --name smoke
python -m aerovision.cli train --data data/mini-t800-v200-s42/dataset.yaml --epochs 20 --batch 2 --name pilot
python -m aerovision.cli compare --data data/mini-t800-v200-s42/dataset.yaml --weights runs/pilot/weights/best.pt
```

The initial model is YOLO11n, not YOLO11s. Each pilot is a new fine-tune from COCO
pretrained weights. The original 640/960/1280 study is a later milestone. Batch 2
is a starting point, not a memory-fit guarantee. Windows uses zero loader workers
by default. Lower batch to 1 if required, and record the deviation.

For a hosted GPU, open [the notebook](notebooks/aerovision_pilot.ipynb) in Colab or
Kaggle, enable GPU acceleration and run the cells. Private source access needs a
read-only GitHub token supplied at runtime. No credentials belong in Git or cell
outputs. Save checkpoints outside temporary notebook storage.

## Measurement protocol

- The seed-42 subset samples official train and validation independently, retains
  all ten target classes, and checks for identical image content across splits.
  This does not establish independence of visually similar scenes.
- Original archives and selected images have SHA-256 hashes in the local manifest.
- Ignored and other-object boxes are omitted from YOLO training targets. This is
  a limitation of this basic training conversion: they can be treated as background.
- Comparison uses pycocotools on the same images with maxDets=300, confidence
  0.001 and NMS IoU 0.7. Ignored regions are represented as per-class crowd boxes.
  This is a documented approximation, not the official VisDrone evaluator.
- Whole-image and tiled prediction use identical network input resolution and
  tile-local NMS settings. Tiled prediction adds global class-aware NMS. At overlap
  0.2, a 640-pixel tile advances about 512 pixels.
- Latency includes preprocessing, GPU inference, merging and result transfer, but
  excludes image-file decoding. GPU work is synchronized and each mode warmed up.
  Sequential tile inference favours low memory consumption over peak throughput.
- AP, AP50, small-object AP/recall, per-class AP and median/95th-percentile latency
  are written with predictions, weights hash, subset hash and versions.
- These are exploratory validation results; no test-dev has been downloaded or used.

Runs save checkpoints every epoch. To recover an interrupted run, restore its run
directory and use `train --data <yaml> --name <name> --resume <path/to/last.pt>`.
New runs reject an existing run directory to avoid silently overwriting results.

## Sources and scope

Independent implementation; uses the YOLO11 model and training engine from
[Ultralytics](https://docs.ultralytics.com/) and the dataset from the
[VisDrone team](https://github.com/VisDrone/VisDrone-Dataset). Source archives are
downloaded from the Ultralytics assets mirror. No dataset or weights are committed.
Upstream software and dataset terms apply to their respective components.
