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

## Train on free Google Colab

The laptop shuts down under GPU load, so training has moved to a hosted GPU.
The old local GPU runner is retired; CLI train/compare now default to CPU.

1. Download [AeroVision_Colab.ipynb](notebooks/AeroVision_Colab.ipynb).
2. Open https://colab.research.google.com/ and choose **File > Upload notebook**.
3. Choose **Runtime > Change runtime type > GPU** and use a hosted runtime.
4. Run the cells in order and approve Google Drive access when prompted.

This notebook embeds a verified source snapshot, so no GitHub token is required.
Keep it private. Dataset images stay on Colab scratch storage; run artifacts and
checkpoints save under `My Drive/AeroVisionLab/runs/` each epoch. Rerunning the
notebook rebuilds the same subset and resumes an unfinished saved checkpoint.
It also saves the exact source bundle in Drive. Free GPU access and runtime
lifetimes vary; this notebook has not yet been executed in your Colab account.

The pilot uses YOLO11n, 800 train / 200 validation images, 640 pixels, batch 2
and 20 epochs. It starts from COCO-pretrained weights. The earlier local smoke
completed; the local pilot was interrupted before recording a completed epoch.

## Evaluate the saved pilot without retraining

Upload [AeroVision_Evaluation.ipynb](notebooks/AeroVision_Evaluation.ipynb) to Colab
and select a hosted GPU. Mount the same Drive used by the pilot. This notebook
checks the checkpoint SHA-256 and the exact original image membership/content,
then compares prediction limits 300, 600 and 1000 with the evaluator limit fixed
at 1000. Detector output and global merge limits vary together; their individual
effects are not isolated. The 300 control is rescored under this new evaluator
limit, so compare within the new table. All results remain validation diagnostics.

Live subprocess output is streamed to the notebook. Fixed-confidence visual error
examples and aggregate greedy-match counts are separate from COCO AP metrics.
The final `colab-pilot-v1-evaluation-v2.zip` saves to Drive and downloads for review.

## CPU development

The local environment remains available for source edits and CPU tests. Do not
run GPU benchmarks or GPU training on this laptop. The tested historical Windows
environment is recorded in `docs/requirements-local.txt`; Colab retains its
preinstalled PyTorch and records the versions actually used in its run metadata.

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
