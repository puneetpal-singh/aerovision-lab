"""Small GPU training and comparable COCO-style evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml


def environment() -> dict:
    import torch

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commit": commit,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("ultralytics", "torch", "torchvision", "numpy", "pycocotools")
        },
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
    }


def train(
    data: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    name: str,
    device: str,
    seed: int = 42,
    workers: int = 0,
    resume: str | None = None,
) -> Path:
    import torch
    from ultralytics import YOLO

    output = Path("runs") / name
    if output.exists() and not resume:
        raise FileExistsError(f"Run already exists: {output}; choose another name or --resume")
    config = dict(
        data=str(data.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        workers=workers,
        model="yolo11n.pt",
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        nbs=batch,
        amp=True,
        cache=False,
        deterministic=True,
        patience=epochs,
        close_mosaic=0,
        mosaic=0.5,
        scale=0.3,
        fliplr=0.5,
        project="runs",
        name=name,
        exist_ok=False,
        plots=True,
        save_period=1,
        max_det=300,
    )
    # nbs=batch disables implicit gradient accumulation: the requested batch is explicit.
    meta = {
        "environment": environment(),
        "requested": config,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path("runs").mkdir(exist_ok=True)
    pending = Path("runs") / f"{name}.metadata.json"
    pending.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = YOLO(resume or "yolo11n.pt")
    if resume:
        result = model.train(resume=True, device=device, workers=workers)
    else:
        result = model.train(**{k: v for k, v in config.items() if k != "model"})
    output = Path(result.save_dir)
    meta["seconds"] = time.perf_counter() - start
    if torch.cuda.is_available():
        meta["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (output / "experiment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"run": str(output), "seconds": meta["seconds"]}), flush=True)
    return output


def compare(
    data: Path,
    weights: Path,
    imgsz: int = 640,
    tile: int = 640,
    overlap: float = 0.2,
    device: str = "0",
    limit: int = 0,
    output: Path = Path("runs/comparison"),
) -> dict:
    import copy
    import cv2
    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO
    from .data import digest
    from .tiles import predict

    spec = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(spec["path"])
    truth = json.loads((root / "val.coco.json").read_text(encoding="utf-8"))
    images = truth["images"][:limit] if limit else truth["images"]
    ids = {r["id"] for r in images}
    truth["images"] = images
    truth["annotations"] = [a for a in truth["annotations"] if a["image_id"] in ids]
    # COCO crowd handling is an explicit approximation to VisDrone ignored regions.
    ignored = json.loads((root / "val.ignored.json").read_text(encoding="utf-8"))
    next_id = max((a["id"] for a in truth["annotations"]), default=0) + 1
    for row in ignored:
        if row["image_id"] not in ids:
            continue
        for box in row["boxes"]:
            for category in truth["categories"]:
                truth["annotations"].append(
                    dict(
                        id=next_id,
                        image_id=row["image_id"],
                        category_id=category["id"],
                        bbox=box,
                        area=box[2] * box[3],
                        iscrowd=1,
                    )
                )
                next_id += 1
    gt = COCO()
    gt.dataset = copy.deepcopy(truth)
    gt.createIndex()
    model = YOLO(str(weights))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": environment(),
        "weights_sha256": digest(weights),
        "manifest_sha256": digest(root / "manifest.json"),
        "protocol": "COCO-style AP, maxDets=300, crowd approximation for ignored regions; not official VisDrone scoring",
        "settings": dict(imgsz=imgsz, tile=tile, overlap=overlap, conf=0.001, iou=0.7, device=device),
        "image_ids": sorted(ids),
        "modes": {},
    }
    for mode in ("whole", "tiled"):
        rows, timings, counts = [], [], []
        # Warm up each mode on an actual image, outside timing.
        first = cv2.imread(str(root / "images/val" / images[0]["file_name"]))
        if first is None:
            raise FileNotFoundError(images[0]["file_name"])
        predict(model, first, mode, imgsz, tile, overlap, 0.001, 0.7, device)
        for record in images:
            image = cv2.imread(str(root / "images/val" / record["file_name"]))
            if image is None:
                raise FileNotFoundError(record["file_name"])
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.synchronize()
            start = time.perf_counter()
            predictions, n = predict(model, image, mode, imgsz, tile, overlap, 0.001, 0.7, device)
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            counts.append(n)
            rows.extend(dict(image_id=record["id"], **p) for p in predictions)
        (output / f"{mode}.predictions.json").write_text(json.dumps(rows), encoding="utf-8")
        if rows:
            detections = gt.loadRes(rows)
        else:
            detections = COCO()
            detections.dataset = {"images": images, "categories": truth["categories"], "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(gt, detections, "bbox")
        evaluator.params.imgIds = sorted(ids)
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()
        # The standard summary hardcodes maxDets=100 for AP; report our explicit
        # maxDets=300 slices instead of printing a misleading -1 headline.
        precision = evaluator.eval["precision"]

        def mean_valid(a):
            valid = a[a >= 0]
            return float(valid.mean()) if valid.size else None

        report["modes"][mode] = dict(
            ap=mean_valid(precision[:, :, :, 0, 2]),
            ap50=mean_valid(precision[0, :, :, 0, 2]),
            ap_small=mean_valid(precision[:, :, :, 1, 2]),
            recall_small=mean_valid(evaluator.eval["recall"][:, :, 1, 2]),
            ap_per_class={
                c["name"]: mean_valid(precision[:, :, k, 0, 2]) for k, c in enumerate(truth["categories"])
            },
            latency_p50_ms=float(np.median(timings)),
            latency_p95_ms=float(np.percentile(timings, 95)),
            mean_tiles=float(np.mean(counts)),
            detections=len(rows),
            images=len(images),
        )
        (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["modes"][mode], indent=2), flush=True)
    return report
