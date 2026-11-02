"""Download original VisDrone archives and make deterministic small subsets.

Source: https://github.com/VisDrone/VisDrone-Dataset
Archive mirror: Ultralytics assets. No upstream conversion code is executed.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import yaml
from PIL import Image

NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
MIRROR = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_rows(text: str, width: int, height: int) -> tuple[list, list]:
    """Return valid targets and ignored regions, with boxes clipped to the image."""
    targets, ignored = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = [float(v.strip()) for v in line.split(",") if v.strip()]
        if len(row) < 8:
            raise ValueError(f"Expected eight VisDrone fields: {line}")
        x, y, w, h, score, category = row[:6]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        box = [x0, y0, x1 - x0, y1 - y0]
        if score == 0 or int(category) in (0, 11):
            ignored.append(box)
        elif 1 <= int(category) <= 10:
            targets.append({"category_id": int(category) - 1, "bbox": box})
        else:
            raise ValueError(f"Unknown category {category}")
    return targets, ignored


def download(split: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    filename = f"VisDrone2019-DET-{split}.zip"
    dest = root / filename
    if not dest.exists() or not zipfile.is_zipfile(dest):
        part = dest.with_suffix(".zip.part")
        print(f"Downloading {filename}", flush=True)
        with urllib.request.urlopen(f"{MIRROR}/{filename}", timeout=60) as response, part.open("wb") as f:
            downloaded = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                f.write(block)
                downloaded += len(block)
                if downloaded % (32 * 1024 * 1024) == 0:
                    print(f"{filename}: {downloaded // (1024 * 1024)} MiB", flush=True)
        if not zipfile.is_zipfile(part):
            raise ValueError(f"Invalid downloaded archive: {part}")
        part.replace(dest)
    with zipfile.ZipFile(dest) as archive:
        # Opening the central directory also rejects incomplete downloads.
        if not archive.namelist():
            raise ValueError(f"Empty archive: {dest}")
    return dest


def prepare(root: Path, train_count: int = 800, val_count: int = 200, seed: int = 42) -> Path:
    root = root.resolve()
    output = root / f"mini-t{train_count}-v{val_count}-s{seed}"
    if (output / "manifest.json").exists():
        print(f"Existing subset: {output}", flush=True)
        return output / "dataset.yaml"
    manifest = {
        "seed": seed,
        "classes": NAMES,
        "splits": {},
        "archives": {},
        "protocol": "YOLO targets omit ignore boxes; evaluation ignores crowd regions. Not official VisDrone metrics.",
    }
    for split, count in (("train", train_count), ("val", val_count)):
        archive_path = download(split, root / "archives")
        manifest["archives"][split] = {"name": archive_path.name, "sha256": digest(archive_path)}
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        records, annotations, ignored_records = [], [], []
        with zipfile.ZipFile(archive_path) as archive:
            images = sorted(n for n in archive.namelist() if "/images/" in n and n.lower().endswith(".jpg"))
            if not 0 < count <= len(images):
                raise ValueError(f"Requested {count} {split} images, available {len(images)}")
            chosen = sorted(random.Random(seed).sample(images, count))
            for image_id, member in enumerate(chosen, 1):
                filename = Path(member).name
                dest = image_dir / filename
                dest.write_bytes(archive.read(member))
                with Image.open(dest) as image:
                    width, height = image.size
                    image.verify()
                ann_member = member.replace("/images/", "/annotations/").rsplit(".", 1)[0] + ".txt"
                raw_text = archive.read(ann_member).decode("utf-8-sig")
                targets, ignored = parse_rows(raw_text, width, height)
                labels = []
                for target in targets:
                    x, y, w, h = target["bbox"]
                    labels.append(
                        f"{target['category_id']} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}"
                    )
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image_id,
                            **target,
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                (label_dir / Path(filename).with_suffix(".txt")).write_text(
                    "\n".join(labels), encoding="utf-8"
                )
                records.append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": width,
                        "height": height,
                        "sha256": digest(dest),
                    }
                )
                ignored_records.append({"image_id": image_id, "boxes": ignored})
        manifest["splits"][split] = records
        coco = {
            "info": {},
            "images": records,
            "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)],
        }
        (output / f"{split}.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        (output / f"{split}.ignored.json").write_text(json.dumps(ignored_records), encoding="utf-8")
        print(f"Prepared {split}: {len(records)} images, {len(annotations)} target boxes", flush=True)
    train_hashes = {r["sha256"] for r in manifest["splits"]["train"]}
    if any(r["sha256"] in train_hashes for r in manifest["splits"]["val"]):
        raise ValueError("Identical image content exists across train and validation")
    config = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(NAMES)),
    }
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
