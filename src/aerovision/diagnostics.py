"""Operating-point error review; distinct from COCO AP evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def overlap(a, b, detection_area=False):
    x, y, w, h = a
    xx, yy, ww, hh = b
    intersection = max(0, min(x + w, xx + ww) - max(x, xx)) * max(0, min(y + h, yy + hh) - max(y, yy))
    denominator = w * h if detection_area else w * h + ww * hh - intersection
    return intersection / denominator if denominator > 0 else 0.0


def match_errors(targets, predictions, ignored, confidence=0.25, iou=0.5):
    """Score-ordered, class-aware greedy matching for visual diagnostics only."""
    unmatched = set(range(len(targets)))
    classified = []
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        if prediction["score"] < confidence:
            continue
        compatible = [i for i in unmatched if targets[i]["category_id"] == prediction["category_id"]]
        best = max(compatible, key=lambda i: overlap(prediction["bbox"], targets[i]["bbox"]), default=None)
        if best is not None and overlap(prediction["bbox"], targets[best]["bbox"]) >= iou:
            unmatched.remove(best)
            kind = "tp"
        elif any(overlap(prediction["bbox"], box, detection_area=True) >= 0.5 for box in ignored):
            kind = "ignored"
        else:
            kind = "fp"
        classified.append(dict(prediction, kind=kind))
    counts = {kind: sum(p["kind"] == kind for p in classified) for kind in ["tp", "fp", "ignored"]}
    counts["fn"] = len(unmatched)
    return counts, classified, [targets[i] for i in sorted(unmatched)]


def review(subset: Path, comparison: Path, confidence=0.25, examples=6):
    from PIL import Image, ImageDraw

    truth = json.loads((subset / "val.coco.json").read_text())
    ignored = {r["image_id"]: r["boxes"] for r in json.loads((subset / "val.ignored.json").read_text())}
    targets = defaultdict(list)
    for a in truth["annotations"]:
        targets[a["image_id"]].append(a)
    names = {c["id"]: c["name"] for c in truth["categories"]}
    reports = {
        "protocol": "Greedy same-class IoU>=0.5 at fixed confidence; not COCO AP matching",
        "confidence": confidence,
        "modes": {},
    }
    output = comparison / "error-review"
    output.mkdir(exist_ok=True)
    for mode in ["whole", "tiled"]:
        predictions = defaultdict(list)
        for p in json.loads((comparison / f"{mode}.predictions.json").read_text()):
            predictions[p["image_id"]].append(p)
        records = []
        for image in truth["images"]:
            counts, classified, missed = match_errors(
                targets[image["id"]], predictions[image["id"]], ignored.get(image["id"], []), confidence
            )
            records.append(dict(image, **counts, classified=classified, missed=missed))
        totals = {key: sum(r[key] for r in records) for key in ["tp", "fp", "fn", "ignored"]}
        totals["precision"] = totals["tp"] / max(1, totals["tp"] + totals["fp"])
        totals["recall"] = totals["tp"] / max(1, totals["tp"] + totals["fn"])
        reports["modes"][mode] = {
            "totals": totals,
            "images": [{k: v for k, v in r.items() if k not in ["classified", "missed"]} for r in records],
        }
        # Same image ids across modes; first examples are chosen by fixed validation ordering.
        for r in records[:examples]:
            image = Image.open(subset / "images/val" / r["file_name"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for item in r["classified"]:
                x, y, w, h = item["bbox"]
                color = {"tp": "lime", "fp": "red", "ignored": "gray"}[item["kind"]]
                draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                draw.text(
                    (x, max(20, y - 12)), f"{names[item['category_id']]} {item['score']:.2f}", fill=color
                )
            for target in r["missed"]:
                x, y, w, h = target["bbox"]
                draw.rectangle((x, y, x + w, y + h), outline="yellow", width=2)
            draw.rectangle((0, 0, image.width, 20), fill="black")
            draw.text(
                (4, 3),
                f"{mode}: TP green / FP red / missed yellow / ignored gray; conf={confidence}",
                fill="white",
            )
            image.save(output / f"{r['id']:03}-{mode}.jpg")
    (output / "errors.json").write_text(json.dumps(reports, indent=2))
    return reports
