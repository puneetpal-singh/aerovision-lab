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



"""CPU-only operating-point selection from saved predictions, not a new AP score."""

from collections import Counter, defaultdict




def sweep(truth, predictions, ignored, thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)):
    thresholds = sorted(set(thresholds))
    if not thresholds or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Thresholds must be in [0, 1]")
    image_ids = {r["id"] for r in truth["images"]}
    category_ids = {c["id"] for c in truth["categories"]}
    targets, proposed = defaultdict(list), defaultdict(list)
    support = Counter()
    for a in truth["annotations"]:
        if not a.get("iscrowd", 0):
            targets[a["image_id"]].append(a)
            support[a["category_id"]] += 1
    for p in predictions:
        if p["image_id"] not in image_ids or p["category_id"] not in category_ids:
            raise ValueError("Prediction image/category is absent from ground truth")
        proposed[p["image_id"]].append(p)
    ignored_by_image = {r["image_id"]: r["boxes"] for r in ignored}
    buckets = {t: defaultdict(Counter) for t in thresholds}
    for image_id in sorted(image_ids):
        # Greedy decisions for higher-scored predictions cannot depend on later,
        # lower-scored predictions. Match once, then take score prefixes.
        _, classified, _ = match_errors(
            targets[image_id], proposed[image_id], ignored_by_image.get(image_id, []), thresholds[0]
        )
        for p in classified:
            for threshold in thresholds:
                if p["score"] >= threshold:
                    buckets[threshold][p["category_id"]][p["kind"]] += 1
    rows = []
    for threshold in thresholds:
        per_class = {}
        for category in truth["categories"]:
            cid = category["id"]
            counts = buckets[threshold][cid]
            tp, fp, fn = counts["tp"], counts["fp"], support[cid] - counts["tp"]
            per_class[category["name"]] = dict(
                tp=tp,
                fp=fp,
                fn=fn,
                support=support[cid],
                precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn),
            )
        tp = sum(c["tp"] for c in per_class.values())
        fp = sum(c["fp"] for c in per_class.values())
        fn = sum(c["fn"] for c in per_class.values())
        supported = [c["f1"] for c in per_class.values() if c["support"]]
        rows.append(
            dict(
                confidence=threshold,
                precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn),
                micro_f1=2 * tp / max(1, 2 * tp + fp + fn),
                macro_f1=sum(supported) / max(1, len(supported)),
                per_class=per_class,
            )
        )
    return rows


# Run in the original Kaggle notebook: the downloaded summary ZIP has no predictions.
ROOT = Path('/kaggle/working/aerovision-lab')
SUBSET = ROOT / 'data/validation-v548-s42'
COMPARISON = ROOT / 'runs/whole-1280'
PREDICTIONS = COMPARISON / 'whole.predictions.json'
assert PREDICTIONS.exists(), (
    'Saved predictions are missing. Use the original Kaggle session. '
    'The resolution-results ZIP contains summaries, not predictions.'
)
comparison = json.loads((COMPARISON / 'comparison.json').read_text())
assert comparison['weights_sha256'] == '9ea4e947efe74f75703ab2c5a32bf5a6af70124945a6ee9bd366c7cd83172840'
assert comparison['settings']['imgsz'] == 1280
truth = json.loads((SUBSET / 'val.coco.json').read_text())
predictions = json.loads(PREDICTIONS.read_text())
ignored = json.loads((SUBSET / 'val.ignored.json').read_text())
print('Reviewing saved predictions on CPU; no model is loaded.', flush=True)
rows = sweep(truth, predictions, ignored)
summary = [{k: v for k, v in row.items() if k != 'per_class'} for row in rows]
print('Confidence | Precision | Recall | Micro F1 | Macro F1')
for row in summary:
    print(f"{row['confidence']:10.2f} | {row['precision']:9.1%} | {row['recall']:6.1%} | "
          f"{row['micro_f1']:8.1%} | {row['macro_f1']:8.1%}")
report = dict(protocol='Validation-only greedy same-class IoU>=0.5; crowd-overlap approximation; not COCO AP',
    image_count=len(truth['images']), weights_sha256=comparison['weights_sha256'],
    settings=comparison['settings'], rows=rows,
    micro_f1_candidate=max(summary, key=lambda r: r['micro_f1'])['confidence'],
    macro_f1_candidate=max(summary, key=lambda r: r['macro_f1'])['confidence'])
OUT = Path('/kaggle/working/aerovision-confidence-results.zip')
import zipfile
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('confidence.json', json.dumps(report, indent=2))
print('Download from the Kaggle files/output panel:', OUT)
