"""CPU-only operating-point selection from saved predictions, not a new AP score."""

from collections import Counter, defaultdict

from .diagnostics import match_errors


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
