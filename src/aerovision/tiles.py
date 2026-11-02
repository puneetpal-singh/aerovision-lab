"""Explicit tile geometry and class-aware duplicate suppression."""

from __future__ import annotations


def starts(length: int, size: int, overlap: float) -> list[int]:
    if length <= 0 or size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Positive dimensions and 0 <= overlap < 1 required")
    last = max(0, length - size)
    step = max(1, round(size * (1 - overlap)))
    return sorted(set([*range(0, last + 1, step), last]))


def windows(width: int, height: int, size: int, overlap: float) -> list[tuple]:
    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in starts(height, size, overlap)
        for x in starts(width, size, overlap)
    ]


def predict(
    model,
    image,
    mode: str,
    imgsz: int,
    tile: int,
    overlap: float,
    conf: float,
    iou: float,
    device: str,
    max_det: int = 300,
):
    import torch
    from torchvision.ops import batched_nms

    h, w = image.shape[:2]
    crops = [(0, 0, w, h)] if mode == "whole" else windows(w, h, tile, overlap)
    boxes, scores, classes = [], [], []
    for x0, y0, x1, y1 in crops:
        result = model.predict(
            image[y0:y1, x0:x1],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            verbose=False,
        )[0]
        b = result.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.detach().clone()
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0
        boxes.append(xyxy)
        scores.append(b.conf.detach())
        classes.append(b.cls.detach().long())
    if not boxes:
        return [], len(crops)
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    if mode == "tiled":
        keep = batched_nms(boxes, scores, classes, iou)[:max_det]
    else:
        keep = scores.argsort(descending=True)[:max_det]
    rows = []
    for box, score, category in zip(
        boxes[keep].cpu().tolist(), scores[keep].cpu().tolist(), classes[keep].cpu().tolist()
    ):
        x0, y0, x1, y1 = box
        rows.append({"category_id": category, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": score})
    return rows, len(crops)
