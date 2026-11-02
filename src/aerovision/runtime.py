"""Lightweight CPU ONNX inference; imports neither PyTorch nor CUDA."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def preprocess(rgb, size=1280, stride=32):
    height, width = rgb.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    pad_w, pad_h = (size - new_w) % stride / 2, (size - new_h) % stride / 2
    image = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(
        image,
        round(pad_h - 0.1),
        round(pad_h + 0.1),
        round(pad_w - 0.1),
        round(pad_w + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, ratio, (pad_w, pad_h)


def numpy_nms(boxes, scores, classes, iou=0.7, limit=1000):
    order = np.argsort(-scores, kind="stable")[:30000]
    kept = []
    while order.size and len(kept) < limit:
        chosen = int(order[0])
        kept.append(chosen)
        rest = order[1:]
        if not rest.size:
            break
        a, b = boxes[chosen], boxes[rest]
        wh = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]))
        intersection = wh[:, 0] * wh[:, 1]
        area_a = np.prod(np.maximum(0, a[2:] - a[:2]))
        area_b = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)
        union = area_a + area_b - intersection
        overlaps = intersection / np.maximum(union, 1e-12)
        order = rest[(classes[rest] != classes[chosen]) | (overlaps <= iou)]
    return np.array(kept, dtype=np.int64)


def decode(output, ratio, padding, original_size, confidence=0.30, iou=0.7, limit=1000):
    proposals = output[0].T
    classes = proposals[:, 4:].argmax(axis=1)
    scores = proposals[:, 4:].max(axis=1)
    valid = scores > confidence
    xywh, scores, classes = proposals[valid, :4], scores[valid], classes[valid]
    if not len(scores):
        return []
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    kept = numpy_nms(boxes, scores, classes, iou=iou, limit=limit)
    boxes = boxes[kept].copy()
    # Ultralytics scale_boxes uses rounded letterbox offsets on the inverse path.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - round(padding[0] - 0.1)) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - round(padding[1] - 0.1)) / ratio
    width, height = original_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return [
        dict(class_id=int(classes[k]), confidence=float(scores[k]), box_xyxy=[float(v) for v in box])
        for k, box in zip(kept, boxes)
    ]


class CpuDetector:
    def __init__(self, model_path: Path, config: dict):
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != config["onnx_sha256"]:
            raise ValueError("Model hash does not match the frozen configuration")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.config = config

    def predict(self, image: Image.Image, confidence=None):
        confidence = self.config["confidence"] if confidence is None else confidence
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        rgb = np.array(image.convert("RGB"))
        start = time.perf_counter()
        tensor, ratio, padding = preprocess(rgb, self.config["input_size"], self.config["stride"])
        output = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]
        detections = decode(
            output,
            ratio,
            padding,
            image.size,
            confidence,
            self.config["nms_iou"],
            self.config["max_detections"],
        )
        elapsed = (time.perf_counter() - start) * 1000
        for row in detections:
            row["label"] = self.config["classes"][row["class_id"]]
        return dict(
            detections=detections,
            latency_ms=elapsed,
            input_shape=list(tensor.shape),
            backend="ONNX Runtime CPU",
            confidence=confidence,
            model_version=self.config["version"],
        )


COLORS = [
    "#55D5CE",
    "#EABF68",
    "#A6B4FA",
    "#6ED99C",
    "#F18D91",
    "#82BCEC",
    "#D6A0DE",
    "#B4D978",
    "#F8AA71",
    "#7AD8DF",
]


def annotate(image, detections, labels=True):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    size = max(12, min(24, image.width // 90))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default(size=size)
    for row in reversed(detections):
        box = row["box_xyxy"]
        color = COLORS[row["class_id"] % len(COLORS)]
        draw.rectangle(box, outline=color, width=max(2, image.width // 700))
        if labels:
            text = f"{row['label']} {row['confidence']:.2f}"
            x, y = box[0], max(size + 4, box[1])
            bounds = draw.textbbox((x, y - size - 3), text, font=font)
            draw.rectangle(bounds, fill="#0B1220")
            draw.text((x, y - size - 3), text, fill=color, font=font)
    return result


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
