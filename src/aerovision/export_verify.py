"""CPU FP32 ONNX export with raw-output and common-postprocessing checks."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data import digest
from .diagnostics import overlap


def detection_agreement(reference, candidate, iou_min=0.99, score_atol=0.001):
    """Match [x1,y1,x2,y2,score,class] outputs without depending on output order."""
    unused = set(range(len(candidate)))
    minimum_iou, maximum_score_delta = 1.0, 0.0
    matched = 0
    for row in reference:
        box = [row[0], row[1], row[2] - row[0], row[3] - row[1]]
        choices = [i for i in unused if int(candidate[i][5]) == int(row[5])]

        def iou(i):
            other = candidate[i]
            return overlap(box, [other[0], other[1], other[2] - other[0], other[3] - other[1]])

        best = max(choices, key=iou, default=None)
        if best is None:
            continue
        value = iou(best)
        delta = abs(float(row[4]) - float(candidate[best][4]))
        minimum_iou = min(minimum_iou, value)
        maximum_score_delta = max(maximum_score_delta, delta)
        if value >= iou_min and delta <= score_atol:
            unused.remove(best)
            matched += 1
    return dict(
        passed=matched == len(reference) == len(candidate),
        reference_count=len(reference),
        candidate_count=len(candidate),
        matched=matched,
        minimum_iou=float(minimum_iou),
        maximum_score_delta=maximum_score_delta,
    )


def verify(weights: Path, images: list[Path], output: Path, expected_sha256: str):
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    from .experiment import environment

    assert digest(weights) == expected_sha256, "Unexpected source checkpoint"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    exported = Path(
        YOLO(str(weights)).export(
            format="onnx",
            imgsz=1280,
            batch=1,
            device="cpu",
            dynamic=True,
            half=False,
            simplify=False,
            opset=17,
            nms=False,
        )
    )
    import shutil

    artifact = output / "aerovision-full-fp32.onnx"
    if exported.resolve() != artifact.resolve():
        shutil.copy2(exported, artifact)
    onnx.checker.check_model(str(artifact))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
    reference = YOLO(str(weights)).model.float().eval().fuse(verbose=False).cpu()
    classes = len(reference.names)
    report = dict(
        environment=environment(),
        provider=session.get_providers(),
        onnx_version=onnx.__version__,
        onnxruntime_version=ort.__version__,
        weights_sha256=expected_sha256,
        onnx_sha256=digest(artifact),
        verification_code_sha256=digest(Path(__file__)),
        protocol="CPU FP32 export sanity check on fixed validation images; not full accuracy or GPU benchmark",
        preprocessing=dict(imgsz=1280, letterbox_auto=True, stride=32, color="RGB", scale="float32 / 255"),
        raw_tolerances=dict(rtol=0.0001, atol=0.001),
        postprocessing=dict(conf=0.30, iou=0.7, max_det=1000, match_iou=0.99, score_atol=0.001),
        images=[],
    )
    report["classes"] = reference.names
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        resized = LetterBox(new_shape=(1280, 1280), auto=True, stride=32)(image=image)
        array = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
        start = time.perf_counter()
        with torch.inference_mode():
            result = reference(tensor)
            result = result[0] if isinstance(result, (tuple, list)) else result
        reference_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        candidate = session.run(None, {session.get_inputs()[0].name: array})[0]
        candidate_ms = (time.perf_counter() - start) * 1000
        raw = result.cpu().numpy()
        raw_pass = raw.shape == candidate.shape and bool(np.allclose(raw, candidate, rtol=1e-4, atol=1e-3))
        delta = np.abs(raw - candidate) if raw.shape == candidate.shape else None
        ref_det = (
            non_max_suppression(result.clone(), conf_thres=0.30, iou_thres=0.7, nc=classes, max_det=1000)[0]
            .cpu()
            .numpy()
        )
        ort_det = non_max_suppression(
            torch.from_numpy(candidate.copy()), conf_thres=0.30, iou_thres=0.7, nc=classes, max_det=1000
        )[0].numpy()
        post = detection_agreement(ref_det, ort_det)
        record = dict(
            file_name=path.name,
            sha256=digest(path),
            input_shape=list(array.shape),
            raw_passed=raw_pass,
            raw_max_abs=float(delta.max()) if delta is not None else None,
            box_max_abs=float(delta[:, :4].max()) if delta is not None else None,
            score_max_abs=float(delta[:, 4:].max()) if delta is not None else None,
            detections=post,
            cpu_reference_ms=reference_ms,
            cpu_onnx_ms=candidate_ms,
        )
        report["images"].append(record)
        print(path.name, "raw:", raw_pass, "detections:", post["passed"], flush=True)
    report["synthetic_shapes"] = []
    rng = np.random.default_rng(42)
    for shape in [(1, 3, 640, 640), (1, 3, 1280, 1280), (1, 3, 960, 1280)]:
        array = rng.random(shape, dtype=np.float32)
        with torch.inference_mode():
            value = reference(torch.from_numpy(array))
            value = value[0] if isinstance(value, (tuple, list)) else value
        candidate = session.run(None, {session.get_inputs()[0].name: array})[0]
        raw = value.cpu().numpy()
        report["synthetic_shapes"].append(
            dict(
                input_shape=list(shape),
                passed=bool(np.allclose(raw, candidate, rtol=1e-4, atol=1e-3)),
                raw_max_abs=float(np.abs(raw - candidate).max()),
            )
        )
    report["passed"] = (
        bool(images)
        and all(r["passed"] for r in report["synthetic_shapes"])
        and all(r["raw_passed"] and r["detections"]["passed"] for r in report["images"])
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
