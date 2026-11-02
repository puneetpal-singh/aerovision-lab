# FP32 ONNX verification

The full-data YOLO11n checkpoint is exported at nominal image size 1280 with
dynamic spatial dimensions, opset 17, no graph simplification and no embedded
NMS. Export and verification run on CPU with `CUDA_VISIBLE_DEVICES=-1`.

Install the optional dependencies with `pip install -e '.[export]'`, then run:

```powershell
python scripts/verify_saved_export.py --results <full-results.zip> --images <original-validation-image-directory> --out runs/export-verification
```

The script checks the known checkpoint hash and selects ten evenly spaced images
from the sorted available original validation images, verifying each image hash
against the original manifest. Their filenames, hashes and input shapes are in
the report. Additional seeded random tensors check three other spatial shapes.

Preprocessing: letterbox with longest side 1280, `auto=True`, stride 32 and default
114 padding; convert BGR to RGB; transpose to NCHW; FP32 scaled by 1/255. The
tested real images all produced shape `[1, 3, 736, 1280]`. Only batch 1 is tested.

Raw PyTorch and ONNX outputs are checked with `rtol=1e-4`, `atol=1e-3`. Both
outputs then use the same CPU postprocessor: confidence 0.30, class-aware NMS
IoU 0.7, maximum 1000 detections. Final detections must agree in count and class,
with matched box IoU >= 0.99 and score difference <= 0.001. These tolerances were
set before inspecting the numerical results. This is not an independent test of
the shared postprocessor or a full validation-set accuracy measurement.

The ONNX output is `[batch, 14, anchors]`: four decoded centre-xywh box channels
in input-pixel coordinates followed by ten class scores, without a separate
objectness channel. Preprocessing, class-aware NMS and inverse letterbox mapping
to original-image coordinates remain application responsibilities. The class
mapping is stored in ONNX metadata and the verification report.

Example with Ultralytics handling preprocessing and postprocessing:

```python
from ultralytics import YOLO
model = YOLO('aerovision-full-fp32.onnx', task='detect')
results = model.predict('image.jpg', imgsz=1280, conf=0.30, iou=0.7,
                        max_det=1000, device='cpu')
```

Per-image CPU timings in the report are diagnostics with no warmup protocol;
they must not be compared with the prior warmed T4 inference measurements.
Passing this small export check does not establish parity on every image,
other precision modes, other batch sizes or other runtimes. Held-out test-dev
evaluation remains a separate step.
