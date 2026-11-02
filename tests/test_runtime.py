import json
import numpy as np
import pytest

from aerovision.runtime import CpuDetector, decode, numpy_nms, preprocess


def test_rgb_letterbox_padding_and_normalization():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    tensor, ratio, padding = preprocess(image, size=32)
    assert tensor.shape == (1, 3, 32, 32)
    assert ratio == 1.6 and padding == (0, 8)
    assert tensor[0, :, 10, 10].tolist() == [1, 0, 0]
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)


def test_class_aware_nms_and_inverse_mapping():
    output = np.zeros((1, 14, 3), dtype=np.float32)
    output[0, :4] = np.array([[16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16]])
    output[0, 4, :2] = [0.9, 0.8]
    output[0, 5, 2] = 0.7
    rows = decode(output, 1.6, (0, 8), (20, 10))
    assert len(rows) == 2
    assert [r["class_id"] for r in rows] == [0, 1]
    assert rows[0]["box_xyxy"] == pytest.approx([5, 0, 15, 10])
    json.dumps(rows)


def test_empty_output_and_detection_limit():
    assert decode(np.zeros((1, 14, 1)), 1, (0, 0), (32, 32)) == []
    boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15]], dtype=float)
    assert numpy_nms(boxes, np.array([0.8, 0.9]), np.array([0, 0]), limit=1).tolist() == [1]


def test_model_integrity_checked_before_loading(tmp_path):
    path = tmp_path / "invalid.onnx"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="hash"):
        CpuDetector(path, {"onnx_sha256": "0" * 64})
