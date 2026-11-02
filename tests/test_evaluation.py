"""Check the scoring path on known geometry, including ignored detections."""

import json

import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.mark.parametrize("has_predictions,expected", [(True, 1), (False, 0)])
@pytest.mark.parametrize("max_det", [300, 600])
@pytest.mark.parametrize("split", ["val", "test-dev"])
def test_known_predictions_and_ignored_regions(
    tmp_path, monkeypatch, has_predictions, expected, max_det, split
):
    from aerovision import experiment, tiles
    from ultralytics import YOLO

    image_dir = tmp_path / "images" / split
    image_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(image_dir / "sample.jpg")
    truth = {
        "info": {},
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 0, "name": "target"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0}
        ],
    }
    (tmp_path / f"{split}.coco.json").write_text(json.dumps(truth))
    (tmp_path / f"{split}.ignored.json").write_text(
        json.dumps([{"image_id": 1, "boxes": [[60, 60, 30, 30]]}])
    )
    (tmp_path / "manifest.json").write_text("{}")
    data = tmp_path / "dataset.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path)}))
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not loaded")
    monkeypatch.setattr(YOLO, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(experiment, "environment", lambda: {})

    def predict(*args, **kwargs):
        if not has_predictions:
            return [], 1
        return [
            {"category_id": 0, "bbox": [65, 65, 10, 10], "score": 0.99},
            {"category_id": 0, "bbox": [10, 10, 10, 10], "score": 0.9},
        ], 1

    monkeypatch.setattr(tiles, "predict", predict)
    report = experiment.compare(
        data,
        weights,
        device="cpu",
        output=tmp_path / "output",
        max_det=max_det,
        eval_max_det=1000,
        split=split,
    )
    assert report["settings"]["max_det"] == max_det
    assert report["settings"]["eval_max_det"] == 1000
    for mode in report["modes"].values():
        assert mode["ap"] == pytest.approx(expected)
        assert mode["ap_small"] == pytest.approx(expected)
        assert mode["recall_small"] == pytest.approx(expected)
        assert mode["images"] == 1
