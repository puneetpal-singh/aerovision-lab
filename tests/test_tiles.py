import pytest

from aerovision.tiles import starts, windows


def test_every_pixel_covered_including_trailing_strips():
    width, height = 113, 79
    covered = set()
    for x0, y0, x1, y1 in windows(width, height, 32, 0.2):
        covered.update((x, y) for y in range(y0, y1) for x in range(x0, x1))
    assert len(covered) == width * height


def test_small_image_is_one_unpadded_tile():
    assert windows(20, 10, 640, 0.2) == [(0, 0, 20, 10)]
    assert starts(100, 32, 0)[-1] == 68


@pytest.mark.parametrize("overlap", [-0.1, 1, 2])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError):
        starts(100, 32, overlap)


def test_tiled_prediction_caps_output_and_preserves_overlapping_classes():
    from types import SimpleNamespace
    import numpy as np
    import torch
    from aerovision.tiles import predict

    class Boxes:
        xyxy = torch.tensor([[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0]])
        conf = torch.tensor([0.9, 0.8, 0.7])
        cls = torch.tensor([0, 0, 1])

        def __len__(self):
            return 3

    class Model:
        calls = []

        def predict(self, image, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(boxes=Boxes())]

    model = Model()
    rows, count = predict(
        model, np.zeros((32, 48, 3), dtype=np.uint8), "tiled", 32, 32, 0.5, 0.001, 0.7, "cpu", max_det=3
    )
    assert count == 2 and len(rows) == 3
    assert {r["category_id"] for r in rows} == {0, 1}
    assert any(r["bbox"][0] == 16 for r in rows)
    assert all(call["max_det"] == 3 for call in model.calls)
