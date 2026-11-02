import pytest
from aerovision.confidence import sweep
from aerovision.diagnostics import match_errors


def test_score_prefixes_equal_independent_matching():
    targets = [
        {"category_id": 0, "bbox": [0, 0, 10, 10], "image_id": 1},
        {"category_id": 1, "bbox": [20, 20, 10, 10], "image_id": 1},
    ]
    predictions = [
        dict(category_id=0, bbox=[0, 0, 10, 10], image_id=1, score=0.9),
        dict(category_id=0, bbox=[0, 0, 10, 10], image_id=1, score=0.6),
        dict(category_id=1, bbox=[20, 20, 10, 10], image_id=1, score=0.2),
    ]
    truth = dict(
        images=[{"id": 1}], annotations=targets, categories=[{"id": 0, "name": "a"}, {"id": 1, "name": "b"}]
    )
    rows = sweep(truth, predictions, [], thresholds=[0.1, 0.5, 0.8])
    for row in rows:
        counts, _, _ = match_errors(targets, predictions, [], row["confidence"])
        for kind in ["tp", "fp", "fn"]:
            assert sum(c[kind] for c in row["per_class"].values()) == counts[kind]
    assert rows[0]["recall"] == 1
    assert rows[-1]["precision"] == 1


def test_unknown_predictions_rejected():
    truth = dict(images=[{"id": 1}], annotations=[], categories=[{"id": 0, "name": "a"}])
    with pytest.raises(ValueError):
        sweep(truth, [dict(image_id=2, category_id=0)], [])
