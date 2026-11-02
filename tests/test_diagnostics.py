from aerovision.diagnostics import match_errors


def test_duplicate_is_false_positive_and_ignored_region_is_not():
    gt = [{"category_id": 0, "bbox": [0, 0, 10, 10]}, {"category_id": 1, "bbox": [30, 30, 10, 10]}]
    predictions = [
        {"category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"category_id": 0, "bbox": [60, 60, 5, 5], "score": 0.7},
    ]
    counts, _, missed = match_errors(gt, predictions, [[60, 60, 10, 10]])
    assert counts == {"tp": 1, "fp": 1, "ignored": 1, "fn": 1}
    assert missed == [gt[1]]


def test_wrong_class_and_low_confidence_do_not_match_target():
    gt = [{"category_id": 0, "bbox": [0, 0, 10, 10]}]
    predictions = [
        {"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.1},
    ]
    counts, _, _ = match_errors(gt, predictions, [])
    assert counts == {"tp": 0, "fp": 1, "ignored": 0, "fn": 1}
