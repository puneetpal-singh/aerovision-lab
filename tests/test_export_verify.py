from aerovision.export_verify import detection_agreement


def test_detection_order_does_not_change_agreement():
    a = [[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 1]]
    assert detection_agreement(a, list(reversed(a)))["passed"]


def test_missing_wrong_class_and_shifted_boxes_fail():
    a = [[0, 0, 10, 10, 0.9, 0]]
    assert not detection_agreement(a, [])["passed"]
    assert not detection_agreement(a, [[0, 0, 10, 10, 0.9, 1]])["passed"]
    assert not detection_agreement(a, [[1, 0, 11, 10, 0.9, 0]])["passed"]


def test_numpy_results_can_be_saved_as_json():
    import json
    import numpy as np

    rows = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
    json.dumps(detection_agreement(rows, rows))
