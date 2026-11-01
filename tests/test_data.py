import pytest

from aerovision.data import parse_rows


def test_categories_clipping_and_ignored_regions():
    targets, ignored = parse_rows("-5,10,20,30,1,1,0,0\n90,90,30,30,1,10,0,0\n0,0,10,10,0,0,0,0\n0,0,5,5,1,11,0,0", 100, 100)
    assert targets == [{"category_id": 0, "bbox": [0, 10, 15, 30]},
                       {"category_id": 9, "bbox": [90, 90, 10, 10]}]
    assert ignored == [[0, 0, 10, 10], [0, 0, 5, 5]]


def test_invalid_boxes_dropped_but_bad_rows_fail():
    assert parse_rows("110,0,5,5,1,1,0,0", 100, 100) == ([], [])
    with pytest.raises(ValueError):
        parse_rows("1,2,3", 100, 100)
