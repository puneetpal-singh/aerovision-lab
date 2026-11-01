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
