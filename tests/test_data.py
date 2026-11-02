import pytest

from aerovision.data import parse_rows


def test_categories_clipping_and_ignored_regions():
    targets, ignored = parse_rows(
        "-5,10,20,30,1,1,0,0\n90,90,30,30,1,10,0,0\n0,0,10,10,0,0,0,0\n0,0,5,5,1,11,0,0", 100, 100
    )
    assert targets == [
        {"category_id": 0, "bbox": [0, 10, 15, 30]},
        {"category_id": 9, "bbox": [90, 90, 10, 10]},
    ]
    assert ignored == [[0, 0, 10, 10], [0, 0, 5, 5]]


def test_invalid_boxes_dropped_but_bad_rows_fail():
    assert parse_rows("110,0,5,5,1,1,0,0", 100, 100) == ([], [])
    with pytest.raises(ValueError):
        parse_rows("1,2,3", 100, 100)


def test_prepare_builds_traceable_disjoint_subset_without_download(tmp_path, monkeypatch):
    import io
    import json
    import zipfile
    from PIL import Image
    from aerovision import data

    archives = {}
    for split, base in (("train", 10), ("val", 100)):
        path = tmp_path / f"{split}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for index in range(3):
                buffer = io.BytesIO()
                Image.new("RGB", (40, 20), color=(base + index * 20, 0, 0)).save(buffer, format="JPEG")
                prefix = f"VisDrone2019-DET-{split}"
                archive.writestr(f"{prefix}/images/{index}.jpg", buffer.getvalue())
                archive.writestr(f"{prefix}/annotations/{index}.txt", "0,0,10,10,1,10,0,0")
        archives[split] = path
    monkeypatch.setattr(data, "download", lambda split, root: archives[split])
    config = data.prepare(tmp_path / "output", 2, 1)
    manifest = json.loads((config.parent / "manifest.json").read_text())
    assert len(manifest["splits"]["train"]) == 2
    assert len(manifest["splits"]["val"]) == 1
    assert len(manifest["classes"]) == 10
    labels = list((config.parent / "labels/train").glob("*.txt"))
    assert all(p.read_text().startswith("9 ") for p in labels)
    original = (config.parent / "manifest.json").read_bytes()
    assert data.prepare(tmp_path / "output", 2, 1) == config
    assert (config.parent / "manifest.json").read_bytes() == original
    validation = data.prepare_validation(tmp_path / "output", 1)
    validation_manifest = json.loads((validation.parent / "manifest.json").read_text())
    assert validation_manifest["splits"]["val"] == manifest["splits"]["val"]
