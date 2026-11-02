"""Validate the delivered notebook without executing GPU or cloud cells."""

import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "AeroVision_Colab.ipynb",
        "AeroVision_Evaluation.ipynb",
        "AeroVision_Kaggle.ipynb",
        "AeroVision_TestDev.ipynb",
    ],
)
def test_colab_cells_compile_and_bundle_matches_source(name):
    root = Path(__file__).resolve().parent.parent
    notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, source in enumerate(cells):
        compile(source, f"cell-{index}", "exec")
    payload = next(c for c in cells if "base64.b64decode(" in c)
    assert "sys.path.insert(0, str(PROJECT / 'src'))" in payload
    encoded = re.search(r"base64.b64decode\('([^']+)'\)", payload).group(1)
    data = base64.b64decode(encoded)
    sha = hashlib.sha256(data).hexdigest()
    assert f"== '{sha}'" in payload
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.namelist():
            assert not member.startswith("/") and ".." not in Path(member).parts
            packed = archive.read(member)
            assert b'\r\n' not in packed
            assert packed == (root / member).read_text(encoding="utf-8").encode("utf-8")
    for source in cells:
        if "run('train'" in source or "run('compare'" in source:
            assert "'--device', '0'" in source
