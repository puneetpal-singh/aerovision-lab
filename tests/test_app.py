"""Initial UI should work without a model installed or any inference."""

from pathlib import Path
import pytest


def test_initial_demo_render():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "demo.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "AeroVision Lab"
    assert app.slider[0].value == 0.55
    assert app.button[0].disabled
    assert not app.tabs
    assert "aerial or drone" in app.get("file_uploader")[0].label
