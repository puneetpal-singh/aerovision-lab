"""Browser smoke check against an already-running local demo with a user-supplied image."""

import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("work/screenshots"))
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.get_by_role("heading", name="AeroVision Lab", exact=True).wait_for(timeout=30000)
        assert page.get_by_role("button", name="Detect objects", exact=True).is_disabled()
        page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for()
        page.screenshot(path=str(args.out / "home.png"), full_page=True)
        page.locator("input[type=file]").set_input_files(str(args.image.resolve()))
        page.get_by_role("button", name="Detect objects", exact=True).click()
        page.get_by_text("This CPU inference", exact=True).wait_for(timeout=60000)
        page.locator('[data-testid="stImage"] img').wait_for()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('[data-testid=stImage] img'))"
            ".every(img => img.complete && img.naturalWidth > 0)"
        )
        page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for()
        assert not page.locator("[data-testid=stException]").count()
        page.screenshot(path=str(args.out / "detections.png"), full_page=True)
        with page.expect_download() as download:
            page.get_by_role("button", name="Download predictions", exact=True).click()
        result_path = args.out / "predictions.json"
        download.value.save_as(str(result_path))
        result = json.loads(result_path.read_text())
        assert result["backend"] == "ONNX Runtime CPU" and result["confidence"] == 0.3
        print("Browser inference detections:", len(result["detections"]))
        page.get_by_role("tab", name="Evidence & limitations").click()
        page.get_by_role("heading", name="A measured improvement on held-out images").wait_for()
        page.get_by_text("Where it struggles", exact=True).wait_for()
        page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for()
        page.screenshot(path=str(args.out / "evidence.png"), full_page=True)
        page.get_by_role("tab", name="Image explorer").click()
        page.locator("input[type=file]").set_input_files(
            dict(name="invalid.png", mimeType="image/png", buffer=b"invalid image")
        )
        page.get_by_text("This file could not be read as an image. Try another JPG or PNG.").wait_for(
            timeout=15000
        )
        assert not errors, errors
        browser.close()
    print("Browser smoke checks passed.")


if __name__ == "__main__":
    main()
