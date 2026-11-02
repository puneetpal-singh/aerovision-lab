"""Exercise the static CPU browser demo using a local image (never upload image bytes)."""

import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173/demos/aerovision/index.html")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("work/web-smoke"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append((request.method, request.url)))
        page.goto(args.url)
        page.get_by_role("heading", name="A new perspective.").wait_for()
        assert page.locator("#confidence").input_value() == "0.55"
        assert page.locator("#detect").is_disabled()
        assert not any(".onnx" in url or ".wasm" in url for _, url in requests)
        page.screenshot(path=str(args.out / "home.png"), full_page=True)
        page.locator("#image-file").set_input_files(str(args.image.resolve()))
        page.locator("#detect").click()
        page.wait_for_function("!document.querySelector('#result').hidden", timeout=180000)
        assert page.locator("#status").inner_text().startswith(("Done.", "No objects"))
        page.screenshot(path=str(args.out / "detections.png"), full_page=True)
        with page.expect_download() as download:
            page.locator("#download-json").click()
        download.value.save_as(str(args.out / "predictions-055.json"))
        result = json.loads((args.out / "predictions-055.json").read_text())
        assert result["confidence"] == 0.55 and result["backend"].endswith("WASM CPU")
        assert result["detections"], "Expected detections for this fixed aerial sample"
        with page.expect_download() as download:
            page.locator("#download-image").click()
        download.value.save_as(str(args.out / "annotated.png"))
        assert (args.out / "annotated.png").stat().st_size > 1000
        page.locator("#confidence").focus()
        page.keyboard.press("Home")
        for _ in range(25):
            page.keyboard.press("ArrowRight")
        assert page.locator("#confidence").input_value() == "0.3"
        assert page.locator("#result").is_hidden()
        page.locator("#detect").click()
        page.wait_for_function("!document.querySelector('#result').hidden", timeout=180000)
        with page.expect_download() as download:
            page.locator("#download-json").click()
        download.value.save_as(str(args.out / "predictions-030.json"))
        lower = json.loads((args.out / "predictions-030.json").read_text())
        assert len(lower["detections"]) >= len(result["detections"])
        assert all(method in ("GET", "HEAD") for method, _ in requests), "Image must stay local"
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.out / "mobile.png"), full_page=True)
        page.locator("#image-file").set_input_files(
            dict(name="invalid.png", mimeType="image/png", buffer=b"not an image")
        )
        page.get_by_text("This file could not be read. Try another JPG or PNG.").wait_for()
        assert page.locator("#detect").is_disabled()
        assert not errors, errors
        browser.close()
    print(json.dumps({"passed": True, "detections_055": len(result["detections"]),
                      "detections_030": len(lower["detections"]), "image_upload_requests": 0,
                      "browser_errors": errors, "url": args.url}, indent=2))


if __name__ == "__main__":
    main()
