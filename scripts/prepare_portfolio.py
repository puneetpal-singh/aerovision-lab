"""Copy the browser demo, verified model, and matching ORT assets to the portfolio."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("portfolio", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    portfolio = args.portfolio.resolve()
    config = json.loads((root / "configs/frozen-v1.json").read_text())
    model = root / "models/aerovision-full-fp32.onnx"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == config["onnx_sha256"]
    package = portfolio / "node_modules/onnxruntime-web"
    assert json.loads((package / "package.json").read_text())["version"] == "1.22.0"
    target = portfolio / "public/demos/aerovision"
    shutil.copytree(root / "web-demo", target, dirs_exist_ok=True)
    (target / "models").mkdir(exist_ok=True)
    shutil.copy2(model, target / "models" / f"aerovision-{config['onnx_sha256'][:12]}.onnx")
    vendor = target / "vendor/1.22.0"
    vendor.mkdir(parents=True, exist_ok=True)
    for name in ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]:
        shutil.copy2(package / "dist" / name, vendor / name)
    print("Prepared browser demo:", target)


if __name__ == "__main__":
    main()
