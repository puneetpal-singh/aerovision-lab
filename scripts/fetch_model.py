"""Download the verified model from this private GitHub repository's release."""

import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "configs/frozen-v1.json").read_text())
    target = root / "models/aerovision-full-fp32.onnx"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == config["onnx_sha256"]:
        print("Verified model already installed:", target)
        return
    with tempfile.TemporaryDirectory(
        prefix="aerovision-", dir=root / "work" if (root / "work").exists() else root
    ) as tmp:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                "onnx-fp32-v0.1",
                "--repo",
                "puneetpal-singh/aerovision-lab",
                "--pattern",
                "aerovision-onnx-verified.zip",
                "--dir",
                tmp,
            ],
            check=True,
        )
        with zipfile.ZipFile(Path(tmp) / "aerovision-onnx-verified.zip") as archive:
            payload = archive.read("aerovision-full-fp32.onnx")
        if hashlib.sha256(payload).hexdigest() != config["onnx_sha256"]:
            raise ValueError("Downloaded model does not match frozen SHA-256")
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(payload)
    print("Verified model installed:", target)


if __name__ == "__main__":
    main()
