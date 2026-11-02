"""Reproduce CPU export verification from the full-data result bundle."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from aerovision.export_verify import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/export-verification"))
    args = parser.parse_args()
    expected = "9ea4e947efe74f75703ab2c5a32bf5a6af70124945a6ee9bd366c7cd83172840"
    scratch = Path("export-work")
    scratch.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.results) as archive:
        weights = archive.read("colab-full-v1-60epochs-b16/weights/best.pt")
        assert hashlib.sha256(weights).hexdigest() == expected
        (scratch / "best.pt").write_bytes(weights)
        manifest = json.loads(archive.read("full-data-manifest.json"))
    hashes = {r["file_name"]: r["sha256"] for r in manifest["splits"]["val"]}
    available = sorted(p for p in args.images.glob("*.jpg") if p.name in hashes)
    if len(available) < 10:
        raise ValueError("At least ten original validation images are required")
    images = [available[round(i * (len(available) - 1) / 9)] for i in range(10)]
    for image in images:
        assert hashlib.sha256(image.read_bytes()).hexdigest() == hashes[image.name]
    report = verify(scratch / "best.pt", images, args.out, expected)
    print("Verification passed:", report["passed"])
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
