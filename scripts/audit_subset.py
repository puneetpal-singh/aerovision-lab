"""Write aggregate subset metadata without committing images or annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from aerovision.data import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/data-audit.json"))
    args = parser.parse_args()
    result = {"manifest_sha256": digest(args.subset / "manifest.json"), "splits": {}}
    for split in ("train", "val"):
        coco = json.loads((args.subset / f"{split}.coco.json").read_text())
        ignored = json.loads((args.subset / f"{split}.ignored.json").read_text())
        counts = Counter(a["category_id"] for a in coco["annotations"])
        result["splits"][split] = {
            "images": len(coco["images"]),
            "target_boxes": len(coco["annotations"]),
            "ignored_boxes": sum(len(r["boxes"]) for r in ignored),
            "boxes_per_class": {c["name"]: counts[c["id"]] for c in coco["categories"]},
            "width_range": [min(r["width"] for r in coco["images"]), max(r["width"] for r in coco["images"])],
            "height_range": [
                min(r["height"] for r in coco["images"]),
                max(r["height"] for r in coco["images"]),
            ],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
