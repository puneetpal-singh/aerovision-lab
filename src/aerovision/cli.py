from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="AeroVision Lab small GPU pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print versions and GPU information")
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, default=Path("data"))
    prep.add_argument("--train", type=int, default=800)
    prep.add_argument("--val", type=int, default=200)
    prep.add_argument("--seed", type=int, default=42)
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--imgsz", type=int, default=640)
    fit.add_argument("--batch", type=int, default=2)
    fit.add_argument("--name", required=True)
    fit.add_argument("--device", default="0")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--resume", default=None, help="Path to last.pt")
    ev = sub.add_parser("compare")
    ev.add_argument("--data", type=Path, required=True)
    ev.add_argument("--weights", type=Path, required=True)
    ev.add_argument("--imgsz", type=int, default=640)
    ev.add_argument("--tile", type=int, default=640)
    ev.add_argument("--overlap", type=float, default=0.2)
    ev.add_argument("--device", default="0")
    ev.add_argument("--limit", type=int, default=0)
    ev.add_argument("--output", type=Path, default=Path("runs/comparison"))
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "doctor":
        from .experiment import environment
        print(json.dumps(environment(), indent=2))
    elif command == "prepare":
        from .data import prepare
        print(prepare(args["root"], args["train"], args["val"], args["seed"]))
    elif command == "train":
        from .experiment import train
        train(**args)
    else:
        from .experiment import compare
        compare(**args)


if __name__ == "__main__":
    main()
