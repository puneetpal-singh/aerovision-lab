"""Sequential local smoke, pilot and comparison; progress survives chat disconnects."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = logs / "local-pilot-status.json"
    status = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": []}
    if os.name == "nt":
        # Keep this training process awake; the request ends when the process exits.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        small = "data/mini-t64-v32-s42/dataset.yaml"
        pilot = "data/mini-t800-v200-s42/dataset.yaml"
        steps = [
            ("smoke", ["train", "--data", small, "--epochs", "2", "--batch", "2", "--name", "local-smoke"]),
            ("pilot", ["train", "--data", pilot, "--epochs", "20", "--batch", "2", "--name", "local-pilot"]),
            (
                "compare",
                [
                    "compare",
                    "--data",
                    pilot,
                    "--weights",
                    "runs/local-pilot/weights/best.pt",
                    "--output",
                    "runs/local-comparison",
                ],
            ),
        ]
        for name, args in steps:
            record = {"step": name, "state": "running", "started": time.time()}
            status["steps"].append(record)
            status_path.write_text(json.dumps(status, indent=2))
            print(f"Starting {name}; log: logs/{name}.log", flush=True)
            with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
                env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
                result = subprocess.run(
                    [sys.executable, "-m", "aerovision.cli", *args],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            record.update(
                state="complete" if result.returncode == 0 else "failed",
                seconds=time.time() - record["started"],
                exit_code=result.returncode,
            )
            status_path.write_text(json.dumps(status, indent=2))
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect logs/{name}.log")
        status["state"] = "complete"
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
