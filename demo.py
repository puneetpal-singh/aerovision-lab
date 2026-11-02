"""Local CPU-only demo entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from aerovision.app import main

main()
