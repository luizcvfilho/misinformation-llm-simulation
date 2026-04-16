from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_DATA_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
