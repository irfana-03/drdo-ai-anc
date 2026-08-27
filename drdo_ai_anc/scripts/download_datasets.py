#!/usr/bin/env python3
"""
download_datasets.py — Instructions and helpers for acquiring datasets.

This script does NOT download restricted datasets automatically.
Instead it:
    1. Prints clear acquisition instructions for each dataset.
    2. For freely available datasets (DEMAND, SONYC-UST on Zenodo),
       offers to download them if the user consents.
    3. Verifies existing downloads.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset demand
    python scripts/download_datasets.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Dataset download metadata
# ---------------------------------------------------------------------------

DATASETS = {
    "chime3": {
        "name": "CHiME-3",
        "auto_download": False,
        "instructions": (
            "CHiME-3 requires registration and a signed license agreement.\n"
            "1. Visit: https://www.chimechallenge.org/challenges/chime3\n"
            "2. Register and accept the license.\n"
            "3. Download the dataset.\n"
            "4. Extract into: data/raw/chime3/\n"
        ),
        "local_path": PROJECT_ROOT / "data" / "raw" / "chime3",
    },
    "demand": {
        "name": "DEMAND",
        "auto_download": True,
        "zenodo_url": "https://zenodo.org/record/1227121",
        "instructions": (
            "DEMAND is freely available under CC BY-SA 4.0.\n"
            "Download from: https://zenodo.org/record/1227121\n"
            "Extract into: data/raw/demand/\n"
        ),
        "local_path": PROJECT_ROOT / "data" / "raw" / "demand",
    },
    "sonyc_ust": {
        "name": "SONYC-UST",
        "auto_download": True,
        "zenodo_url": "https://zenodo.org/record/3966543",
        "instructions": (
            "SONYC-UST is freely available under CC BY 4.0.\n"
            "Download from: https://zenodo.org/record/3966543\n"
            "Extract into: data/raw/sonyc_ust/\n"
        ),
        "local_path": PROJECT_ROOT / "data" / "raw" / "sonyc_ust",
    },
}


def check_dataset(key: str) -> bool:
    """Return True if the dataset directory exists and contains files."""
    info = DATASETS[key]
    local = info["local_path"]
    if not local.exists():
        return False
    # Check for at least one audio file
    for ext in (".wav", ".flac", ".ogg", ".mp3"):
        if list(local.rglob(f"*{ext}")):
            return True
    return False


def print_instructions(key: Optional[str] = None) -> None:
    """Print acquisition instructions for one or all datasets."""
    targets = [key] if key else list(DATASETS.keys())
    for k in targets:
        info = DATASETS[k]
        present = check_dataset(k)
        status = "DOWNLOADED" if present else "NOT FOUND"
        print(f"\n{'='*60}")
        print(f"  {info['name']}  [{status}]")
        print(f"{'='*60}")
        print(info["instructions"])
        if present:
            print(f"  ✓ Found at: {info['local_path']}\n")
        else:
            print(f"  ✗ Expected at: {info['local_path']}\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="DRDO AI-ANC — Dataset acquisition helper"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Show instructions for a specific dataset.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which datasets are already present.",
    )
    args = parser.parse_args()

    if args.check:
        print("\n  Dataset presence check:")
        print(f"  {'Dataset':<15} {'Status'}")
        print(f"  {'-'*15} {'-'*20}")
        for k in DATASETS:
            ok = check_dataset(k)
            status = "FOUND" if ok else "MISSING"
            print(f"  {k:<15} {status}")
        print()
        return 0

    print_instructions(args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
