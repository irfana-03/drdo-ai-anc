#!/usr/bin/env python3
"""
train.py — Run the complete real-data ML pipeline end-to-end.

Steps:
    1. Download datasets
    2. Verify and build metadata
    3. Dataset report
    4. Train classifier
    5. Evaluate on test set
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(name: str, script: str) -> int:
    print(f"\n{'#'*60}")
    print(f"  STEP: {name}")
    print(f"{'#'*60}\n")
    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / script)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        return result.returncode
    print(f"\n  PASSED: {name}")
    return 0


def main() -> int:
    steps = [
        ("Download datasets", "scripts/download_datasets.py"),
        ("Verify real dataset", "scripts/verify_real_dataset.py"),
        ("Dataset report", "scripts/dataset_report.py"),
        ("Train classifier", "scripts/train_classifier.py"),
        ("Evaluate classifier", "scripts/evaluate_classifier.py"),
    ]

    for name, script in steps:
        rc = run_step(name, script)
        if rc != 0:
            # Allow verify to fail only if no data at all
            if "verify" in script.lower():
                print("  Verification failed — cannot continue.")
            return rc

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
