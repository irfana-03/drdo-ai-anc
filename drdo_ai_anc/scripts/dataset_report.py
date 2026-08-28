#!/usr/bin/env python3
"""
dataset_report.py — Print and save a report of real dataset statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_mapping import CONTEXT_CLASSES

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def main() -> int:
    meta_path = METADATA_DIR / "audio_metadata.csv"
    if not meta_path.exists():
        print("ERROR: audio_metadata.csv not found. Run verify_real_dataset.py first.")
        return 1

    df = pd.read_csv(meta_path)
    verify_path = METADATA_DIR / "verification_report.json"
    corrupt = 0
    duplicates = 0
    if verify_path.exists():
        with open(verify_path, "r", encoding="utf-8") as fh:
            vr = json.load(fh)
            corrupt = vr.get("corrupt_files", 0)
            duplicates = vr.get("duplicate_files", 0)

    report: dict = {
        "datasets": {},
        "total_recordings": len(df),
        "total_duration_s": round(df["duration"].sum(), 2),
        "total_duration_h": round(df["duration"].sum() / 3600, 2),
        "splits": {},
        "classes": {},
        "corrupt_files": corrupt,
        "duplicate_files": duplicates,
    }

    print("\n====================================")
    print("REAL DATASET REPORT")
    print("====================================\n")

    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        sr_vals = sub["sample_rate"].unique()
        sr_str = ", ".join(str(int(s)) for s in sr_vals)
        classes_in_ds = sub["mapped_context"].value_counts().to_dict()

        report["datasets"][ds] = {
            "recordings": len(sub),
            "duration_s": round(sub["duration"].sum(), 2),
            "sample_rates": sr_vals.tolist(),
            "classes": classes_in_ds,
        }

        print(f"Dataset: {ds}")
        print(f"  Number of recordings: {len(sub)}")
        print(f"  Total duration: {sub['duration'].sum()/3600:.2f} hours")
        print(f"  Sampling rate(s): {sr_str}")
        print(f"  Classes:")
        for c in CONTEXT_CLASSES:
            n = classes_in_ds.get(c, 0)
            if n > 0:
                print(f"    {c}: {n}")
        print()

    for split in ("train", "validation", "test"):
        n = len(df[df["split"] == split])
        report["splits"][split] = n
        print(f"{split.capitalize()} recordings: {n}")

    print(f"\nCorrupt files: {corrupt}")
    print(f"Duplicate files: {duplicates}")

    for c in CONTEXT_CLASSES:
        n = len(df[df["mapped_context"] == c])
        report["classes"][c] = n

    print("\n====================================")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dataset_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
