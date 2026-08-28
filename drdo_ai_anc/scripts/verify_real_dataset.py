#!/usr/bin/env python3
"""
verify_real_dataset.py — Scan real audio, validate integrity, build metadata,
and create recording-level train/validation/test splits.

Outputs:
    data/metadata/audio_metadata.csv
    data/metadata/label_mapping.csv
    data/metadata/train.csv
    data/metadata/validation.csv
    data/metadata/test.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

# Ensure project root on path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_loader import SUPPORTED_EXTENSIONS, scan_audio_files
from src.data.label_mapping import (
    CONTEXT_CLASSES,
    get_all_mapping_entries,
    map_chime3_label,
    map_demand_label,
    map_sonyc_coarse_labels,
)

logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def _load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _file_hash(path: Path, nbytes: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def _validate_audio_file(path: Path) -> Dict[str, Any]:
    """Validate a single audio file; return metadata or error info."""
    result: Dict[str, Any] = {
        "file_path": str(path),
        "valid": False,
        "error": None,
    }
    try:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError("File missing or empty")

        info = sf.info(str(path))
        data, _ = sf.read(str(path), dtype="float32")

        if data.size == 0:
            raise ValueError("Zero samples")
        if np.any(np.isnan(data)):
            raise ValueError("Contains NaN")
        if np.any(np.isinf(data)):
            raise ValueError("Contains Inf")

        channels = 1 if data.ndim == 1 else data.shape[1]
        duration = info.frames / info.samplerate

        result.update(
            {
                "valid": True,
                "duration": round(duration, 4),
                "sample_rate": info.samplerate,
                "channels": channels,
                "num_samples": info.frames,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result


def _demand_environment_from_path(path: Path) -> str:
    """Extract DEMAND environment code from path or filename."""
    for part in path.parts:
        upper = part.upper()
        for env in (
            "NFIELD", "NRIVER", "NPARK", "OOFFICE", "OHALLWAY", "OMEETING",
            "DKITCHEN", "DLIVING", "DWASHING", "PCAFETER", "PRESTO", "SCAFE",
            "PSTATION", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO",
        ):
            if env in upper:
                return env
    # Fallback: first token of stem
    return path.stem.split("_")[0].upper()


def _should_include_demand_file(path: Path) -> bool:
    """Use only channel 1 per environment to avoid 16× duplication."""
    name = path.stem.lower()
    if "ch" in name:
        return bool(re.match(r"ch0?1$", name))
    return True


def _scan_demand() -> List[Dict[str, Any]]:
    demand_dir = RAW_DIR / "demand"
    rows: List[Dict[str, Any]] = []
    if not demand_dir.exists():
        return rows

    seen_env_ch1: Set[str] = set()
    for fp in scan_audio_files(demand_dir):
        if not _should_include_demand_file(fp):
            continue

        env = _demand_environment_from_path(fp)
        if env in seen_env_ch1:
            continue
        seen_env_ch1.add(env)

        mapping = map_demand_label(env)
        if mapping is None:
            continue

        rows.append(
            {
                "dataset": "demand",
                "file_path": str(fp),
                "recording_id": f"demand_{env}",
                "original_label": env,
                "mapped_context": mapping.mapped_context,
            }
        )
    return rows


def _sonyc_coarse_columns(columns: List[str]) -> List[str]:
    """Return SONYC coarse-level presence columns (exclude fine-level)."""
    coarse = []
    for c in columns:
        if not c.endswith("_presence"):
            continue
        base = c[: -len("_presence")]
        # Fine-level columns start with digit-digit pattern, e.g. 1-1_...
        if re.match(r"^\d+-\d+", base):
            continue
        coarse.append(c)
    return coarse


def _load_sonyc_annotations() -> Dict[str, Dict[str, Any]]:
    """Build filename -> {coarse_labels, sensor_id, split} from annotations.csv."""
    ann_path = RAW_DIR / "sonyc_ust" / "annotations.csv"
    if not ann_path.exists():
        return {}

    df = pd.read_csv(ann_path, low_memory=False)
    coarse_cols = _sonyc_coarse_columns(df.columns.tolist())

    ann_map: Dict[str, Dict[str, Any]] = {}

    for fname, group in df.groupby("audio_filename"):
        # Prefer ground truth (annotator_id == 0)
        gt_rows = group[group["annotator_id"] == 0]
        rows = gt_rows if len(gt_rows) > 0 else group[group["annotator_id"] > 0]

        present: List[str] = []
        for col in coarse_cols:
            vals = rows[col].replace(-1, np.nan).dropna()
            if len(vals) == 0:
                continue
            # Majority vote for citizen science; any positive for ground truth
            if (vals == 1).sum() > len(vals) / 2 or (gt_rows.shape[0] > 0 and (vals == 1).any()):
                parts = col.replace("_presence", "").split("_", 1)
                if len(parts) == 2:
                    present.append(parts[1])

        first = group.iloc[0]
        ann_map[str(fname)] = {
            "coarse_labels": present,
            "sensor_id": str(first.get("sensor_id", "unknown")),
            "original_split": str(first.get("split", "")),
        }
    return ann_map


def _scan_sonyc() -> List[Dict[str, Any]]:
    sonyc_dir = RAW_DIR / "sonyc_ust"
    rows: List[Dict[str, Any]] = []
    if not sonyc_dir.exists():
        return rows

    ann_map = _load_sonyc_annotations()
    for fp in scan_audio_files(sonyc_dir):
        fname = fp.name
        ann = ann_map.get(fname, {})
        coarse = ann.get("coarse_labels", [])
        mapping = map_sonyc_coarse_labels(coarse)
        sensor = ann.get("sensor_id", fp.parent.name)

        rows.append(
            {
                "dataset": "sonyc_ust",
                "file_path": str(fp),
                "recording_id": f"sonyc_{sensor}_{fname}",
                "original_label": ",".join(coarse) if coarse else "unlabeled",
                "mapped_context": mapping.mapped_context,
                "sonyc_split": ann.get("original_split", ""),
            }
        )
    return rows


def _scan_chime3() -> List[Dict[str, Any]]:
    chime_dir = RAW_DIR / "chime3"
    rows: List[Dict[str, Any]] = []
    if not chime_dir.exists():
        return rows

    for fp in scan_audio_files(chime_dir):
        env = fp.parent.name.upper()
        mapping = map_chime3_label(env)
        rows.append(
            {
                "dataset": "chime3",
                "file_path": str(fp),
                "recording_id": f"chime3_{env}_{fp.stem}",
                "original_label": env,
                "mapped_context": mapping.mapped_context,
            }
        )
    return rows


def _assign_splits(
    rows: List[Dict[str, Any]],
    ratios: Dict[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    """Assign train/validation/test at recording_id level."""
    # Group by recording_id
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["recording_id"]].append(row)

    # Respect SONYC original split when available
    sonyc_train, sonyc_val, sonyc_test = [], [], []
    other_groups: Dict[str, List[Dict[str, Any]]] = {}

    for rid, items in groups.items():
        if items[0]["dataset"] == "sonyc_ust":
            split = items[0].get("sonyc_split", "")
            if split == "train":
                sonyc_train.extend(items)
            elif split in ("validate", "validation"):
                sonyc_val.extend(items)
            elif split == "test":
                sonyc_test.extend(items)
            else:
                other_groups[rid] = items
        else:
            other_groups[rid] = items

    # Split non-SONYC groups by hash
    rng = np.random.RandomState(seed)
    group_ids = sorted(other_groups.keys())
    rng.shuffle(group_ids)

    n = len(group_ids)
    train_end = int(n * ratios.get("train", 0.70))
    val_end = train_end + int(n * ratios.get("validation", 0.15))

    for i, gid in enumerate(group_ids):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "validation"
        else:
            split = "test"
        for item in other_groups[gid]:
            item["split"] = split

    for item in sonyc_train:
        item["split"] = "train"
    for item in sonyc_val:
        item["split"] = "validation"
    for item in sonyc_test:
        item["split"] = "test"

    return rows


def verify_and_build_metadata() -> Dict[str, Any]:
    """Main pipeline: scan, validate, map labels, split."""
    config = _load_config()
    split_cfg = config.get("splitting", {})
    ratios = split_cfg.get("ratios", {"train": 0.70, "validation": 0.15, "test": 0.15})
    seed = split_cfg.get("random_seed", 42)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write label mapping reference
    mapping_entries = get_all_mapping_entries()
    mapping_path = METADATA_DIR / "label_mapping.csv"
    with open(mapping_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "original_label", "mapped_context", "reason"],
        )
        writer.writeheader()
        for e in mapping_entries:
            writer.writerow(
                {
                    "dataset": e.dataset,
                    "original_label": e.original_label,
                    "mapped_context": e.mapped_context,
                    "reason": e.reason,
                }
            )

    # Collect all recordings
    all_rows = _scan_demand() + _scan_sonyc() + _scan_chime3()
    if not all_rows:
        logger.error("No audio files found in data/raw/. Run download_datasets.py first.")
        return {"status": "no_data", "total_files": 0}

    # Validate each file
    valid_rows: List[Dict[str, Any]] = []
    corrupt: List[str] = []
    hash_seen: Dict[str, str] = {}
    duplicates: List[str] = []

    for row in all_rows:
        fp = Path(row["file_path"])
        v = _validate_audio_file(fp)
        if not v["valid"]:
            corrupt.append(f"{fp}: {v['error']}")
            continue

        row["duration"] = v["duration"]
        row["sample_rate"] = v["sample_rate"]
        row["channels"] = v["channels"]

        fhash = _file_hash(fp)
        if fhash in hash_seen:
            duplicates.append(f"{fp} duplicates {hash_seen[fhash]}")
        else:
            hash_seen[fhash] = str(fp)
            valid_rows.append(row)

    # Assign splits
    valid_rows = _assign_splits(valid_rows, ratios, seed)

    # Write master metadata
    fieldnames = [
        "dataset", "file_path", "recording_id", "duration", "sample_rate",
        "channels", "original_label", "mapped_context", "split",
    ]
    meta_path = METADATA_DIR / "audio_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid_rows)

    # Write split CSVs
    for split_name in ("train", "validation", "test"):
        split_rows = [r for r in valid_rows if r.get("split") == split_name]
        split_path = METADATA_DIR / f"{split_name}.csv"
        with open(split_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_rows)

    summary = {
        "status": "ok",
        "total_files": len(valid_rows),
        "corrupt_files": len(corrupt),
        "duplicate_files": len(duplicates),
        "train": sum(1 for r in valid_rows if r.get("split") == "train"),
        "validation": sum(1 for r in valid_rows if r.get("split") == "validation"),
        "test": sum(1 for r in valid_rows if r.get("split") == "test"),
        "total_duration_s": sum(r["duration"] for r in valid_rows),
        "classes": {c: sum(1 for r in valid_rows if r["mapped_context"] == c) for c in CONTEXT_CLASSES},
        "corrupt_list": corrupt[:20],
        "duplicate_list": duplicates[:20],
    }

    with open(METADATA_DIR / "verification_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Valid recordings: {summary['total_files']}")
    print(f"  Corrupt: {summary['corrupt_files']}")
    print(f"  Duplicates skipped: {summary['duplicate_files']}")
    print(f"  Train/Val/Test: {summary['train']}/{summary['validation']}/{summary['test']}")
    print(f"  Metadata: {meta_path}")
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    summary = verify_and_build_metadata()
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
