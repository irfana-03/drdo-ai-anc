#!/usr/bin/env python3
"""Create local jury presentation package (no raw audio)."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent / "DRDO_AI_ANC_JURY_PACKAGE"


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    print(f"Creating jury package at: {PACKAGE_ROOT}")
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)

    # 01_PROJECT
    proj = PACKAGE_ROOT / "01_PROJECT"
    for name in ("README.md", "requirements.txt", "run_dashboard.bat", "train.py"):
        _copy(PROJECT_ROOT / name, proj / name)
    _copy(PROJECT_ROOT / "config" / "config.yaml", proj / "config" / "config.yaml")

    # 02_DATASET_INFO
    ds = PACKAGE_ROOT / "02_DATASET_INFO"
    _copy(PROJECT_ROOT / "data" / "metadata" / "audio_metadata.csv", ds / "dataset_metadata.csv")
    _copy(PROJECT_ROOT / "data" / "metadata" / "label_mapping.csv", ds / "label_mapping.csv")
    _copy(PROJECT_ROOT / "results" / "metrics" / "dataset_report.json", ds / "dataset_report.json")

    report_path = PROJECT_ROOT / "results" / "metrics" / "dataset_report.json"
    summary_lines = ["DRDO AI ANC — Dataset Summary", "=" * 40, ""]
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        summary_lines.append(f"Total recordings: {data.get('total_recordings')}")
        summary_lines.append(f"Total duration (h): {data.get('total_duration_h')}")
        summary_lines.append(f"Train: {data.get('splits', {}).get('train')}")
        summary_lines.append(f"Validation: {data.get('splits', {}).get('validation')}")
        summary_lines.append(f"Test: {data.get('splits', {}).get('test')}")
        summary_lines.append("")
        for name, info in data.get("datasets", {}).items():
            summary_lines.append(f"{name.upper()}: {info.get('recordings')} recordings, {info.get('duration_s')}s")
    (ds / "dataset_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    (ds / "README.md").write_text(
        "# Dataset Information\n\n"
        "Real recorded audio from DEMAND and SONYC-UST.\n"
        "CHiME-3 requires manual download.\n\n"
        "Raw audio is NOT included in this package (too large).\n"
        "Metadata CSVs document actual recordings used for training.\n",
        encoding="utf-8",
    )

    # 03_MODEL
    model_dir = PACKAGE_ROOT / "03_MODEL"
    _copy(PROJECT_ROOT / "models" / "custom" / "class_mapping.json", model_dir / "class_mapping.json")
    _copy(PROJECT_ROOT / "models" / "custom" / "training_config.json", model_dir / "training_config.json")
    pt = PROJECT_ROOT / "models" / "custom" / "noise_context_classifier.pt"
    if pt.exists() and pt.stat().st_size < 50_000_000:
        _copy(pt, model_dir / "noise_context_classifier.pt")
        model_loc = "Included in 03_MODEL/"
    else:
        model_loc = str(pt)

    info = [
        "DRDO AI ANC — Model Information",
        "=" * 40,
        f"Generated: {datetime.now().isoformat()}",
        f"Model path: {model_loc}",
        "Classes: STATIONARY, DYNAMIC, IMPULSIVE, SPEECH, OTHER",
        "Architecture: Compact CNN on log-mel spectrogram",
    ]
    (model_dir / "model_info.txt").write_text("\n".join(info), encoding="utf-8")

    # 04_RESULTS
    res = PACKAGE_ROOT / "04_RESULTS"
    _copy(PROJECT_ROOT / "results" / "metrics" / "classification_report.txt", res / "classification_report.txt")
    _copy(PROJECT_ROOT / "results" / "metrics" / "metrics.json", res / "metrics.json")
    _copy(PROJECT_ROOT / "results" / "figures" / "confusion_matrix.png", res / "confusion_matrix.png")

    print("Jury package created successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
