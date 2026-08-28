"""Load evaluation metrics, dataset metadata, and session logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_metrics() -> Optional[Dict[str, Any]]:
    path = PROJECT_ROOT / "results" / "metrics" / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset_report() -> Optional[Dict[str, Any]]:
    path = PROJECT_ROOT / "results" / "metrics" / "dataset_report.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_audio_metadata() -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / "data" / "metadata" / "audio_metadata.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_classification_report() -> Optional[str]:
    path = PROJECT_ROOT / "results" / "metrics" / "classification_report.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def get_confusion_matrix_path() -> Optional[Path]:
    path = PROJECT_ROOT / "results" / "figures" / "confusion_matrix.png"
    return path if path.exists() else None


def load_session_logs() -> pd.DataFrame:
    log_dir = PROJECT_ROOT / "results" / "realtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "sessions.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp", "noise_context", "confidence", "impulsive_event",
                "speech_probability", "latency_ms", "rtf", "input_rms",
                "output_rms", "engine", "mode",
            ]
        )
    return pd.read_csv(path)


def check_dataset_availability() -> Dict[str, str]:
    raw = PROJECT_ROOT / "data" / "raw"
    status = {}
    for name in ("demand", "sonyc_ust", "chime3"):
        d = raw / name
        if not d.exists():
            status[name] = "NOT AVAILABLE"
            continue
        wavs = list(d.rglob("*.wav"))
        if wavs:
            status[name] = f"AVAILABLE ({len(wavs)} files)"
        else:
            status[name] = "NOT AVAILABLE / MANUAL DOWNLOAD REQUIRED"
    return status


def find_sample_wav() -> Optional[Path]:
    raw = PROJECT_ROOT / "data" / "raw"
    for sub in ("sonyc_ust", "demand", "chime3"):
        d = raw / sub
        if d.exists():
            wavs = sorted(d.rglob("*.wav"))
            if wavs:
                return wavs[0]
    return None
