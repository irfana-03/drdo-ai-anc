#!/usr/bin/env python3
"""
verify_installation.py - End-to-end verification of the DRDO AI-ANC project.

Runs all component checks and prints a consolidated PASS / FAIL report.

Usage:
    python scripts/verify_installation.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Make sure project root is on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check(label: str, func) -> bool:
    """Run *func*; return True on success, False on failure."""
    try:
        func()
        print(f"  [PASS] {label}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version():
    assert sys.version_info >= (3, 9), (
        f"Python >= 3.9 required, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def check_import_torch():
    import torch  # noqa: F401


def check_import_torchaudio():
    import torchaudio  # noqa: F401


def check_import_numpy():
    import numpy  # noqa: F401


def check_import_scipy():
    import scipy  # noqa: F401


def check_import_librosa():
    import librosa  # noqa: F401


def check_import_soundfile():
    import soundfile  # noqa: F401


def check_import_sklearn():
    import sklearn  # noqa: F401


def check_import_sounddevice():
    import sounddevice  # noqa: F401


def check_import_yaml():
    import yaml  # noqa: F401


def check_import_matplotlib():
    import matplotlib  # noqa: F401


def check_import_pandas():
    import pandas  # noqa: F401


def check_config_exists():
    cfg = PROJECT_ROOT / "config" / "config.yaml"
    assert cfg.exists(), f"Config not found at {cfg}"


def check_config_loads():
    import yaml

    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert "datasets" in cfg, "Config missing 'datasets' section"
    assert "audio" in cfg, "Config missing 'audio' section"


def check_project_structure():
    required_dirs = [
        "config",
        "data/raw",
        "data/processed",
        "data/metadata",
        "models/pretrained",
        "models/custom",
        "src/data",
        "src/preprocessing",
        "src/enhancement",
        "src/classification",
        "src/adaptive",
        "src/realtime",
        "src/evaluation",
        "scripts",
        "tests",
        "app",
    ]
    for d in required_dirs:
        full = PROJECT_ROOT / d
        assert full.exists(), f"Missing directory: {d}"


def check_module_audio_loader():
    from src.data.audio_loader import load_audio, save_audio, scan_audio_files, AudioMeta  # noqa: F401


def check_module_dataset_manager():
    from src.data.dataset_manager import DatasetManager, DatasetRecord  # noqa: F401


def check_module_dataset_validator():
    from src.data.dataset_validator import DatasetValidator  # noqa: F401


def check_module_audio_preprocessing():
    from src.preprocessing.audio_preprocessing import AudioPreprocessor  # noqa: F401


def check_module_feature_extraction():
    from src.preprocessing.feature_extraction import FeatureExtractor, FeatureConfig  # noqa: F401


def check_dataset_manager_loads():
    from src.data.dataset_manager import DatasetManager

    mgr = DatasetManager()
    datasets = mgr.list_datasets()
    assert len(datasets) >= 3, f"Expected >= 3 datasets, got {len(datasets)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    width = 64
    print("=" * width)
    print("  DRDO AI-ANC - Installation Verification Report")
    print("=" * width)

    checks = [
        # ---- Environment ----
        ("Python >= 3.9", check_python_version),
        ("import torch", check_import_torch),
        ("import torchaudio", check_import_torchaudio),
        ("import numpy", check_import_numpy),
        ("import scipy", check_import_scipy),
        ("import librosa", check_import_librosa),
        ("import soundfile", check_import_soundfile),
        ("import scikit-learn", check_import_sklearn),
        ("import sounddevice", check_import_sounddevice),
        ("import PyYAML", check_import_yaml),
        ("import matplotlib", check_import_matplotlib),
        ("import pandas", check_import_pandas),
        # ---- Project structure ----
        ("config/config.yaml exists", check_config_exists),
        ("config.yaml loads correctly", check_config_loads),
        ("Project directory structure", check_project_structure),
        # ---- Modules ----
        ("Module: audio_loader", check_module_audio_loader),
        ("Module: dataset_manager", check_module_dataset_manager),
        ("Module: dataset_validator", check_module_dataset_validator),
        ("Module: audio_preprocessing", check_module_audio_preprocessing),
        ("Module: feature_extraction", check_module_feature_extraction),
        # ---- Integration ----
        ("DatasetManager loads registry", check_dataset_manager_loads),
    ]

    pass_count = 0
    fail_count = 0

    print()
    for label, func in checks:
        ok = _check(label, func)
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    print()
    print("=" * width)
    print(f"  Results: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("  [OK]  All checks passed - project is ready.")
    else:
        print("  [!!]  Some checks failed - review errors above.")
    print("=" * width + "\n")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
