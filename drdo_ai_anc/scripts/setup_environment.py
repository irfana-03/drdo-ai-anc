#!/usr/bin/env python3
"""
setup_environment.py — Validate that the runtime environment meets all
requirements for the DRDO AI-ANC project.

This script checks for:
    - Python version (>= 3.9)
    - All required packages and their importability
    - CUDA / GPU availability (optional)
    - Audio device availability (optional)

Usage:
    python scripts/setup_environment.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Required packages
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    # (import_name, display_name)
    ("torch", "PyTorch"),
    ("torchaudio", "torchaudio"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("sklearn", "scikit-learn"),
    ("sounddevice", "sounddevice"),
    ("yaml", "PyYAML"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("tqdm", "tqdm"),
    ("click", "click"),
]


def _version(mod) -> str:
    """Best-effort version string from a module."""
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if v is not None:
            return str(v)
    return "installed (version unknown)"


def check_python() -> Tuple[bool, str]:
    ver = sys.version_info
    ok = ver >= (3, 9)
    msg = f"Python {ver.major}.{ver.minor}.{ver.micro}"
    if not ok:
        msg += "  (>= 3.9 required)"
    return ok, msg


def check_packages() -> List[Tuple[str, bool, str]]:
    results = []
    for import_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = _version(mod)
            results.append((display_name, True, ver))
        except ImportError as exc:
            results.append((display_name, False, str(exc)))
    return results


def check_cuda() -> Tuple[bool, str]:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, f"CUDA available — {name}"
        return False, "CUDA not available (CPU-only mode)"
    except Exception as exc:
        return False, f"Cannot query CUDA: {exc}"


def check_audio_device() -> Tuple[bool, str]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        if input_devs:
            default = sd.query_devices(kind="input")
            return True, f"Input device: {default['name']}"
        return False, "No input audio devices found"
    except Exception as exc:
        return False, f"Cannot query audio devices: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    width = 60
    print("=" * width)
    print("  DRDO AI-ANC — Environment Validation")
    print("=" * width)

    all_pass = True

    # Python version
    ok, msg = check_python()
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] Python version: {msg}")
    all_pass = all_pass and ok

    # Platform info
    print(f"  [INFO] Platform: {platform.platform()}")
    print(f"  [INFO] Architecture: {platform.machine()}")

    # Required packages
    print(f"\n  {'Package':<20} {'Status':<8} {'Version / Error'}")
    print(f"  {'-'*20} {'-'*8} {'-'*30}")
    results = check_packages()
    for name, ok, info in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<20} [{status}]   {info}")
        all_pass = all_pass and ok

    # CUDA / GPU
    ok, msg = check_cuda()
    status = "PASS" if ok else "INFO"
    print(f"\n  [{status}] GPU: {msg}")

    # Audio device
    ok, msg = check_audio_device()
    status = "PASS" if ok else "WARN"
    print(f"  [{status}] Audio: {msg}")

    # Summary
    print("\n" + "=" * width)
    if all_pass:
        print("  ✓  All required dependencies are satisfied.")
    else:
        print("  ✗  Some dependencies are MISSING — install them with:")
        print("       pip install -r requirements.txt")
    print("=" * width + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
