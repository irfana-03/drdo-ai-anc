#!/usr/bin/env python3
"""
test_audio_load.py — Simple test that loads one REAL audio file and prints
its properties.

Usage:
    python tests/test_audio_load.py <path_to_audio_file>

Example:
    python tests/test_audio_load.py data/raw/demand/DKITCHEN/ch01.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_loader import load_audio


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python tests/test_audio_load.py <path_to_audio_file>\n"
            "\n"
            "Example:\n"
            "  python tests/test_audio_load.py data/raw/demand/DKITCHEN/ch01.wav\n"
        )
        # If no file provided, try to find any audio file in data/raw
        raw_dir = PROJECT_ROOT / "data" / "raw"
        candidates = []
        if raw_dir.exists():
            for ext in (".wav", ".flac", ".ogg", ".mp3"):
                candidates.extend(raw_dir.rglob(f"*{ext}"))
        if candidates:
            filepath = candidates[0]
            print(f"[AUTO] Found audio file: {filepath}\n")
        else:
            print(
                "[INFO] No audio files found in data/raw/.\n"
                "       Download a dataset first (see scripts/download_datasets.py).\n"
                "       You can also pass any audio file path as an argument.\n"
            )
            return 1
    else:
        filepath = Path(sys.argv[1])
        if not filepath.is_absolute():
            filepath = PROJECT_ROOT / filepath

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return 1

    print(f"Loading: {filepath}\n")

    try:
        waveform, meta = load_audio(filepath)
    except Exception as exc:
        print(f"[ERROR] Failed to load audio: {exc}")
        return 1

    print("=" * 50)
    print("  Audio File Properties")
    print("=" * 50)
    print(f"  Filename       : {meta.filename}")
    print(f"  Full path      : {meta.filepath}")
    print(f"  Duration       : {meta.duration_seconds:.4f} s")
    print(f"  Sample rate    : {meta.sample_rate} Hz")
    print(f"  Channels       : {meta.channels}")
    print(f"  Num samples    : {meta.num_samples}")
    print(f"  RMS            : {meta.rms:.6f}")
    print(f"  Peak amplitude : {meta.peak_amplitude:.6f}")
    print(f"  Format         : {meta.format_info}")
    print("=" * 50)
    print("\n  ✓  Audio file loaded and inspected successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
