#!/usr/bin/env python3
"""Quick smoke test for presentation readiness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf

from app.utils.data_loader import find_sample_wav, load_metrics
from app.utils.model_loader import build_pipeline, get_audio_devices, get_enhancement_status, load_classifier


def main() -> int:
    results = {}
    clf, model_st = load_classifier()
    results["model_load"] = model_st == "READY"

    wav = find_sample_wav()
    results["sample_wav"] = wav is not None

    if clf and wav:
        pipe = build_pipeline(clf)
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        r = pipe.process_file(audio[:48000], sr, block_size=4800)
        s = r["state"]
        results["inference"] = s.noise_context not in ("", "WAITING FOR AUDIO")
        results["predicted_class"] = s.noise_context
        results["confidence"] = round(s.confidence, 4)
        results["engine"] = s.engine
    else:
        results["inference"] = False

    metrics = load_metrics()
    results["metrics_available"] = metrics is not None
    if metrics:
        results["accuracy"] = metrics["accuracy"]
        results["macro_f1"] = metrics["macro_f1"]

    devices = get_audio_devices()
    results["audio_inputs"] = len(devices["inputs"])
    results["enhancement"] = get_enhancement_status()

    print("=" * 40)
    print("DRDO AI ANC — SMOKE TEST")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k}: {v}")

    ok = results.get("model_load") and results.get("inference") and results.get("metrics_available")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
