"""Model and system resource loading with caching."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.noise_classifier import NoiseContextClassifier
from src.enhancement.engine import EnhancementEngine
from src.preprocessing.feature_extraction import FeatureConfig
from src.realtime.pipeline import AudioPipeline


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_model_paths() -> Dict[str, Path]:
    return {
        "classifier": PROJECT_ROOT / "models" / "custom" / "noise_context_classifier.pt",
        "class_mapping": PROJECT_ROOT / "models" / "custom" / "class_mapping.json",
        "training_config": PROJECT_ROOT / "models" / "custom" / "training_config.json",
    }


def load_classifier() -> Tuple[Optional[NoiseContextClassifier], str]:
    paths = get_model_paths()
    if not paths["classifier"].exists():
        return None, "NOT FOUND"
    try:
        with open(paths["class_mapping"], "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
        cfg = load_config()
        n_mels = cfg.get("features", {}).get("mel", {}).get("n_mels", 64)
        clf = NoiseContextClassifier(class_names=mapping["class_names"], n_mels=n_mels)
        clf.load(paths["classifier"])
        return clf, "READY"
    except Exception as exc:
        return None, f"ERROR: {exc}"


def build_pipeline(classifier: NoiseContextClassifier) -> AudioPipeline:
    cfg = load_config()
    feat_cfg = FeatureConfig.from_config(cfg.get("features", {}))
    train_cfg = cfg.get("training", {})
    rt_cfg = cfg.get("realtime", {})
    impulse_cfg = cfg.get("impulse_detector", {})
    return AudioPipeline(
        classifier=classifier,
        feature_config=feat_cfg,
        impulse_config=impulse_cfg,
        classifier_sr=train_cfg.get("sample_rate", 16000),
        processing_sr=cfg.get("audio", {}).get("target_sample_rate", 48000),
        clip_duration_s=train_cfg.get("clip_duration_s", 4.0),
        filter_length=rt_cfg.get("filter_length", 512),
    )


def get_enhancement_status() -> Dict[str, str]:
    engine = EnhancementEngine()
    return engine.status()


def get_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cuda": "AVAILABLE" if torch.cuda.is_available() else "NOT AVAILABLE",
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_used_gb"] = round(mem.used / (1024**3), 1)
        info["ram_pct"] = mem.percent
    except ImportError:
        info["ram_total_gb"] = None
        info["ram_used_gb"] = None
        info["ram_pct"] = None
    return info


def get_audio_devices() -> Dict[str, list]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs, outputs = [], []
        for i, d in enumerate(devices):
            entry = {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            if d["max_input_channels"] > 0:
                inputs.append(entry)
            if d["max_output_channels"] > 0:
                outputs.append({"index": i, "name": d["name"], "channels": d["max_output_channels"]})
        return {"inputs": inputs, "outputs": outputs}
    except Exception:
        return {"inputs": [], "outputs": []}
