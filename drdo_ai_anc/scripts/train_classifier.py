#!/usr/bin/env python3
"""
train_classifier.py — Train noise context classifier on real audio data.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import create_dataloaders
from src.classification.noise_classifier import NoiseContextClassifier
from src.data.label_mapping import CONTEXT_CLASSES
from src.preprocessing.feature_extraction import FeatureConfig

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
MODELS_DIR = PROJECT_ROOT / "models" / "custom"


def _load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    config = _load_config()
    train_cfg = config.get("training", {})
    feat_cfg = FeatureConfig.from_config(config.get("features", {}))

    seed = train_cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_csv = METADATA_DIR / "train.csv"
    val_csv = METADATA_DIR / "validation.csv"
    test_csv = METADATA_DIR / "test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            print(f"ERROR: Missing {p}. Run verify_real_dataset.py first.")
            return 1

    train_df = pd.read_csv(train_csv)
    if len(train_df) == 0:
        print("ERROR: Training set is empty.")
        return 1

    # Only use classes present in training data
    present_classes = sorted(train_df["mapped_context"].unique())
    class_names = [c for c in CONTEXT_CLASSES if c in present_classes]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"\n  Classes: {class_names}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(pd.read_csv(val_csv))}")
    print(f"  Test samples: {len(pd.read_csv(test_csv))}")

    target_sr = train_cfg.get("sample_rate", 16000)
    batch_size = train_cfg.get("batch_size", 16)
    clip_duration = train_cfg.get("clip_duration_s", 4.0)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        target_sr=target_sr,
        clip_duration_s=clip_duration,
        feature_config=feat_cfg,
    )

    # Class weights for imbalanced data
    counts = Counter(train_df["mapped_context"])
    weights = []
    total = len(train_df)
    for c in class_names:
        w = total / (len(class_names) * counts.get(c, 1))
        weights.append(w)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"  Class weights: {dict(zip(class_names, weights))}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "noise_context_classifier.pt"

    classifier = NoiseContextClassifier(
        class_names=class_names,
        n_mels=feat_cfg.n_mels,
    )

    print("\n  Training...")
    history = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.get("epochs", 30),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        class_weights=class_weights,
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
        checkpoint_path=model_path,
    )

    # Save class mapping and training config
    with open(MODELS_DIR / "class_mapping.json", "w", encoding="utf-8") as fh:
        json.dump({"class_to_idx": class_to_idx, "class_names": class_names}, fh, indent=2)

    training_config = {
        "sample_rate": target_sr,
        "batch_size": batch_size,
        "clip_duration_s": clip_duration,
        "epochs_run": len(history["train_loss"]),
        "random_seed": seed,
        "n_mels": feat_cfg.n_mels,
        "n_fft": feat_cfg.n_fft,
        "hop_length": feat_cfg.hop_length,
        "train_samples": len(train_df),
        "classes": class_names,
    }
    with open(MODELS_DIR / "training_config.json", "w", encoding="utf-8") as fh:
        json.dump(training_config, fh, indent=2)

    print(f"\n  Model saved: {model_path}")
    print(f"  Best val acc: {max(history['val_acc']):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
