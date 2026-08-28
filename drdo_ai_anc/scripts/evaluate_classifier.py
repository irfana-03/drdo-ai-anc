#!/usr/bin/env python3
"""
evaluate_classifier.py — Evaluate trained classifier on held-out test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.dataset import create_dataloaders
from src.classification.noise_classifier import NoiseContextClassifier
from src.evaluation.classification_metrics import compute_metrics, save_results
from src.preprocessing.feature_extraction import FeatureConfig

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
MODELS_DIR = PROJECT_ROOT / "models" / "custom"
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    config = _load_config()
    train_cfg = config.get("training", {})
    feat_cfg = FeatureConfig.from_config(config.get("features", {}))

    model_path = MODELS_DIR / "noise_context_classifier.pt"
    mapping_path = MODELS_DIR / "class_mapping.json"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}. Run train_classifier.py first.")
        return 1

    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    class_names = mapping["class_names"]
    class_to_idx = mapping["class_to_idx"]

    classifier = NoiseContextClassifier(class_names=class_names, n_mels=feat_cfg.n_mels)
    classifier.load(model_path)

    _, _, test_loader = create_dataloaders(
        train_csv=METADATA_DIR / "train.csv",
        val_csv=METADATA_DIR / "validation.csv",
        test_csv=METADATA_DIR / "test.csv",
        class_to_idx=class_to_idx,
        batch_size=train_cfg.get("batch_size", 16),
        target_sr=train_cfg.get("sample_rate", 16000),
        clip_duration_s=train_cfg.get("clip_duration_s", 4.0),
        feature_config=feat_cfg,
    )

    y_true, y_pred = [], []
    classifier.model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(classifier.device)
            outputs = classifier.model(features)
            _, predicted = outputs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    if not y_true:
        print("ERROR: Test set is empty.")
        return 1

    metrics = compute_metrics(y_true, y_pred, class_names)
    report_path, json_path, cm_path = save_results(metrics, RESULTS_DIR, class_names)

    print("\n  === TEST SET EVALUATION ===")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"  Report:     {report_path}")
    print(f"  Metrics:    {json_path}")
    print(f"  Confusion:  {cm_path}")
    print(f"\n{metrics['classification_report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
