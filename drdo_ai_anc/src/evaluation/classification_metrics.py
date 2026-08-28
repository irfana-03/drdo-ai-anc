"""
classification_metrics.py — Evaluation metrics for noise context classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict:
    """Compute full classification metrics on held-out test set."""
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_p = precision_score(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    per_class_r = recall_score(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            class_names[i]: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_results(
    metrics: Dict,
    output_dir: Path,
    class_names: List[str],
) -> Tuple[Path, Path, Path]:
    """Save metrics JSON, classification report, and confusion matrix figure."""
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report_path = metrics_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(metrics["classification_report"])

    json_path = metrics_dir / "metrics.json"
    json_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_metrics, fh, indent=2)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix — Noise Context Classifier",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    return report_path, json_path, cm_path
