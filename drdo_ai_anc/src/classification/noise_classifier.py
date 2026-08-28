"""
noise_classifier.py — Lightweight CNN for acoustic context classification.

Classes: STATIONARY, DYNAMIC, IMPULSIVE, SPEECH, OTHER
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class NoiseContextCNN(nn.Module):
    """Compact CNN for log-mel spectrogram classification."""

    def __init__(self, n_classes: int = 5, n_mels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class NoiseContextClassifier:
    """High-level classifier with train/predict/save/load API."""

    def __init__(
        self,
        class_names: List[str],
        n_mels: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.n_mels = n_mels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NoiseContextCNN(n_classes=len(class_names), n_mels=n_mels).to(
            self.device
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 5,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict:
        """Train the classifier with early stopping on validation loss."""
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * features.size(0)
            train_loss /= len(train_loader.dataset)

            # Validate
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * features.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            val_loss /= max(len(val_loader.dataset), 1)
            val_acc = correct / max(total, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"  Epoch {epoch+1}/{epochs} — "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        return history

    def predict(self, features: torch.Tensor) -> Dict:
        """Predict class for a single feature tensor (1, n_mels, time)."""
        result = self.predict_proba(features)
        idx = int(np.argmax(result["probabilities"]))
        return {
            "noise_class": self.idx_to_class[idx],
            "confidence": float(result["probabilities"][idx]),
            "probabilities": result["probabilities"],
        }

    def predict_proba(self, features: torch.Tensor) -> Dict:
        """Return class probabilities."""
        self.model.eval()
        if features.dim() == 3:
            features = features.unsqueeze(0)
        features = features.to(self.device)
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {
            "probabilities": probs.tolist(),
            "class_names": self.class_names,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "class_names": self.class_names,
                "n_mels": self.n_mels,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint["class_names"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.n_mels = checkpoint.get("n_mels", 64)
        self.model = NoiseContextCNN(
            n_classes=len(self.class_names), n_mels=self.n_mels
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
