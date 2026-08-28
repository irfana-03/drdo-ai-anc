"""
dataset.py — PyTorch Dataset/DataLoader for noise context classification.

Loads audio on-the-fly; does not load entire dataset into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.audio_loader import load_audio
from src.preprocessing.feature_extraction import FeatureConfig, FeatureExtractor


class NoiseContextDataset(Dataset):
    """Dataset for acoustic context classification from real audio files."""

    def __init__(
        self,
        metadata_csv: str | Path,
        class_to_idx: Dict[str, int],
        target_sr: int = 16000,
        clip_duration_s: float = 4.0,
        feature_config: Optional[FeatureConfig] = None,
        augment: bool = False,
    ) -> None:
        self.df = pd.read_csv(metadata_csv)
        self.class_to_idx = class_to_idx
        self.target_sr = target_sr
        self.clip_samples = int(clip_duration_s * target_sr)
        self.feature_extractor = FeatureExtractor(feature_config)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        waveform, meta = load_audio(
            row["file_path"], target_sr=self.target_sr, mono=True
        )

        # Pad or truncate to fixed length
        if len(waveform) > self.clip_samples:
            if self.augment:
                start = np.random.randint(0, len(waveform) - self.clip_samples)
            else:
                start = (len(waveform) - self.clip_samples) // 2
            waveform = waveform[start : start + self.clip_samples]
        elif len(waveform) < self.clip_samples:
            pad = self.clip_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad), mode="constant")

        log_mel = self.feature_extractor.log_mel_spectrogram(waveform, self.target_sr)
        # Shape: (n_mels, time) -> (1, n_mels, time) for CNN
        features = torch.from_numpy(log_mel).float().unsqueeze(0)

        label = self.class_to_idx[row["mapped_context"]]
        return features, label


def create_dataloaders(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    class_to_idx: Dict[str, int],
    batch_size: int = 16,
    target_sr: int = 16000,
    clip_duration_s: float = 4.0,
    feature_config: Optional[FeatureConfig] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    train_ds = NoiseContextDataset(
        train_csv, class_to_idx, target_sr, clip_duration_s, feature_config, augment=True
    )
    val_ds = NoiseContextDataset(
        val_csv, class_to_idx, target_sr, clip_duration_s, feature_config, augment=False
    )
    test_ds = NoiseContextDataset(
        test_csv, class_to_idx, target_sr, clip_duration_s, feature_config, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
