"""
dataset_manager.py — Registry, CLI and orchestration for all supported datasets.

Responsibilities:
    - Load the dataset registry from ``config/config.yaml``.
    - Expose ``--list``, ``--validate``, ``--metadata``, and ``--prepare`` CLI
      commands.
    - Orchestrate validation, metadata generation, preprocessing and splitting.
    - Prevent data leakage: splits are performed at recording / source level.

Usage:
    python -m src.data.dataset_manager --list
    python -m src.data.dataset_manager --validate chime3
    python -m src.data.dataset_manager --metadata demand
    python -m src.data.dataset_manager --prepare sonyc_ust
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Sibling imports
from src.data.audio_loader import scan_audio_files, load_audio
from src.data.dataset_validator import DatasetValidator
from src.preprocessing.audio_preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk upward from this file until we find ``config/config.yaml``."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    # Fallback: assume CWD
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load and return the YAML configuration dictionary."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecord:
    """In-memory representation of a single dataset from the registry."""

    key: str
    name: str
    description: str
    source_url: str
    license: str
    local_path: Path
    expected_sr: Optional[int]
    notes: str = ""
    num_files: int = 0
    status: str = "not_downloaded"  # not_downloaded | downloaded | validated | prepared

    @classmethod
    def from_config(cls, key: str, entry: dict, root: Path) -> "DatasetRecord":
        return cls(
            key=key,
            name=entry.get("name", key),
            description=entry.get("description", ""),
            source_url=entry.get("source_url", ""),
            license=entry.get("license", "unknown"),
            local_path=(root / entry.get("local_path", f"data/raw/{key}")).resolve(),
            expected_sr=entry.get("expected_sr"),
            notes=entry.get("notes", ""),
        )

    def refresh_status(self) -> None:
        """Update *status* and *num_files* based on what is on disk."""
        if not self.local_path.exists():
            self.status = "not_downloaded"
            self.num_files = 0
            return
        audio_files = scan_audio_files(self.local_path)
        self.num_files = len(audio_files)
        if self.num_files == 0:
            self.status = "not_downloaded"
        else:
            self.status = "downloaded"


# ---------------------------------------------------------------------------
# Dataset Manager
# ---------------------------------------------------------------------------

class DatasetManager:
    """Central hub for listing, validating, and preparing datasets."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_config()
        self.root = PROJECT_ROOT
        self.datasets: Dict[str, DatasetRecord] = {}
        self._load_registry()

    # ---- registry ---------------------------------------------------------

    def _load_registry(self) -> None:
        ds_section = self.config.get("datasets", {})
        for key, entry in ds_section.items():
            record = DatasetRecord.from_config(key, entry, self.root)
            record.refresh_status()
            self.datasets[key] = record

    def list_datasets(self) -> List[DatasetRecord]:
        """Return all registered datasets."""
        return list(self.datasets.values())

    def get_dataset(self, key: str) -> DatasetRecord:
        if key not in self.datasets:
            raise KeyError(
                f"Unknown dataset '{key}'. "
                f"Available: {list(self.datasets.keys())}"
            )
        return self.datasets[key]

    # ---- actions ----------------------------------------------------------

    def validate(self, key: str) -> dict:
        """Run full validation for the specified dataset.

        Returns a summary dict with counts and any errors found.
        """
        record = self.get_dataset(key)
        validator = DatasetValidator(
            dataset_path=record.local_path,
            expected_sr=record.expected_sr,
        )
        summary = validator.validate()
        if summary["corrupt_files"] == 0 and summary["total_files"] > 0:
            record.status = "validated"
        return summary

    def generate_metadata(self, key: str) -> Path:
        """Generate a metadata CSV for the specified dataset.

        Returns the path to the CSV.
        """
        record = self.get_dataset(key)
        validator = DatasetValidator(
            dataset_path=record.local_path,
            expected_sr=record.expected_sr,
        )
        metadata_dir = self.root / "data" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metadata_dir / f"{key}_metadata.csv"
        validator.generate_metadata_csv(csv_path)
        return csv_path

    def prepare(self, key: str) -> Path:
        """Preprocess a dataset: resample, convert to mono, normalise.

        Writes processed files to ``data/processed/<key>/`` — raw data is
        never modified.  Returns the output directory path.
        """
        record = self.get_dataset(key)
        audio_cfg = self.config.get("audio", {})
        target_sr = audio_cfg.get("target_sample_rate", 48000)
        norm_cfg = audio_cfg.get("normalization", {})

        preprocessor = AudioPreprocessor(
            target_sr=target_sr,
            mono=audio_cfg.get("mono", True),
            normalization_method=norm_cfg.get("method", "peak"),
            normalization_target_db=norm_cfg.get("target_db", -3.0),
        )

        output_dir = (self.root / "data" / "processed" / key).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = scan_audio_files(record.local_path)
        if not audio_files:
            logger.warning("No audio files found for dataset '%s'.", key)
            return output_dir

        for af in audio_files:
            try:
                preprocessor.process_file(
                    input_path=af,
                    output_dir=output_dir,
                )
            except Exception as exc:
                logger.error("Failed to process %s: %s", af, exc)

        record.status = "prepared"
        logger.info(
            "Prepared %d files for dataset '%s' → %s",
            len(audio_files),
            key,
            output_dir,
        )
        return output_dir

    # ---- dataset splitting ------------------------------------------------

    def split_dataset(
        self,
        key: str,
        ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> Dict[str, List[Path]]:
        """Split a dataset at recording / source level.

        The split is deterministic (seeded hash of the source-level
        identifier), preventing data leakage.

        Parameters
        ----------
        key : str
            Dataset key.
        ratios : dict, optional
            e.g. ``{"train": 0.70, "validation": 0.15, "test": 0.15}``.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Mapping from split name to list of file paths.
        """
        record = self.get_dataset(key)
        if ratios is None:
            split_cfg = self.config.get("splitting", {})
            ratios = split_cfg.get(
                "ratios", {"train": 0.70, "validation": 0.15, "test": 0.15}
            )
            seed = split_cfg.get("random_seed", seed)

        audio_files = scan_audio_files(record.local_path)
        if not audio_files:
            return {"train": [], "validation": [], "test": []}

        # Group files by their *recording source* — defined as the
        # immediate parent directory.  This prevents segments from the
        # same recording from leaking across splits.
        source_groups: Dict[str, List[Path]] = {}
        for fp in audio_files:
            source_id = fp.parent.name  # e.g. "BUS", "CAF", "PED", …
            source_groups.setdefault(source_id, []).append(fp)

        # Deterministic assignment via hashed source id
        rng = np.random.RandomState(seed)
        source_keys = sorted(source_groups.keys())
        rng.shuffle(source_keys)

        n = len(source_keys)
        train_end = int(n * ratios.get("train", 0.70))
        val_end = train_end + int(n * ratios.get("validation", 0.15))

        splits: Dict[str, List[Path]] = {"train": [], "validation": [], "test": []}
        for i, sk in enumerate(source_keys):
            if i < train_end:
                splits["train"].extend(source_groups[sk])
            elif i < val_end:
                splits["validation"].extend(source_groups[sk])
            else:
                splits["test"].extend(source_groups[sk])

        for split_name, files in splits.items():
            logger.info(
                "Split '%s' — %s: %d files", key, split_name, len(files)
            )

        return splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_dataset_table(records: List[DatasetRecord]) -> None:
    header = f"{'Key':<15} {'Name':<50} {'Files':>6} {'Status':<16} {'License'}"
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r.key:<15} {r.name:<50} {r.num_files:>6} {r.status:<16} {r.license}"
        )
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="dataset_manager",
        description="DRDO AI-ANC Dataset Manager",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered datasets and their status.",
    )
    parser.add_argument(
        "--validate",
        metavar="DATASET",
        help="Run validation on the specified dataset.",
    )
    parser.add_argument(
        "--metadata",
        metavar="DATASET",
        help="Generate a metadata CSV for the specified dataset.",
    )
    parser.add_argument(
        "--prepare",
        metavar="DATASET",
        help="Preprocess (resample, mono, normalise) a dataset.",
    )
    parser.add_argument(
        "--split",
        metavar="DATASET",
        help="Split a dataset into train / validation / test.",
    )

    args = parser.parse_args()
    mgr = DatasetManager()

    if args.list:
        _print_dataset_table(mgr.list_datasets())
        return

    if args.validate:
        summary = mgr.validate(args.validate)
        print(f"\n=== Validation summary for '{args.validate}' ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    if args.metadata:
        csv_path = mgr.generate_metadata(args.metadata)
        print(f"\nMetadata CSV written to: {csv_path}")
        return

    if args.prepare:
        out = mgr.prepare(args.prepare)
        print(f"\nProcessed files written to: {out}")
        return

    if args.split:
        splits = mgr.split_dataset(args.split)
        print(f"\n=== Split summary for '{args.split}' ===")
        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} files")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
