"""
dataset_validator.py — Validate audio datasets on disk.

Responsibilities:
    - Walk a dataset directory and verify each audio file is readable.
    - Check sample rate and channel count against expectations.
    - Detect corrupt / unreadable files.
    - Generate a per-file metadata CSV.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from src.data.audio_loader import SUPPORTED_EXTENSIONS, load_audio, scan_audio_files

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate and catalogue audio files within a dataset directory."""

    def __init__(
        self,
        dataset_path: str | Path,
        expected_sr: Optional[int] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.expected_sr = expected_sr

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Run a full validation pass over the dataset directory.

        Returns
        -------
        dict
            Summary with keys:
            - ``total_files``
            - ``valid_files``
            - ``corrupt_files``
            - ``sr_mismatch_files``
            - ``errors`` — list of ``(filepath, error_message)``
        """
        audio_files = scan_audio_files(self.dataset_path)

        total = len(audio_files)
        valid = 0
        corrupt = 0
        sr_mismatches = 0
        errors: List[tuple] = []

        if total == 0:
            logger.warning(
                "No audio files found in %s. "
                "Please ensure the dataset has been downloaded and extracted.",
                self.dataset_path,
            )
            return {
                "total_files": 0,
                "valid_files": 0,
                "corrupt_files": 0,
                "sr_mismatch_files": 0,
                "errors": [
                    (str(self.dataset_path), "Directory contains no audio files.")
                ],
            }

        for fp in audio_files:
            try:
                info = sf.info(str(fp))
            except Exception as exc:
                corrupt += 1
                errors.append((str(fp), f"Corrupt / unreadable: {exc}"))
                logger.error("CORRUPT: %s — %s", fp, exc)
                continue

            # Quick readability check: try loading first 1024 frames
            try:
                data, _ = sf.read(str(fp), frames=1024, dtype="float32")
                if data.size == 0:
                    raise ValueError("File has zero samples.")
            except Exception as exc:
                corrupt += 1
                errors.append((str(fp), f"Read failed: {exc}"))
                logger.error("READ FAIL: %s — %s", fp, exc)
                continue

            # Sample-rate check
            if self.expected_sr is not None and info.samplerate != self.expected_sr:
                sr_mismatches += 1
                errors.append(
                    (
                        str(fp),
                        f"SR mismatch: expected {self.expected_sr}, "
                        f"got {info.samplerate}",
                    )
                )
                logger.warning(
                    "SR MISMATCH: %s — expected %d, got %d",
                    fp,
                    self.expected_sr,
                    info.samplerate,
                )

            valid += 1

        summary = {
            "total_files": total,
            "valid_files": valid,
            "corrupt_files": corrupt,
            "sr_mismatch_files": sr_mismatches,
            "errors": errors,
        }

        logger.info(
            "Validation complete: %d total, %d valid, %d corrupt, %d SR mismatch",
            total,
            valid,
            corrupt,
            sr_mismatches,
        )
        return summary

    # ------------------------------------------------------------------
    # Metadata CSV generation
    # ------------------------------------------------------------------

    def generate_metadata_csv(self, output_path: str | Path) -> Path:
        """Scan every audio file and write a CSV with per-file metadata.

        Columns:
            filepath, filename, sample_rate, channels, duration_s,
            num_samples, rms, peak_amplitude, format_info, status

        Parameters
        ----------
        output_path : str or Path
            Destination CSV path.

        Returns
        -------
        Path
            The written CSV path.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_files = scan_audio_files(self.dataset_path)
        fieldnames = [
            "filepath",
            "filename",
            "sample_rate",
            "channels",
            "duration_s",
            "num_samples",
            "rms",
            "peak_amplitude",
            "format_info",
            "status",
        ]

        rows: list[dict] = []

        for fp in audio_files:
            try:
                _, meta = load_audio(fp)
                rows.append(
                    {
                        "filepath": meta.filepath,
                        "filename": meta.filename,
                        "sample_rate": meta.sample_rate,
                        "channels": meta.channels,
                        "duration_s": meta.duration_seconds,
                        "num_samples": meta.num_samples,
                        "rms": meta.rms,
                        "peak_amplitude": meta.peak_amplitude,
                        "format_info": meta.format_info,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "filepath": str(fp),
                        "filename": fp.name,
                        "sample_rate": "",
                        "channels": "",
                        "duration_s": "",
                        "num_samples": "",
                        "rms": "",
                        "peak_amplitude": "",
                        "format_info": "",
                        "status": f"error: {exc}",
                    }
                )

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Metadata CSV written: %s (%d rows)", output_path, len(rows))
        return output_path
