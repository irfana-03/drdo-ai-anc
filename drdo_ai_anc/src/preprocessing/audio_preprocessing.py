"""
audio_preprocessing.py — Safe preprocessing pipeline.

Responsibilities:
    - Convert any supported audio to mono.
    - Resample to the model-required sample rate (default 48 kHz).
    - Normalise amplitude (peak or RMS) safely.
    - Preserve original recordings — processed outputs go to a separate
      directory; raw data is **never** overwritten.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.audio_loader import load_audio, save_audio

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Stateless, configurable audio preprocessing pipeline."""

    def __init__(
        self,
        target_sr: int = 48000,
        mono: bool = True,
        normalization_method: str = "peak",
        normalization_target_db: float = -3.0,
    ) -> None:
        """
        Parameters
        ----------
        target_sr : int
            Target sample rate in Hz.
        mono : bool
            If True, mix to mono.
        normalization_method : str
            ``"peak"`` — scale so peak equals *normalization_target_db*.
            ``"rms"``  — scale so RMS equals *normalization_target_db*.
            ``"none"`` — do not normalise.
        normalization_target_db : float
            Target level in dBFS.
        """
        self.target_sr = target_sr
        self.mono = mono
        self.normalization_method = normalization_method.lower()
        self.normalization_target_db = normalization_target_db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        output_format: str = ".wav",
    ) -> Path:
        """Process a single audio file and write the result.

        Parameters
        ----------
        input_path : str or Path
            Source audio file.
        output_dir : str or Path
            Directory where the processed file will be saved.
        output_format : str
            Extension for the output file (default ``.wav``).

        Returns
        -------
        Path
            Path to the written processed file.
        """
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        waveform, meta = load_audio(
            input_path, target_sr=self.target_sr, mono=self.mono
        )

        waveform = self.normalize(waveform)

        output_name = input_path.stem + output_format
        output_path = output_dir / output_name

        save_audio(
            waveform,
            output_path,
            sample_rate=self.target_sr,
            overwrite=True,  # processed dir is ours to manage
        )

        logger.info("Processed %s → %s", input_path.name, output_path)
        return output_path

    def process_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Process an in-memory waveform (resample + mono + normalise).

        Parameters
        ----------
        waveform : np.ndarray
            Input waveform.
        sr : int
            Current sample rate.

        Returns
        -------
        np.ndarray
            Processed waveform at *target_sr*.
        """
        # Mono
        if self.mono and waveform.ndim > 1 and waveform.shape[-1] > 1:
            waveform = waveform.mean(axis=-1)

        # Resample
        if sr != self.target_sr:
            import librosa

            waveform = librosa.resample(
                waveform.astype(np.float32),
                orig_sr=sr,
                target_sr=self.target_sr,
            )

        waveform = self.normalize(waveform)
        return waveform.astype(np.float32)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Apply the configured normalisation to *waveform* (in-place safe).

        Parameters
        ----------
        waveform : np.ndarray
            1-D float waveform.

        Returns
        -------
        np.ndarray
            Normalised waveform.
        """
        if self.normalization_method == "none":
            return waveform

        target_linear = 10.0 ** (self.normalization_target_db / 20.0)

        if self.normalization_method == "peak":
            peak = np.max(np.abs(waveform))
            if peak < 1e-8:
                logger.warning("Waveform is near-silent; skipping normalisation.")
                return waveform
            waveform = waveform * (target_linear / peak)

        elif self.normalization_method == "rms":
            rms = np.sqrt(np.mean(waveform**2))
            if rms < 1e-8:
                logger.warning("Waveform RMS is near-zero; skipping normalisation.")
                return waveform
            waveform = waveform * (target_linear / rms)

        else:
            logger.warning(
                "Unknown normalisation method '%s'; skipping.",
                self.normalization_method,
            )

        return waveform
