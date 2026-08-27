"""
feature_extraction.py — Audio feature extraction for analysis and modelling.

Supported features:
    - STFT (magnitude + phase)
    - Log-mel spectrogram
    - RMS energy
    - Zero-crossing rate
    - Spectral centroid
    - Spectral flux
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Parameters for feature extraction."""

    # STFT
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    window: str = "hann"

    # Mel
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 20000.0

    # RMS
    rms_frame_length: int = 2048
    rms_hop_length: int = 512

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureConfig":
        """Build from the ``features`` section of config.yaml."""
        stft = cfg.get("stft", {})
        mel = cfg.get("mel", {})
        rms = cfg.get("rms", {})
        return cls(
            n_fft=stft.get("n_fft", 2048),
            hop_length=stft.get("hop_length", 512),
            win_length=stft.get("win_length", 2048),
            window=stft.get("window", "hann"),
            n_mels=mel.get("n_mels", 128),
            fmin=mel.get("fmin", 20.0),
            fmax=mel.get("fmax", 20000.0),
            rms_frame_length=rms.get("frame_length", 2048),
            rms_hop_length=rms.get("hop_length", 512),
        )


class FeatureExtractor:
    """Extract spectral and time-domain features from a waveform."""

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.cfg = config or FeatureConfig()

    # ------------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------------

    def stft(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute the Short-Time Fourier Transform.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float waveform.
        sr : int
            Sample rate (unused here but kept for API symmetry).

        Returns
        -------
        np.ndarray
            Complex STFT matrix, shape ``(1 + n_fft/2, T)``.
        """
        import librosa

        return librosa.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.cfg.window,
        )

    # ------------------------------------------------------------------
    # Log-mel spectrogram
    # ------------------------------------------------------------------

    def log_mel_spectrogram(
        self, waveform: np.ndarray, sr: int
    ) -> np.ndarray:
        """Compute log-scaled mel spectrogram.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float waveform.
        sr : int
            Sample rate.

        Returns
        -------
        np.ndarray
            Log-mel spectrogram, shape ``(n_mels, T)``.
        """
        import librosa

        S = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.cfg.window,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
        )
        return librosa.power_to_db(S, ref=np.max)

    # ------------------------------------------------------------------
    # RMS energy
    # ------------------------------------------------------------------

    def rms(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute frame-level RMS energy.

        Returns
        -------
        np.ndarray
            RMS values, shape ``(1, T)``.
        """
        import librosa

        return librosa.feature.rms(
            y=waveform,
            frame_length=self.cfg.rms_frame_length,
            hop_length=self.cfg.rms_hop_length,
        )

    # ------------------------------------------------------------------
    # Zero-crossing rate
    # ------------------------------------------------------------------

    def zero_crossing_rate(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute frame-level zero-crossing rate.

        Returns
        -------
        np.ndarray
            ZCR values, shape ``(1, T)``.
        """
        import librosa

        return librosa.feature.zero_crossing_rate(
            y=waveform,
            frame_length=self.cfg.rms_frame_length,
            hop_length=self.cfg.rms_hop_length,
        )

    # ------------------------------------------------------------------
    # Spectral centroid
    # ------------------------------------------------------------------

    def spectral_centroid(
        self, waveform: np.ndarray, sr: int
    ) -> np.ndarray:
        """Compute frame-level spectral centroid.

        Returns
        -------
        np.ndarray
            Centroid in Hz, shape ``(1, T)``.
        """
        import librosa

        return librosa.feature.spectral_centroid(
            y=waveform,
            sr=sr,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
        )

    # ------------------------------------------------------------------
    # Spectral flux
    # ------------------------------------------------------------------

    def spectral_flux(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectral flux (onset strength).

        Spectral flux measures the change in spectral energy between
        consecutive frames — useful for detecting impulsive noise.

        Returns
        -------
        np.ndarray
            1-D array of spectral flux values per frame.
        """
        import librosa

        return librosa.onset.onset_strength(
            y=waveform,
            sr=sr,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
        )

    # ------------------------------------------------------------------
    # Convenience: extract all
    # ------------------------------------------------------------------

    def extract_all(
        self, waveform: np.ndarray, sr: int
    ) -> Dict[str, np.ndarray]:
        """Extract all supported features and return as a dict.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float waveform.
        sr : int
            Sample rate.

        Returns
        -------
        dict
            Keys: ``stft``, ``log_mel``, ``rms``, ``zcr``,
            ``spectral_centroid``, ``spectral_flux``.
        """
        return {
            "stft": self.stft(waveform, sr),
            "log_mel": self.log_mel_spectrogram(waveform, sr),
            "rms": self.rms(waveform, sr),
            "zcr": self.zero_crossing_rate(waveform, sr),
            "spectral_centroid": self.spectral_centroid(waveform, sr),
            "spectral_flux": self.spectral_flux(waveform, sr),
        }
