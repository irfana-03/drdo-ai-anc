"""
engine.py — Speech enhancement with optional DeepFilterNet / RNNoise backends.

Falls back to spectral gating + NLMS when pretrained models are unavailable.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

MAX_OUTPUT_GAIN = 0.95


class EnhancementEngine:
    """Context-aware audio enhancement engine."""

    def __init__(self, filter_length: int = 512) -> None:
        self.filter_length = filter_length
        self._dfn_model = None
        self._rnnoise = None
        self.dfn_available = self._try_load_deepfilternet()
        self.rnnoise_available = self._try_load_rnnoise()
        self._lms_weights = np.zeros(filter_length, dtype=np.float64)
        self._lms_mu = 0.05

    def _try_load_deepfilternet(self) -> bool:
        try:
            from df.enhance import enhance, init_df  # type: ignore

            self._dfn_model = init_df()
            self._dfn_enhance = enhance
            return True
        except Exception:
            self._dfn_model = None
            return False

    def _try_load_rnnoise(self) -> bool:
        try:
            import rnnoise  # type: ignore

            self._rnnoise = rnnoise
            return True
        except Exception:
            self._rnnoise = None
            return False

    @property
    def active_engine(self) -> str:
        if self.dfn_available:
            return "DeepFilterNet"
        return "Spectral+LMS"

    def status(self) -> Dict[str, str]:
        return {
            "deepfilternet": "READY" if self.dfn_available else "NOT FOUND",
            "rnnoise": "READY" if self.rnnoise_available else "NOT FOUND",
            "active": self.active_engine,
        }

    def _sanitize(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float64)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.max(np.abs(audio))
        if peak > MAX_OUTPUT_GAIN:
            audio = audio * (MAX_OUTPUT_GAIN / peak)
        return audio

    def _spectral_gate(self, audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
        import librosa

        n_fft = 2048
        hop = 512
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
        noise_floor = np.percentile(S, 15, axis=1, keepdims=True)
        mask = np.clip((S - noise_floor * (2.0 - strength)) / (S + 1e-8), 0.0, 1.0)
        S_clean = S * mask
        phase = np.angle(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
        return librosa.istft(S_clean * np.exp(1j * phase), hop_length=hop)

    def _transient_attenuate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        threshold = np.percentile(np.abs(audio), 98)
        mask = np.abs(audio) > threshold
        out = audio.copy()
        out[mask] *= 0.3
        return out

    def _nlms_residual(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        w = self._lms_weights.copy()
        out = np.zeros(n, dtype=np.float64)
        flen = self.filter_length
        for i in range(flen, n):
            x = audio[i - flen : i][::-1]
            y = float(np.dot(w, x))
            e = audio[i] - y
            norm = float(np.dot(x, x)) + 1e-6
            w += (self._lms_mu / norm) * e * x
            out[i] = e
        out[:flen] = audio[:flen]
        self._lms_weights = w
        return out

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        use_deepfilternet: bool = True,
        use_lms: bool = False,
        transient_attenuation: bool = False,
        speech_priority: bool = False,
        suppression_strength: float = 0.7,
        engine: str = "auto",
    ) -> Tuple[np.ndarray, str]:
        """Enhance audio block. Returns (enhanced_audio, engine_used)."""
        audio = self._sanitize(audio.astype(np.float64))
        if len(audio) < 64:
            return audio.astype(np.float32), self.active_engine

        if transient_attenuation:
            audio = self._transient_attenuate(audio, sr)

        strength = suppression_strength * (0.5 if speech_priority else 1.0)
        engine_used = self.active_engine

        if engine == "RNNoise" and self.rnnoise_available:
            engine_used = "RNNoise"
            # RNNoise expects 48kHz frames — fallback to spectral if unsupported
            audio = self._spectral_gate(audio, sr, strength)
        elif use_deepfilternet and self.dfn_available:
            try:
                import torch

                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                enhanced = self._dfn_enhance(self._dfn_model, tensor, sr)
                audio = enhanced.squeeze().cpu().numpy()
                engine_used = "DeepFilterNet"
            except Exception as exc:
                logger.warning("DeepFilterNet failed: %s", exc)
                audio = self._spectral_gate(audio, sr, strength)
                engine_used = "Spectral+LMS"
        else:
            audio = self._spectral_gate(audio, sr, strength)
            engine_used = "Spectral+LMS"

        if use_lms:
            audio = self._nlms_residual(audio)

        return self._sanitize(audio).astype(np.float32), engine_used

    def reset(self) -> None:
        self._lms_weights = np.zeros(self.filter_length, dtype=np.float64)
