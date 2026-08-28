"""
impulse_detector.py — Detect impulsive events in real audio using measurable features.

Uses RMS, peak amplitude, crest factor, and spectral flux.
Does NOT claim gunshot detection — generic impulsive event detection only.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.preprocessing.feature_extraction import FeatureConfig, FeatureExtractor


class ImpulseDetector:
    """Rule-based impulsive event detector using spectral/temporal features."""

    def __init__(
        self,
        rms_threshold: float = 0.15,
        crest_factor_threshold: float = 8.0,
        flux_threshold: float = 2.0,
        confidence_threshold: float = 0.5,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.crest_factor_threshold = crest_factor_threshold
        self.flux_threshold = flux_threshold
        self.confidence_threshold = confidence_threshold
        self.extractor = FeatureExtractor(feature_config)

    def detect(self, waveform: np.ndarray, sr: int) -> Dict:
        """Detect impulsive events in a waveform.

        Returns
        -------
        dict
            is_impulsive, confidence, event_strength
        """
        waveform = waveform.astype(np.float64)
        rms = float(np.sqrt(np.mean(waveform**2)))
        peak = float(np.max(np.abs(waveform)))
        crest_factor = peak / (rms + 1e-10)

        flux = self.extractor.spectral_flux(waveform, sr)
        flux_peak = float(np.max(flux)) if len(flux) > 0 else 0.0
        flux_mean = float(np.mean(flux)) if len(flux) > 0 else 0.0

        # Normalised scores (0–1)
        rms_score = min(rms / self.rms_threshold, 1.0)
        crest_score = min(crest_factor / self.crest_factor_threshold, 1.0)
        flux_score = min(flux_peak / self.flux_threshold, 1.0)

        # Weighted combination
        confidence = 0.3 * rms_score + 0.4 * crest_score + 0.3 * flux_score
        event_strength = float(
            0.25 * rms + 0.35 * (crest_factor / 10.0) + 0.40 * flux_peak
        )

        is_impulsive = confidence >= self.confidence_threshold

        return {
            "is_impulsive": bool(is_impulsive),
            "confidence": round(confidence, 4),
            "event_strength": round(event_strength, 4),
            "features": {
                "rms": round(rms, 6),
                "peak_amplitude": round(peak, 6),
                "crest_factor": round(crest_factor, 4),
                "spectral_flux_peak": round(flux_peak, 4),
                "spectral_flux_mean": round(flux_mean, 4),
            },
        }
