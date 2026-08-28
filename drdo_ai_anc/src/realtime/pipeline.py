"""
pipeline.py — End-to-end real-time audio processing pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.adaptive.controller import AdaptiveController
from src.classification.impulse_detector import ImpulseDetector
from src.classification.noise_classifier import NoiseContextClassifier
from src.enhancement.engine import EnhancementEngine
from src.preprocessing.feature_extraction import FeatureConfig, FeatureExtractor


@dataclass
class PipelineState:
    """Latest pipeline output for dashboard consumption."""

    noise_context: str = "WAITING FOR AUDIO"
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    speech_probability: float = 0.0
    is_impulsive: bool = False
    impulse_confidence: float = 0.0
    impulse_strength: float = 0.0
    strategy_name: str = ""
    strategy_description: str = ""
    engine: str = ""
    latency_ms: float = 0.0
    rtf: float = 0.0
    input_rms: float = 0.0
    output_rms: float = 0.0
    input_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0))
    output_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spectrogram: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    features: Dict[str, float] = field(default_factory=dict)
    running: bool = False
    history_context: List[str] = field(default_factory=list)
    history_impulsive: List[bool] = field(default_factory=list)
    history_timestamps: List[float] = field(default_factory=list)


class AudioPipeline:
    """Unified processing pipeline for live and offline modes."""

    def __init__(
        self,
        classifier: NoiseContextClassifier,
        feature_config: FeatureConfig,
        impulse_config: Optional[dict] = None,
        classifier_sr: int = 16000,
        processing_sr: int = 48000,
        clip_duration_s: float = 4.0,
        filter_length: int = 512,
    ) -> None:
        self.classifier = classifier
        self.classifier_sr = classifier_sr
        self.processing_sr = processing_sr
        self.clip_samples = int(clip_duration_s * classifier_sr)
        self.feature_extractor = FeatureExtractor(feature_config)
        impulse_cfg = impulse_config or {}
        self.impulse_detector = ImpulseDetector(
            rms_threshold=impulse_cfg.get("rms_threshold", 0.15),
            crest_factor_threshold=impulse_cfg.get("crest_factor_threshold", 8.0),
            flux_threshold=impulse_cfg.get("flux_threshold", 2.0),
            confidence_threshold=impulse_cfg.get("confidence_threshold", 0.5),
            feature_config=feature_config,
        )
        self.controller = AdaptiveController()
        self.enhancer = EnhancementEngine(filter_length=filter_length)
        self.state = PipelineState()
        self._display_samples = 4800  # ~100ms at 48kHz for waveform display

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or len(audio) == 0:
            return audio
        import librosa

        return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

    def _classify(self, audio: np.ndarray, sr: int) -> Dict:
        import librosa

        mono = audio.astype(np.float32)
        if sr != self.classifier_sr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=self.classifier_sr)

        if len(mono) > self.clip_samples:
            mono = mono[-self.clip_samples :]
        elif len(mono) < self.clip_samples:
            mono = np.pad(mono, (self.clip_samples - len(mono), 0))

        log_mel = self.feature_extractor.log_mel_spectrogram(mono, self.classifier_sr)
        tensor = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)
        result = self.classifier.predict(tensor)
        probs = dict(zip(self.classifier.class_names, result["probabilities"]))
        return {
            "noise_class": result["noise_class"],
            "confidence": result["confidence"],
            "probabilities": probs,
            "speech_probability": probs.get("SPEECH", 0.0),
        }

    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        rms = float(np.sqrt(np.mean(audio**2)))
        centroid = self.feature_extractor.spectral_centroid(audio, sr)
        flux = self.feature_extractor.spectral_flux(audio, sr)
        zcr = self.feature_extractor.zero_crossing_rate(audio, sr)
        return {
            "rms": round(rms, 6),
            "spectral_centroid": round(float(np.mean(centroid)), 2),
            "spectral_flux": round(float(np.mean(flux)), 4),
            "zero_crossing_rate": round(float(np.mean(zcr)), 4),
        }

    def _compute_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        if len(audio) < 256:
            return np.zeros((1, 1))
        S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
        return librosa.amplitude_to_db(S, ref=np.max)

    def process_block(
        self,
        audio: np.ndarray,
        sr: int,
        engine_choice: str = "auto",
    ) -> PipelineState:
        """Process one audio block and update state."""
        t0 = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) == 0:
            return self.state

        proc_audio = self._resample(audio, sr, self.processing_sr)

        classification = self._classify(proc_audio, self.processing_sr)
        impulse = self.impulse_detector.detect(proc_audio, self.processing_sr)
        features = self._extract_features(proc_audio, self.processing_sr)

        context = classification["noise_class"]
        strategy = self.controller.get_strategy(context)

        enhanced, engine_used = self.enhancer.enhance(
            proc_audio,
            self.processing_sr,
            use_deepfilternet=strategy.use_deepfilternet,
            use_lms=strategy.use_lms,
            transient_attenuation=strategy.transient_attenuation or impulse["is_impulsive"],
            speech_priority=strategy.speech_priority,
            suppression_strength=strategy.suppression_strength,
            engine=engine_choice,
        )

        elapsed = time.perf_counter() - t0
        block_duration = len(proc_audio) / self.processing_sr
        latency_ms = elapsed * 1000.0
        rtf = elapsed / block_duration if block_duration > 0 else 0.0

        display_in = proc_audio[-self._display_samples :]
        display_out = enhanced[-self._display_samples :]

        self.state = PipelineState(
            noise_context=context,
            confidence=classification["confidence"],
            probabilities=classification["probabilities"],
            speech_probability=classification["speech_probability"],
            is_impulsive=impulse["is_impulsive"],
            impulse_confidence=impulse["confidence"],
            impulse_strength=impulse["event_strength"],
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            engine=engine_used,
            latency_ms=latency_ms,
            rtf=rtf,
            input_rms=features["rms"],
            output_rms=float(np.sqrt(np.mean(enhanced**2))),
            input_buffer=display_in,
            output_buffer=display_out,
            spectrogram=self._compute_spectrogram(proc_audio, self.processing_sr),
            features=features,
            running=True,
            history_context=self.state.history_context[-199:] + [context],
            history_impulsive=self.state.history_impulsive[-199:] + [impulse["is_impulsive"]],
            history_timestamps=self.state.history_timestamps[-199:] + [time.time()],
        )
        return self.state

    def process_file(
        self,
        audio: np.ndarray,
        sr: int,
        block_size: int = 4800,
        engine_choice: str = "auto",
    ) -> Dict:
        """Process entire file in blocks (offline demo mode)."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        outputs: List[np.ndarray] = []
        t0 = time.perf_counter()

        for start in range(0, len(audio), block_size):
            block = audio[start : start + block_size]
            proc_block = self._resample(block, sr, self.processing_sr)
            classification = self._classify(proc_block, self.processing_sr)
            impulse = self.impulse_detector.detect(proc_block, self.processing_sr)
            strategy = self.controller.get_strategy(classification["noise_class"])
            enhanced, engine_used = self.enhancer.enhance(
                proc_block,
                self.processing_sr,
                use_deepfilternet=strategy.use_deepfilternet,
                use_lms=strategy.use_lms,
                transient_attenuation=strategy.transient_attenuation or impulse["is_impulsive"],
                speech_priority=strategy.speech_priority,
                suppression_strength=strategy.suppression_strength,
                engine=engine_choice,
            )
            # Resample back to original rate if needed
            if self.processing_sr != sr:
                import librosa
                enhanced = librosa.resample(enhanced, orig_sr=self.processing_sr, target_sr=sr)
            outputs.append(enhanced)

            # Update state with last block for UI
            features = self._extract_features(proc_block, self.processing_sr)
            elapsed = time.perf_counter() - t0
            block_duration = len(proc_block) / self.processing_sr
            self.state = PipelineState(
                noise_context=classification["noise_class"],
                confidence=classification["confidence"],
                probabilities=classification["probabilities"],
                speech_probability=classification["speech_probability"],
                is_impulsive=impulse["is_impulsive"],
                impulse_confidence=impulse["confidence"],
                impulse_strength=impulse["event_strength"],
                strategy_name=strategy.name,
                strategy_description=strategy.description,
                engine=engine_used,
                latency_ms=elapsed * 1000.0,
                rtf=elapsed / block_duration if block_duration > 0 else 0.0,
                input_rms=features["rms"],
                output_rms=float(np.sqrt(np.mean(enhanced**2))),
                input_buffer=proc_block[-self._display_samples :],
                output_buffer=enhanced[-self._display_samples :] if len(enhanced) >= self._display_samples else enhanced,
                spectrogram=self._compute_spectrogram(proc_block, self.processing_sr),
                features=features,
                running=False,
                history_context=self.state.history_context[-199:] + [classification["noise_class"]],
                history_impulsive=self.state.history_impulsive[-199:] + [impulse["is_impulsive"]],
                history_timestamps=self.state.history_timestamps[-199:] + [time.time()],
            )

        enhanced_full = np.concatenate(outputs) if outputs else audio
        if len(enhanced_full) > len(audio):
            enhanced_full = enhanced_full[: len(audio)]
        elif len(enhanced_full) < len(audio):
            enhanced_full = np.pad(enhanced_full, (0, len(audio) - len(enhanced_full)))

        total_time = time.perf_counter() - t0
        duration = len(audio) / sr

        return {
            "enhanced": enhanced_full,
            "input": audio,
            "sr": sr,
            "state": self.state,
            "latency_ms": self.state.latency_ms,
            "rtf": total_time / duration if duration > 0 else 0.0,
            "total_time_s": total_time,
        }

    def reset(self) -> None:
        self.enhancer.reset()
        self.state = PipelineState()
