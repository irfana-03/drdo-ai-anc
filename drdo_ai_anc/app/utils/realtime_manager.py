"""Real-time audio capture and processing manager."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_LOG = PROJECT_ROOT / "results" / "realtime" / "sessions.csv"


class RealtimeManager:
    """Thread-safe microphone capture with pipeline processing."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self.sample_rate = 48000
        self.block_size = 4800
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.engine_choice = "auto"
        self._log_interval = 1.0
        self._last_log = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    def _log_session(self, state, mode: str = "live") -> None:
        SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not SESSION_LOG.exists()
        row = {
            "timestamp": datetime.now().isoformat(),
            "noise_context": state.noise_context,
            "confidence": round(state.confidence, 4),
            "impulsive_event": state.is_impulsive,
            "speech_probability": round(state.speech_probability, 4),
            "latency_ms": round(state.latency_ms, 2),
            "rtf": round(state.rtf, 4),
            "input_rms": round(state.input_rms, 6),
            "output_rms": round(state.output_rms, 6),
            "engine": state.engine,
            "mode": mode,
        }
        with open(SESSION_LOG, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if not self._running:
            return
        block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        with self._lock:
            state = self.pipeline.process_block(block, self.sample_rate, self.engine_choice)
            now = time.time()
            if now - self._last_log >= self._log_interval:
                self._log_session(state, "live")
                self._last_log = now

    def start(self, input_device: Optional[int] = None, block_size: int = 4800, sample_rate: int = 48000) -> bool:
        if self._running:
            return True
        try:
            import sounddevice as sd

            self.input_device = input_device
            self.block_size = block_size
            self.sample_rate = sample_rate
            self._running = True
            self._last_log = 0.0

            self._stream = sd.InputStream(
                device=input_device,
                channels=1,
                samplerate=sample_rate,
                blocksize=block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            return True
        except Exception as exc:
            self._running = False
            self.pipeline.state.running = False
            raise RuntimeError(f"Failed to start audio stream: {exc}") from exc

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.pipeline.state.running = False

    def reset(self) -> None:
        self.stop()
        self.pipeline.reset()

    def get_state(self):
        with self._lock:
            return self.pipeline.state

    def process_offline(self, audio: np.ndarray, sr: int) -> dict:
        with self._lock:
            result = self.pipeline.process_file(audio, sr, block_size=self.block_size, engine_choice=self.engine_choice)
            self._log_session(self.pipeline.state, "offline")
            return result

    @staticmethod
    def feedback_warning(input_device: Optional[int], output_device: Optional[int]) -> bool:
        if input_device is None or output_device is None:
            return False
        return input_device == output_device
