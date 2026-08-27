"""
audio_loader.py — Low-level audio I/O utilities.

Responsibilities:
    - Load an audio file from disk (any supported format).
    - Return waveform as a numpy array alongside metadata.
    - Provide a safe writer that *never* overwrites raw data.

All functions are format-agnostic via `soundfile` / `librosa`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Formats that soundfile / librosa can typically handle
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


@dataclass
class AudioMeta:
    """Metadata returned alongside a loaded waveform."""

    filepath: str
    filename: str
    sample_rate: int
    channels: int
    duration_seconds: float
    num_samples: int
    rms: float = 0.0
    peak_amplitude: float = 0.0
    format_info: str = ""
    errors: list[str] = field(default_factory=list)


def load_audio(
    filepath: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = False,
) -> Tuple[np.ndarray, AudioMeta]:
    """Load an audio file and return (waveform, metadata).

    Parameters
    ----------
    filepath : str or Path
        Path to an audio file.
    target_sr : int, optional
        If given, resample the audio to this rate.
    mono : bool
        If True, mix all channels to mono.

    Returns
    -------
    waveform : np.ndarray
        Shape ``(num_samples,)`` for mono or ``(num_samples, channels)`` for
        multi-channel.
    meta : AudioMeta
        Metadata about the loaded file.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    RuntimeError
        If the file cannot be decoded.
    """
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "Extension '%s' may not be supported; attempting load anyway.", ext
        )

    try:
        # soundfile returns (samples, channels) for multi-channel
        waveform, sr = sf.read(str(filepath), dtype="float64")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read audio file {filepath}: {exc}"
        ) from exc

    # Ensure 2-D shape: (samples, channels)
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]

    channels = waveform.shape[1]

    # Mix to mono if requested
    if mono and channels > 1:
        waveform = waveform.mean(axis=1, keepdims=True)
        channels = 1

    # Resample if requested
    if target_sr is not None and target_sr != sr:
        try:
            import librosa

            # librosa.resample expects (channels, samples) or (samples,)
            mono_flag = channels == 1
            if mono_flag:
                resampled = librosa.resample(
                    waveform[:, 0], orig_sr=sr, target_sr=target_sr
                )
                waveform = resampled[:, np.newaxis]
            else:
                resampled_channels = []
                for ch in range(channels):
                    resampled_channels.append(
                        librosa.resample(
                            waveform[:, ch], orig_sr=sr, target_sr=target_sr
                        )
                    )
                waveform = np.stack(resampled_channels, axis=1)
            sr = target_sr
        except ImportError:
            logger.error(
                "librosa is required for resampling but is not installed."
            )
            raise

    num_samples = waveform.shape[0]
    duration = num_samples / sr

    # Squeeze mono back to 1-D for convenience
    if channels == 1:
        waveform = waveform[:, 0]

    rms = float(np.sqrt(np.mean(waveform**2)))
    peak = float(np.max(np.abs(waveform)))

    # Format info from soundfile
    try:
        info = sf.info(str(filepath))
        fmt = f"{info.format} / {info.subtype}"
    except Exception:
        fmt = ext.lstrip(".")

    meta = AudioMeta(
        filepath=str(filepath),
        filename=filepath.name,
        sample_rate=sr,
        channels=channels,
        duration_seconds=round(duration, 4),
        num_samples=num_samples,
        rms=round(rms, 6),
        peak_amplitude=round(peak, 6),
        format_info=fmt,
    )

    return waveform.astype(np.float32), meta


def save_audio(
    waveform: np.ndarray,
    filepath: str | Path,
    sample_rate: int,
    *,
    overwrite: bool = False,
    subtype: str = "PCM_16",
) -> Path:
    """Write a waveform to disk.

    Parameters
    ----------
    waveform : np.ndarray
        1-D (mono) or 2-D (samples × channels).
    filepath : str or Path
        Destination path.
    sample_rate : int
        Sampling rate.
    overwrite : bool
        Must be ``True`` to overwrite an existing file.
    subtype : str
        PCM subtype for WAV output.

    Returns
    -------
    Path
        The absolute path to the written file.

    Raises
    ------
    FileExistsError
        If the file already exists and *overwrite* is False.
    """
    filepath = Path(filepath).resolve()
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists and overwrite=False: {filepath}"
        )
    filepath.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(filepath), waveform, sample_rate, subtype=subtype)
    logger.info("Saved audio to %s", filepath)
    return filepath


def scan_audio_files(
    directory: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """Return a sorted list of audio files in *directory*.

    Parameters
    ----------
    directory : str or Path
        Root directory to scan.
    recursive : bool
        If True, recurse into subdirectories.

    Returns
    -------
    list[Path]
        Sorted list of audio file paths.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        logger.warning("Not a directory: %s", directory)
        return []

    files: list[Path] = []
    pattern = "**/*" if recursive else "*"
    for p in directory.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)

    files.sort()
    logger.info("Found %d audio files in %s", len(files), directory)
    return files
