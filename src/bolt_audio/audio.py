from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def window_starts(samples: int, window: int, hop: int) -> list[int]:
    if samples <= 0 or window <= 0 or not 0 < hop <= window:
        raise ValueError("Require nonempty audio and 0 < hop <= window")
    last = max(0, samples - window)
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def quality_report(audio: np.ndarray, info) -> dict:
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_U8": 8}.get(info.subtype)
    positive_rail = 1 - 2 ** (1 - bits) if bits else 1.0
    full_scale = float(np.mean((audio <= -1) | (audio >= positive_rail)))
    return {
        "sample_rate": info.samplerate, "channels": info.channels, "subtype": info.subtype,
        "duration_seconds": len(audio) / info.samplerate,
        "peak_amplitude": float(np.abs(audio).max()),
        "rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 1e-12))),
        "full_scale_sample_fraction": full_scale, "clipping_warning": full_scale > 0.001,
        "channel_handling": "mean_to_mono", "gain_normalization": False,
    }


def read_audio(path: Path, sample_rate=32000, max_seconds=60.0):
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0 or not 1 <= info.channels <= 8:
        raise ValueError("Audio must be nonempty and have between 1 and 8 channels")
    if info.duration > max_seconds:
        raise ValueError(f"Audio duration {info.duration:.3f}s exceeds {max_seconds}s")
    if info.samplerate > 192000:
        raise ValueError("Sample rates above 192 kHz are unsupported")
    original, rate = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(original).all():
        raise ValueError("Audio contains non-finite samples")
    quality = quality_report(original, info)
    waveform = original.mean(axis=1)
    if rate != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=sample_rate)
    return np.ascontiguousarray(waveform, dtype=np.float32), quality
