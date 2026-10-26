from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np

from .audio import read_audio, window_starts
from .config import Settings

ACOUSTIC_NAMES = [
    "rms_dbfs", "peak", "crest_factor", "kurtosis", "zero_crossing_rate",
    "centroid_hz", "bandwidth_hz", "energy_0_500", "energy_500_2000",
    "energy_2000_8000", "energy_8000_16000", "positive_spectral_flux",
]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acoustic_features(waveform: np.ndarray, sr=32000) -> np.ndarray:
    x = waveform.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    centered = x - x.mean()
    variance = float(np.mean(centered ** 2))
    kurtosis = float(np.mean(centered ** 4) / max(variance ** 2, 1e-24))
    magnitude = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=320))
    power = magnitude.astype(np.float64) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=1024)
    total = max(float(power.sum()), 1e-24)
    centroid = float((frequencies[:, None] * power).sum() / total)
    bandwidth = float(np.sqrt(((frequencies[:, None] - centroid) ** 2 * power).sum() / total))
    ratios = [float(power[(frequencies >= low) & (frequencies < high)].sum() / total)
              for low, high in [(0, 500), (500, 2000), (2000, 8000), (8000, 16001)]]
    normalized = magnitude / np.maximum(magnitude.sum(axis=0, keepdims=True), 1e-12)
    flux = float(np.maximum(np.diff(normalized, axis=1), 0).sum(axis=0).mean())
    return np.asarray([
        20 * np.log10(max(rms, 1e-12)), peak, peak / max(rms, 1e-12), kurtosis,
        float(np.mean(np.diff(np.signbit(waveform)))), centroid, bandwidth, *ratios, flux,
    ], dtype=np.float32)


class FeatureExtractor:
    """One model instance shared by multiple files or requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tagger = None
        checkpoint_hash = None
        if settings.feature_set != "acoustic":
            import torch
            from .vendor.panns.inference import AudioTagging

            torch.set_num_threads(settings.threads)
            checkpoint = settings.checkpoint_dir / "Cnn14_mAP=0.431.pth"
            checkpoint_hash = file_sha256(checkpoint)
            self.tagger = AudioTagging(checkpoint_path=checkpoint, device=settings.device)
        self.spec = {
            "schema_version": 1, "feature_set": settings.feature_set,
            "sample_rate": settings.sample_rate, "window_seconds": settings.window_seconds,
            "hop_seconds": settings.hop_seconds, "gain_normalization": False,
            "channel_handling": "mean_to_mono", "checkpoint_sha256": checkpoint_hash,
            "acoustic_features": ACOUSTIC_NAMES if settings.feature_set != "panns" else [],
        }

    def extract(self, path: Path) -> dict:
        cfg = self.settings
        waveform, quality = read_audio(path, cfg.sample_rate, cfg.max_audio_seconds)
        window, hop = round(cfg.window_seconds * cfg.sample_rate), round(cfg.hop_seconds * cfg.sample_rate)
        starts = window_starts(len(waveform), window, hop)
        feature_rows, all_tags = [], []
        for offset in range(0, len(starts), cfg.batch_size):
            batch = [np.pad(waveform[start:start + window], (0, max(0, start + window - len(waveform))))
                     for start in starts[offset:offset + cfg.batch_size]]
            batch = np.stack(batch)
            parts = []
            if self.tagger is not None:
                tags, embeddings = self.tagger.inference(batch)
                parts.append(embeddings)
                all_tags.append(tags)
            if cfg.feature_set != "panns":
                parts.append(np.stack([acoustic_features(row, cfg.sample_rate) for row in batch]))
            feature_rows.append(np.concatenate(parts, axis=1))
        features = np.concatenate(feature_rows).astype(np.float32)
        if not np.isfinite(features).all():
            raise RuntimeError("Non-finite features")
        return {
            "features": features, "quality": quality,
            "start_s": np.asarray(starts, dtype=np.float64) / cfg.sample_rate,
            "end_s": np.minimum((np.asarray(starts) + window) / cfg.sample_rate, quality["duration_seconds"]),
            "tag_scores": np.concatenate(all_tags) if all_tags else None,
        }
