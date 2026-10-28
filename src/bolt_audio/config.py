from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("BOLT_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()


@dataclass(frozen=True)
class Settings:
    project_dir: Path
    checkpoint_dir: Path
    feature_set: str = "fusion"
    sample_rate: int = 32000
    window_seconds: float = 2.0
    hop_seconds: float = 0.5
    negative_guard_seconds: float = 0.2
    batch_size: int = 2
    threads: int = 4
    device: str = "cpu"
    merge_gap_seconds: float = 0.25
    min_positive_windows: int = 1
    max_audio_seconds: float = 60.0

    @classmethod
    def load(cls, path: Path | None = None, **overrides) -> "Settings":
        path = (path or project_root() / "configs/default.json").resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        data.update({key: value for key, value in overrides.items() if value is not None})
        root = Path(data.pop("project_dir", ".."))
        root = root if root.is_absolute() else path.parent / root
        root = root.resolve()
        for name in ("checkpoint_dir",):
            value = data.get(name)
            if value is not None:
                value = Path(value)
                data[name] = value.resolve() if value.is_absolute() else (root / value).resolve()
        settings = cls(project_dir=root, **data)
        settings.validate()
        return settings

    def validate(self):
        numbers = (self.window_seconds, self.hop_seconds, self.negative_guard_seconds,
                   self.merge_gap_seconds, self.max_audio_seconds)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Settings must be finite")
        if self.sample_rate != 32000:
            raise ValueError("The official checkpoints require a 32000 Hz model input")
        if not (1 <= self.window_seconds <= 10 and 0 < self.hop_seconds <= self.window_seconds):
            raise ValueError("Require 1 <= window_seconds <= 10 and 0 < hop_seconds <= window_seconds")
        if min(self.batch_size, self.threads, self.min_positive_windows) < 1:
            raise ValueError("Batch size, threads and min_positive_windows must be positive")
        if min(self.negative_guard_seconds, self.merge_gap_seconds) < 0 or self.max_audio_seconds <= 0:
            raise ValueError("Invalid duration settings")
        if self.feature_set not in ("fusion", "panns", "acoustic") or self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid feature_set or device")
