from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .classifier import LinearClassifier, merge_windows
from .config import Settings
from .features import FeatureExtractor
from .vendor.panns.config import labels


class BoltDetector:
    def __init__(self, settings: Settings):
        started = time.perf_counter()
        self.extractor = FeatureExtractor(settings)
        self.classifier = LinearClassifier.load(settings.classifier_path) if settings.classifier_path else None
        if self.classifier is not None and self.classifier.artifact["feature_spec"] != self.extractor.spec:
            raise ValueError("Classifier feature specification differs from runtime settings/checkpoint")
        self.model_load_seconds = time.perf_counter() - started

    def detect(self, path: Path) -> dict:
        started = time.perf_counter()
        result = self.extractor.extract(path)
        output = {
            "audio_name": path.name, "quality": result["quality"],
            "scope": "uploaded audio only; no conclusion for the 90-second recording gap",
            "bolt_detector_trained": self.classifier is not None,
            "deployment_validated": False, "bolt_detected": None, "events": [],
            "feature_shape": list(result["features"].shape), "feature_set": self.extractor.settings.feature_set,
            "model_load_seconds": self.model_load_seconds,
        }
        if self.classifier is None:
            output["notice"] = "No bolt classifier loaded. Generic scores cannot establish bolt dropping."
        else:
            artifact = self.classifier.artifact
            scores = self.classifier.predict_scores(result["features"])
            events = merge_windows(result["start_s"], result["end_s"], scores, self.classifier.threshold,
                                   artifact["merge_gap_seconds"], artifact["min_positive_windows"])
            output.update(bolt_detected=bool(events), events=events, threshold=self.classifier.threshold,
                          windows=[{"start_s": float(start), "end_s": float(end), "score": float(score)}
                                   for start, end, score in zip(result["start_s"], result["end_s"], scores)],
                          notice="Prototype classifier. Intervals are window coverage, not exact impact onset/offset.")
        if result["tag_scores"] is not None:
            maximum = result["tag_scores"].max(axis=0)
            output["generic_tags"] = [{"label": labels[index], "score": float(maximum[index])}
                                      for index in np.argsort(maximum)[-10:][::-1]]
        output["processing_seconds"] = time.perf_counter() - started
        return output
