from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .classifier import LinearClassifier, merge_windows
from .config import Settings
from .features import FeatureExtractor, file_sha256
from .reporting import write_json
from .vendor.panns.config import labels


class UntrainedDetectorError(RuntimeError):
    pass


class BoltDetector:
    def __init__(self, settings: Settings):
        started = time.perf_counter()
        self.settings = settings
        self.extractor = FeatureExtractor(settings)
        self.classifier = LinearClassifier.load(settings.classifier_path) if settings.classifier_path else None
        self.classifier_sha256 = file_sha256(settings.classifier_path) if settings.classifier_path else None
        if self.classifier is not None and self.classifier.artifact["feature_spec"] != self.extractor.spec:
            raise ValueError("Classifier feature specification differs from runtime settings/checkpoint")
        self.model_load_seconds = time.perf_counter() - started

    def diagnose(self, path: Path) -> dict:
        """Detailed analysis, also usable before a target classifier has been trained."""
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

    def detect(self, path: Path, *, log_path: Path | None = None, audio_name: str | None = None,
               request_id: str | None = None) -> dict:
        """Public detection result; detailed scores and provenance go to a separate log."""
        request_id = request_id or uuid4().hex
        log_path = log_path or self.settings.project_dir / "logs/inference" / f"{request_id}.json"
        log = {
            "request_id": request_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "audio_name": audio_name or path.name,
            "classifier_path": str(self.settings.classifier_path) if self.settings.classifier_path else None,
            "classifier_sha256": self.classifier_sha256,
            "feature_spec": self.extractor.spec,
        }
        try:
            if self.classifier is None:
                raise UntrainedDetectorError("No bolt classifier loaded. Train a classifier and provide --classifier; "
                                             "use diagnose for untrained feature analysis.")
            details = self.diagnose(path)
            log.update(details)
            log["audio_name"] = audio_name or path.name
            output = {
                "bolt_detected": details["bolt_detected"],
                "events": [{"start_s": event["start_s"], "end_s": event["end_s"],
                            "score": event["max_score"]} for event in details["events"]],
                "score": max(window["score"] for window in details["windows"]),
                "quality": {key: details["quality"][key] for key in (
                    "duration_seconds", "sample_rate", "channels", "peak_amplitude", "rms_dbfs",
                    "full_scale_sample_fraction", "clipping_warning")},
                "processing_seconds": details["processing_seconds"],
            }
            log.update(status="ok", result=output,
                       score_definition="Maximum window classifier score, uncalibrated; not event probability",
                       timing_definition="Decode, resample, features and classification; excludes model load, upload and log IO")
        except Exception as error:
            log.update(status="error", error_type=type(error).__name__, error=str(error))
            write_json(log_path, log)
            raise
        write_json(log_path, log)
        return output
