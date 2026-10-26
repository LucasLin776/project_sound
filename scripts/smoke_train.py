"""Manually run a small train-only PANNs pipeline check; --check-only never starts training."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4


def preflight(settings, manifest: Path) -> tuple[list[dict], dict]:
    """Read configuration, metadata and WAV headers; do not instantiate a model or fit anything."""
    import numpy as np
    from bolt_audio.audio import window_starts
    from bolt_audio.dataset import load_manifest, window_labels

    rows = load_manifest(manifest.resolve(strict=True), require_ready=True)
    if {row["split"] for row in rows} != {"train"}:
        raise ValueError("The smoke manifest must contain train rows only")
    classes = [bool(row["events"]) for row in rows]
    if set(classes) != {False, True}:
        raise ValueError("The smoke manifest needs event recordings and fully negative recordings")
    dependencies = {name: version(name) for name in
                    ("numpy", "scipy", "soundfile", "librosa", "scikit-learn", "torch", "torchlibrosa")}
    checkpoint = None
    if settings.feature_set != "acoustic":
        checkpoint = settings.checkpoint_dir / "Cnn14_mAP=0.431.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024 * 1024:
            raise FileNotFoundError(f"Missing or incomplete PANNs checkpoint: {checkpoint}")
    all_labels = []
    window, hop = round(settings.window_seconds * settings.sample_rate), round(settings.hop_seconds * settings.sample_rate)
    for row in rows:
        if row["duration_s"] > settings.max_audio_seconds:
            raise ValueError(f"Recording exceeds configured duration limit: {row['recording_id']}")
        frames = round(row["duration_s"] * settings.sample_rate)
        starts = np.asarray(window_starts(frames, window, hop), dtype=np.float64) / settings.sample_rate
        ends = np.minimum(starts + settings.window_seconds, row["duration_s"])
        all_labels.extend(window_labels(starts, ends, row["intervals"], settings.negative_guard_seconds).tolist())
    if not {0, 1} <= set(all_labels):
        raise ValueError("The selected windows must include both training classes")
    plan = {
        "status": "ready", "check_only": True, "training_started": False, "model_initialized": False,
        "manifest": str(manifest.resolve()), "python": sys.executable, "dependencies": dependencies,
        "feature_set": settings.feature_set, "device": settings.device,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "recordings": len(rows), "positive_recordings": sum(classes),
        "negative_recordings": len(rows) - sum(classes),
        "recording_ids": [row["recording_id"] for row in rows],
        "planned_windows": len(all_labels),
        "planned_window_labels": {str(label): all_labels.count(label) for label in (-1, 0, 1)},
        "expected_feature_dimensions": {"fusion": 2060, "panns": 2048, "acoustic": 12}[settings.feature_set],
        "threshold": .5, "performance_evaluation": False,
    }
    return rows, plan


def validate_detection_result(result: dict) -> None:
    """Success concerns executable output and its format, not whether the prediction is correct."""
    if not {"bolt_detected", "events", "score", "quality", "processing_seconds"} <= result.keys():
        raise ValueError("Detection result is missing required fields")
    if type(result["bolt_detected"]) is not bool or not isinstance(result["events"], list):
        raise ValueError("Invalid detection result types")
    if result["bolt_detected"] != bool(result["events"]):
        raise ValueError("Detection flag and event list disagree")
    if not math.isfinite(result["score"]) or not 0 <= result["score"] <= 1:
        raise ValueError("Detection score is invalid")
    if not math.isfinite(result["processing_seconds"]) or result["processing_seconds"] < 0:
        raise ValueError("Detection timing is invalid")
    duration = result["quality"]["duration_seconds"]
    for event in result["events"]:
        if not (0 <= event["start_s"] < event["end_s"] <= duration
                and math.isfinite(event["score"]) and 0 <= event["score"] <= 1):
            raise ValueError("Detection event is outside the recording or has an invalid score")


def run_smoke(settings, manifest: Path, rows: list[dict], plan: dict, output: Path) -> dict:
    """Called only by the user without --check-only; reuse the actual extract/train/detect path."""
    import numpy as np
    from bolt_audio.classifier import LinearClassifier, train_classifier
    from bolt_audio.dataset import extract_manifest
    from bolt_audio.detector import BoltDetector
    from bolt_audio.features import FeatureExtractor
    from bolt_audio.reporting import write_json

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(output / "RUNNING.json", {"status": "running", "smoke_test_only": True})
    write_json(output / "plan.json", plan)
    completed = []
    report = {"status": "running", "smoke_test_only": True, "performance_evaluated": False,
              "manifest": str(manifest.resolve()), "output": str(output.resolve()), "steps_completed": completed}
    try:
        print(f"[1/4] Extracting {settings.feature_set} features from {len(rows)} recordings...", flush=True)
        cache = output / "features.npz"
        extractor = FeatureExtractor(settings)
        extraction = extract_manifest(manifest, cache, extractor)
        del extractor
        gc.collect()
        with np.load(cache, allow_pickle=False) as values:
            X, y = values["X"], values["y"]
        if X.shape != (plan["planned_windows"], plan["expected_feature_dimensions"]) or not np.isfinite(X).all():
            raise ValueError("Feature cache has unexpected dimensions or non-finite values")
        completed.append("feature_extraction")

        print("[2/4] Fitting scaler + logistic classifier; fixed threshold 0.5...", flush=True)
        model_dir = output / "model"
        artifact = train_classifier(cache, model_dir, settings.merge_gap_seconds,
                                    settings.min_positive_windows, smoke_test=True)
        completed.append("training")

        print("[3/4] Reloading classifier JSON and checking numerical predictions...", flush=True)
        model_path = model_dir / "classifier.json"
        reloaded = LinearClassifier.load(model_path, allow_smoke_test=True)
        scores = reloaded.predict_scores(X)
        if scores.shape != (len(y),) or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("Reloaded classifier returned invalid scores")
        if not artifact["smoke_test_only"]:
            raise ValueError("Smoke classifier is missing its purpose marker")
        completed.append("model_reload")

        print("[4/4] Running the WAV detection interface on one positive and one negative file...", flush=True)
        detector = BoltDetector(replace(settings, classifier_path=model_path), allow_smoke_test=True)
        examples = [next(row for row in rows if row["events"]), next(row for row in rows if not row["events"])]
        predictions = []
        for row in examples:
            result = detector.detect(row["path"], log_path=output / "logs" / f"{row['recording_id']}.json")
            validate_detection_result(result)
            destination = output / "predictions" / f"{row['recording_id']}.json"
            write_json(destination, result)
            predictions.append({"recording_id": row["recording_id"], "source_has_event": bool(row["events"]),
                                "bolt_detected": result["bolt_detected"], "result": str(destination.resolve())})
        completed.append("wav_detection")
        report.update(status="passed", recordings=len(rows), windows=extraction["windows"],
                      feature_dimensions=X.shape[1], threshold=artifact["threshold"],
                      classifier=str(model_path.resolve()), predictions=predictions,
                      processing_seconds=time.perf_counter() - started,
                      notice="Pipeline completed. Prediction correctness is not a pass/fail condition.")
        write_json(output / "smoke_report.json", report)
        (output / "RUNNING.json").unlink()
        print(f"SMOKE TEST PASSED\nReport: {output / 'smoke_report.json'}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      processing_seconds=time.perf_counter() - started)
        write_json(output / "smoke_report.json", report)
        raise


def main(argv=None) -> int:
    project = Path(__file__).resolve().parents[1]
    # Resolve this checkout before an editable installation pointing to an older directory.
    sys.path.insert(0, str(project / "src"))
    from bolt_audio.config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project / "configs/default.json")
    parser.add_argument("--manifest", type=Path, default=project / "data/manifests/bolt_aug_v2_eq_smoke.jsonl")
    parser.add_argument("--output-dir", type=Path, help="New directory only; default: artifacts/smoke_runs/<run ID>")
    parser.add_argument("--check-only", action="store_true", help="Read-only preflight; no extraction, model loading or training")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config.resolve(), classifier_path=None)
    manifest = args.manifest.resolve(strict=True)
    rows, plan = preflight(settings, manifest)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return 0
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output = (args.output_dir or settings.project_dir / "artifacts/smoke_runs" / run_id).resolve()
    run_smoke(settings, manifest, rows, plan, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
