"""Preparation checks only: every test explicitly prevents feature extraction and model fitting."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from sklearn.linear_model import LogisticRegression

from bolt_audio.classifier import LinearClassifier, train_classifier
from bolt_audio.config import Settings
from bolt_audio.features import FeatureExtractor

_SPEC = importlib.util.spec_from_file_location("smoke_entry", Path(__file__).parents[1] / "scripts/smoke_train.py")
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture(autouse=True)
def forbid_training_and_extraction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation tests must not initialize an extractor or fit a classifier")

    monkeypatch.setattr(FeatureExtractor, "__init__", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def make_manifest(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for label in (0, 1):
        path = tmp_path / f"audio_{label}.wav"
        sf.write(path, rng.normal(0, .01, 32000 * 4), 32000)
        rows.append({"recording_id": str(label), "audio": path.name, "group_id": "same_training_group",
                     "split": "train", "reviewed": True,
                     "events": [{"start_s": 1., "end_s": 2.}] if label else []})
    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest, rows


def test_preflight_reads_metadata_without_model_or_training(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    rows, plan = smoke.preflight(settings, manifest)
    assert len(rows) == 2 and plan["planned_windows"] == 10
    assert plan["positive_recordings"] == plan["negative_recordings"] == 1
    assert plan["expected_feature_dimensions"] == 12
    assert plan["model_initialized"] is False and plan["training_started"] is False
    assert not (tmp_path / "artifacts").exists()


def test_check_only_never_calls_smoke_runner_or_creates_outputs(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"project_dir": str(tmp_path), "checkpoint_dir": str(tmp_path),
                                  "feature_set": "acoustic"}), encoding="utf-8")
    output = tmp_path / "would_train_here"
    with patch.object(smoke, "run_smoke", side_effect=AssertionError("Runner must not execute")):
        assert smoke.main(["--config", str(config), "--manifest", str(manifest),
                           "--output-dir", str(output), "--check-only"]) == 0
    assert not output.exists()


def test_preflight_rejects_validation_rows(tmp_path):
    manifest, rows = make_manifest(tmp_path)
    rows[0].update(split="val", group_id="separate")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    with pytest.raises(ValueError, match="train rows only"):
        smoke.preflight(settings, manifest)


def test_preflight_checks_checkpoint_without_loading_it(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="fusion")
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        smoke.preflight(settings, manifest)


def make_cache(tmp_path, splits):
    cache = tmp_path / "cache.npz"
    count = len(splits)
    np.savez(cache, X=np.arange(count * 2, dtype=float).reshape(count, 2),
             y=np.array([0, 1] * (count // 2)), split=splits,
             recording_id=[f"r{index}" for index in range(count)],
             group_id=[f"g{index}" for index in range(count)],
             start_s=np.zeros(count), end_s=np.ones(count),
             feature_spec=json.dumps({"feature_set": "acoustic"}),
             records_json=json.dumps([{"sha256": str(index), "split": split} for index, split in enumerate(splits)]))
    return cache


def test_normal_training_still_requires_validation_before_any_fit(tmp_path):
    cache = make_cache(tmp_path, ["train", "train"])
    with pytest.raises(ValueError, match="val split"):
        train_classifier(cache, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_smoke_training_rejects_val_test_before_any_fit(tmp_path):
    cache = make_cache(tmp_path, ["train", "train", "val", "val"])
    with pytest.raises(ValueError, match="train-only cache"):
        train_classifier(cache, tmp_path / "unused", smoke_test=True)
    assert not (tmp_path / "unused").exists()


def test_smoke_artifact_needs_explicit_loader_opt_in(tmp_path):
    path = tmp_path / "classifier.json"
    artifact = {"schema_version": 1, "model_type": "standard_scaler_logistic_regression",
                "mean": [0.], "scale": [1.], "coefficient": [.5], "intercept": 0.,
                "threshold": .5, "merge_gap_seconds": .25, "min_positive_windows": 1,
                "smoke_test_only": True}
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke tests only"):
        LinearClassifier.load(path)
    assert LinearClassifier.load(path, allow_smoke_test=True).threshold == .5


def test_output_check_accepts_either_prediction_without_accuracy_gate():
    for detected in (False, True):
        result = {"bolt_detected": detected, "events": [{"start_s": 1., "end_s": 2., "score": .8}] if detected else [],
                  "score": .8 if detected else .1, "quality": {"duration_seconds": 30.}, "processing_seconds": .2}
        smoke.validate_detection_result(result)
    result["events"][0]["end_s"] = 31.
    with pytest.raises(ValueError, match="outside the recording"):
        smoke.validate_detection_result(result)
