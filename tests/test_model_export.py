import json

import pytest

from bolt_audio.features import ACOUSTIC_NAMES
from bolt_audio.model_export import export_model


def model(tmp_path):
    path = tmp_path / "trained.json"
    artifact = {"schema_version": 1, "model_type": "standard_scaler_logistic_regression",
                "feature_spec": {"schema_version": 1, "feature_set": "acoustic", "sample_rate": 32000,
                                 "window_seconds": 2.0, "hop_seconds": 0.5, "gain_normalization": False,
                                 "channel_handling": "mean_to_mono", "checkpoint_sha256": None,
                                 "acoustic_features": ACOUSTIC_NAMES},
                "mean": [0.0] * 12, "scale": [1.0] * 12, "coefficient": [0.0] * 12,
                "intercept": 2.0, "threshold": 0.7, "merge_gap_seconds": 0.25, "min_positive_windows": 1,
                "reports": {"private": True}, "training_records": [{"audio": "private/path.wav"}]}
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_export_is_portable_and_omits_training_records(tmp_path):
    source = model(tmp_path)
    output = tmp_path / "bundle"
    result = export_model(source, output)
    assert result["training_started"] is False
    data = json.loads((output / "classifier.json").read_text(encoding="utf-8"))
    assert "reports" not in data and "training_records" not in data
    assert data["threshold"] == 0.7
    assert json.loads((output / "bundle.json").read_text(encoding="utf-8"))["format"] == "wind-bolt-model"
    with pytest.raises(FileExistsError):
        export_model(source, output)


def test_export_refuses_smoke_artifact(tmp_path):
    source = model(tmp_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    data["smoke_test_only"] = True
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke tests only"):
        export_model(source, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
