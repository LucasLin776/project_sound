"""全量测试入口的准备检查：禁止初始化特征模型、提取特征或拟合参数。"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bolt_audio.classifier import LinearClassifier, metrics
from bolt_audio.config import Settings
from bolt_audio.dataset import load_manifest
from bolt_audio.features import FeatureExtractor

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location("full_entry", Path(__file__).parents[1] / "scripts/full_test.py")
full = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(full)
_SPLIT_SPEC = importlib.util.spec_from_file_location("split_entry", Path(__file__).parents[1] / "scripts/split_dataset.py")
splitter = importlib.util.module_from_spec(_SPLIT_SPEC)
_SPLIT_SPEC.loader.exec_module(splitter)


@pytest.fixture(autouse=True)
def forbid_model_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("准备检查不得初始化特征模型或拟合参数")

    monkeypatch.setattr(FeatureExtractor, "__init__", forbidden)
    monkeypatch.setattr(FeatureExtractor, "extract", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)


def batch_rows():
    rows, logs = [], []
    for index in range(1, 83):
        positive = index in (1, 10) if index <= 18 else index % 2 == 1
        row = {"recording_id": f"{index:02d}", "audio": f"audio/{index:02d}.wav", "group_id": "original_group",
               "split": "train", "reviewed": True, "notes": "original note",
               "events": [{"start_s": 1., "end_s": 2.}] if positive else []}
        details = {"pair_id": f"pair_{(index - 19) // 2}"} if index >= 19 else {}
        rows.append(row)
        logs.append({**row, "augmentation": details})
    return rows, logs


def test_split_covers_every_record_and_class_with_exact_counts():
    rows, logs = batch_rows()
    original = json.loads(json.dumps(rows))
    assigned, summary = splitter.create_splits(rows, logs)
    assert rows == original
    assert len(assigned) == 82 and len({row["recording_id"] for row in assigned}) == 82
    assert {key: (value["total"], value["positive"], value["negative"])
            for key, value in summary["splits"].items()} == {"train": (58, 24, 34), "val": (12, 5, 7), "test": (12, 5, 7)}
    assert {row["group_id"] for row in assigned} == {"original_group"}
    assert all(row["pipeline_test"] and row["source_split"] == "train" for row in assigned)
    assert [row["events"] for row in assigned] == [row["events"] for row in rows]


def test_split_is_reproducible_and_keeps_pairs_together():
    rows, logs = batch_rows()
    first, _ = splitter.create_splits(rows, logs, 17)
    second, _ = splitter.create_splits(rows, logs, 17)
    different, _ = splitter.create_splits(rows, logs, 18)
    assert first == second and first != different
    by_id = {row["recording_id"]: row["split"] for row in first}
    for index in range(19, 83, 2):
        assert by_id[f"{index:02d}"] == by_id[f"{index + 1:02d}"]


def make_manifest(tmp_path, *, related=False):
    rng = np.random.default_rng(44)
    rows = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            name = f"{split}_{label}.wav"
            sf.write(tmp_path / name, rng.normal(0, .01, 128000), 32000)
            group = "same_source_group" if related else f"source_{split}"
            rows.append({"recording_id": name, "audio": name, "group_id": group, "split": split,
                         "reviewed": True, "pipeline_test": True,
                         "events": [{"start_s": 1., "end_s": 2.}] if label else []})
    path = tmp_path / "all.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def test_full_preflight_accepts_three_splits_without_loading_model(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    rows, plan = full.preflight(settings, manifest, full_data=True)
    assert len(rows) == 6 and plan["planned_windows"] == 30
    assert plan["training_started"] is False and plan["model_initialized"] is False
    assert plan["performance_evaluation"] is True
    assert plan["allow_related_splits"] is False
    assert all(counts["positive"] == counts["negative"] == 1 for counts in plan["split_counts"].values())


def test_related_opt_in_preserves_other_manifest_checks(tmp_path):
    manifest, _ = make_manifest(tmp_path, related=True)
    with pytest.raises(ValueError, match="group_id crosses splits"):
        load_manifest(manifest, require_ready=True)
    assert len(load_manifest(manifest, require_ready=True, allow_related_splits=True)) == 6
    (tmp_path / "val_0.wav").write_bytes((tmp_path / "train_0.wav").read_bytes())
    with pytest.raises(ValueError, match="identical audio crosses splits"):
        load_manifest(manifest, require_ready=True, allow_related_splits=True)


def test_full_check_only_prints_chinese_and_never_calls_runner(tmp_path, capsys):
    manifest, _ = make_manifest(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"project_dir": str(tmp_path), "checkpoint_dir": str(tmp_path),
                                  "feature_set": "acoustic"}), encoding="utf-8")
    output = tmp_path / "unused_output"
    with patch.object(full, "run_full", side_effect=AssertionError("不得开始训练")):
        assert full.main(["--config", str(config), "--manifest", str(manifest),
                          "--output-dir", str(output), "--check-only"]) == 0
    text = capsys.readouterr().out
    assert "数据与环境预检通过" in text and "训练集：2条，正例1条，反例1条" in text
    assert "验证集：2条" in text and "测试集：2条" in text
    assert not output.exists()


def test_full_entry_requires_explicit_opt_in_for_related_sources(tmp_path):
    manifest, _ = make_manifest(tmp_path, related=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"project_dir": str(tmp_path), "checkpoint_dir": str(tmp_path),
                                  "feature_set": "acoustic"}), encoding="utf-8")
    args = ["--config", str(config), "--manifest", str(manifest), "--check-only"]
    with patch.object(full, "run_full", side_effect=AssertionError("不得开始训练")):
        with pytest.raises(ValueError, match="group_id crosses splits"):
            full.main(args)
        assert full.main([*args, "--allow-related-splits"]) == 0


@pytest.mark.parametrize("field", ["source_ids", "source_group_ids", "source_sha256"])
def test_preflight_rejects_shared_original_even_with_separate_output_groups(tmp_path, field):
    manifest, rows = make_manifest(tmp_path)
    rows[0][field] = ["shared_original"]
    rows[2][field] = ["shared_original"]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    with pytest.raises(ValueError, match="原始来源跨集合"):
        full.preflight(settings, manifest, full_data=True)


def test_full_preflight_requires_both_classes_in_each_split(tmp_path):
    manifest, rows = make_manifest(tmp_path)
    rows[-1]["events"] = []
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    with pytest.raises(ValueError, match="必须同时包含正例和反例"):
        full.preflight(settings, manifest, full_data=True)


def test_accuracy_reporting_and_pipeline_model_marker(tmp_path):
    assert metrics([0, 1], [1, 1])["accuracy"] == .5
    path = tmp_path / "classifier.json"
    path.write_text(json.dumps({"schema_version": 1, "model_type": "standard_scaler_logistic_regression",
                                "mean": [0.], "scale": [1.], "coefficient": [.5], "intercept": 0.,
                                "threshold": .5, "merge_gap_seconds": .25, "min_positive_windows": 1,
                                "evaluation_scope": "pipeline_test"}), encoding="utf-8")
    assert LinearClassifier.load(path).artifact["evaluation_scope"] == "pipeline_test"
