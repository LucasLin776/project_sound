import json

import numpy as np
import pytest
import soundfile as sf

from bolt_audio.annotations import export_labels, import_labels, read_audacity_labels
from bolt_audio.audio import window_starts
from bolt_audio.classifier import train_classifier
from bolt_audio.config import Settings
from bolt_audio.dataset import extract_manifest, load_manifest, window_labels
from bolt_audio.evaluation import event_metrics
from bolt_audio.features import FeatureExtractor


def test_tail_coverage_and_short_recording():
    assert window_starts(300, 64, 16)[-1] + 64 == 300
    assert window_starts(20, 64, 16) == [0]
    with pytest.raises(ValueError):
        window_starts(0, 64, 16)


def test_event_windows_do_not_label_the_entire_recording():
    starts = np.asarray([0, 2, 4, 6], dtype=float)
    labels = window_labels(starts, starts + 2, [(2.1, 3)], guard_seconds=0.2)
    assert labels.tolist() == [-1, 1, 0, 0]


def test_group_and_identical_audio_leakage_rejected(tmp_path):
    waveform = np.random.default_rng(7).normal(0, 0.01, 32000).astype(np.float32)
    sf.write(tmp_path / "one.wav", waveform, 32000)
    sf.write(tmp_path / "two.wav", waveform, 32000)
    rows = [{"recording_id": str(index), "audio": name, "group_id": f"g{index}",
             "split": split, "reviewed": True, "events": []}
            for index, (name, split) in enumerate([("one.wav", "train"), ("two.wav", "val")])]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="identical audio crosses splits"):
        load_manifest(manifest)
    rows[1]["group_id"] = "g0"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="group_id crosses splits"):
        load_manifest(manifest)


def test_acoustic_training_and_evaluation(tmp_path, monkeypatch):
    """Synthetic sounds verify the software path, not bolt detection performance."""
    rng = np.random.default_rng(99)
    rows = []
    rate = 32000
    for split in ("train", "val", "test"):
        for target in (0, 1):
            audio = rng.normal(0, 0.003, rate * 4).astype(np.float32)
            if target:
                time = np.arange(len(audio)) / rate
                audio += (0.2 * np.sin(2 * np.pi * 3500 * time)).astype(np.float32)
            name = f"{split}_{target}.wav"
            sf.write(tmp_path / name, audio, rate)
            rows.append({"recording_id": name, "audio": name, "group_id": name,
                         "split": split, "reviewed": True,
                         "events": [{"start_s": 0, "end_s": 4}] if target else []})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    cache = tmp_path / "features.npz"
    extract_manifest(manifest, cache, FeatureExtractor(settings))
    artifact = train_classifier(cache, tmp_path / "model")
    assert set(artifact["reports"]) == {"train", "val", "test"}
    assert artifact["reports"]["test"]["event"]["true_positives"] == 1


def test_unreviewed_recording_cannot_be_used_for_training(tmp_path):
    sf.write(tmp_path / "reference.wav", np.zeros(32000), 32000)
    row = {"recording_id": "reference", "audio": "reference.wav", "group_id": "trial",
           "split": "unassigned", "reviewed": False, "events": []}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="review event intervals"):
        load_manifest(manifest, require_ready=True)


def test_uncertain_windows_are_retained_for_full_recording_evaluation(tmp_path):
    sf.write(tmp_path / "event.wav", np.random.default_rng(4).normal(0, 0.01, 128000), 32000)
    row = {"recording_id": "event", "audio": "event.wav", "group_id": "trial",
           "split": "train", "reviewed": True, "events": [{"start_s": 2.1, "end_s": 3.0}]}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row))
    settings = Settings(project_dir=tmp_path, checkpoint_dir=tmp_path, feature_set="acoustic")
    extract_manifest(manifest, tmp_path / "features.npz", FeatureExtractor(settings))
    with np.load(tmp_path / "features.npz", allow_pickle=False) as cache:
        assert cache["X"].shape[0] == 5
        assert cache["y"].tolist() == [-1, 1, 1, 1, 1]


def test_audacity_review_import_and_validation(tmp_path):
    sf.write(tmp_path / "audio.wav", np.zeros(128000), 32000)
    row = {"recording_id": "one", "audio": "audio.wav", "group_id": "REVIEW_INDEPENDENT_TRIAL",
           "split": "unassigned", "reviewed": False, "events": [], "notes": "preserve me"}
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text(json.dumps(row))
    labels = tmp_path / "audio.txt"
    original = manifest.read_bytes()
    for text in ("1\t1\tbolt_drop\n", "1\t5\tbolt_drop\n", "1\t2\tunknown\n", ""):
        labels.write_text(text)
        with pytest.raises(ValueError):
            import_labels(manifest, "one", "trial_1", "train", labels)
        assert manifest.read_bytes() == original
    labels.write_text("\ufeff1.2\t2.3\tbolt_drop\n", encoding="utf-8")
    import_labels(manifest, "one", "trial_1", "train", labels)
    rows = load_manifest(manifest, require_ready=True)
    assert rows[0]["intervals"] == [(1.2, 2.3)] and rows[0]["notes"] == "preserve me"
    assert manifest.with_suffix(".jsonl.bak").read_bytes() == original
    with pytest.raises(ValueError, match="already reviewed"):
        import_labels(manifest, "one", "trial_1", "train", normal=True)
    import_labels(manifest, "one", "trial_1", "train", normal=True, replace=True)
    assert load_manifest(manifest, require_ready=True)[0]["events"] == []
    export_labels([{"start_s": 0.01, "end_s": 4.0}], labels)
    assert read_audacity_labels(labels, 4.0) == [{"start_s": 0.01, "end_s": 4.0}]


def test_event_matching_rejects_wrong_time_and_duplicate_alarms():
    records = [{"recording_id": "one", "duration_s": 30.0, "event_intervals": [[10.0, 12.0]]}]
    wrong = event_metrics({"one": [{"start_s": 1.0, "end_s": 3.0}]}, records, 0.3)
    assert (wrong["true_positives"], wrong["false_positives"], wrong["false_negatives"]) == (0, 1, 1)
    duplicate = event_metrics({"one": [{"start_s": 10.0, "end_s": 12.0}] * 2}, records, 0.3)
    assert (duplicate["true_positives"], duplicate["false_positives"], duplicate["false_negatives"]) == (1, 1, 0)
    assert duplicate["matched_onset_mae_seconds"] == 0.0
