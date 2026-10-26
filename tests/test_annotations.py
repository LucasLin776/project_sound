import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bolt_audio.annotations import export_labels, import_labels, read_audition_labels, read_label_file
from bolt_audio.cli import main
from bolt_audio.dataset import load_manifest


def write_markers(path, rows, delimiter="\t"):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(["Name", "Start", "Duration", "Time Format", "Type", "Description"])
        writer.writerows(rows)


@pytest.fixture
def pending_manifest(tmp_path):
    sf.write(tmp_path / "trial.wav", np.zeros(30 * 32000, dtype=np.float32), 32000)
    row = {"recording_id": "trial_001", "audio": "trial.wav", "group_id": "REVIEW_INDEPENDENT_TRIAL",
           "split": "unassigned", "reviewed": False, "events": [], "notes": "keep original notes"}
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    return manifest


def test_actual_audition_example_bom_and_duration():
    sample = Path(__file__).resolve().parents[1] / "examples/audition_markers.csv"
    assert sample.read_bytes().startswith(b"\xef\xbb\xbf")
    events, metadata = read_label_file(sample, 30.0)
    assert events == [{"start_s": 12.808, "end_s": 17.572}]
    assert metadata["format"] == "audition"
    assert metadata["markers"][0]["Duration"] == "0:04.764"
    assert metadata["markers"][0]["Type"] == "Cue"


def test_multiple_audition_regions_imported_through_cli(pending_manifest, tmp_path, monkeypatch, capsys):
    labels = tmp_path / "markers.csv"
    # Export order need not be chronological; label intervals must be sorted for training.
    write_markers(labels, [["bolt_drop", "0:23.100", "0:00.900", "decimal", "Cue", "第二次掉落"],
                          ["bolt_drop", "0:12.808", "0:04.764", "decimal", "Cue", "滚动, 碰撞"]])
    original = pending_manifest.read_bytes()
    monkeypatch.setattr(sys, "argv", ["bolt-audio", "import-labels", "--manifest", str(pending_manifest),
                                    "--recording-id", "trial_001", "--labels", str(labels),
                                    "--group-id", "session_01", "--split", "train"])
    main()
    output = json.loads(capsys.readouterr().out)
    assert output["label_format"] == "audition"
    assert output["events"] == [{"start_s": 12.808, "end_s": 17.572}, {"start_s": 23.1, "end_s": 24.0}]
    record = load_manifest(pending_manifest, require_ready=True)[0]
    assert record["intervals"] == [(12.808, 17.572), (23.1, 24.0)]
    assert record["notes"] == "keep original notes"
    assert record["annotation"]["tool"] == "adobe_audition_markers"
    assert record["annotation"]["markers"][0]["Description"] == "第二次掉落"
    assert pending_manifest.with_suffix(".jsonl.bak").read_bytes() == original


@pytest.mark.parametrize("start, span, expected, duration", [
    ("0:12.808", "0:04.764", (12.808, 17.572), 30.0),
    ("12.808", "4.764", (12.808, 17.572), 30.0),
    ("1:02:03.250", "0:00:00.125", (3723.25, 3723.375), 4000.0),
    ("0:29.000", "0:01.003", (29.0, 30.002721088435376), 30.002721088435376),
])
def test_decimal_time_and_millisecond_tail_rounding(tmp_path, start, span, expected, duration):
    labels = tmp_path / "times.csv"
    write_markers(labels, [["bolt_drop", start, span, "decimal", "Cue", ""]])
    assert read_audition_labels(labels, duration) == [{"start_s": expected[0], "end_s": expected[1]}]


@pytest.mark.parametrize("rows", [
    [["bolt_drop", "0:12.808", "0:00.000", "decimal", "Cue", "point"]],
    [["Marker 01", "0:12.808", "0:04.764", "decimal", "Cue", ""]],
    [["bolt_drop", "0:12.808", "0:04.764", "samples", "Cue", ""]],
    [["bolt_drop", "0:12.808", "0:04.764", "30 fps", "Cue", ""]],
    [["bolt_drop", "0:60.001", "0:01.000", "decimal", "Cue", ""]],
    [["bolt_drop", "-0.1", "0:01.000", "decimal", "Cue", ""]],
    [["bolt_drop", "NaN", "0:01.000", "decimal", "Cue", ""]],
    [["bolt_drop", "0:29.000", "0:01.002", "decimal", "Cue", "outside"]],
    [["bolt_drop", "0:12.000", "0:05.000", "decimal", "Cue", ""],
     ["bolt_drop", "0:16.000", "0:02.000", "decimal", "Cue", "overlap"]],
    [],
])
def test_bad_audition_markers_do_not_change_manifest(pending_manifest, tmp_path, rows):
    labels = tmp_path / "invalid.csv"
    write_markers(labels, rows)
    original = pending_manifest.read_bytes()
    with pytest.raises(ValueError):
        import_labels(pending_manifest, "trial_001", "session", "train", labels)
    assert pending_manifest.read_bytes() == original
    assert not pending_manifest.with_suffix(".jsonl.bak").exists()


def test_comma_csv_quoted_description_and_audition_export(tmp_path):
    labels = tmp_path / "comma.csv"
    description = '滚动, 碰撞\t"复听"'
    write_markers(labels, [["bolt_drop", "0:12.808", "0:04.764", "decimal", "Cue", description]], delimiter=",")
    events, metadata = read_label_file(labels, 30.0)
    assert metadata["markers"][0]["Description"] == description
    proposal = tmp_path / "proposals.csv"
    export_labels(events, proposal)
    assert read_audition_labels(proposal, 30.0) == events
    # Keep more than millisecond precision at the exact audio tail on export.
    tail = [{"start_s": 29.0, "end_s": 30.002721088435376}]
    export_labels(tail, proposal)
    assert read_audition_labels(proposal, tail[0]["end_s"]) == tail


def test_missing_or_duplicate_audition_columns_rejected(tmp_path):
    labels = tmp_path / "bad_header.csv"
    for text in ("Name\tStart\tTime Format\nbolt_drop\t0:01.000\tdecimal\n",
                 "Name\tName\tStart\tDuration\tTime Format\nbolt_drop\tx\t1\t1\tdecimal\n"):
        labels.write_text(text, encoding="utf-8-sig")
        with pytest.raises(ValueError, match="unique Audition columns"):
            read_label_file(labels, 30.0)
