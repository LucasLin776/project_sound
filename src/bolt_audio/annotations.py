from __future__ import annotations

import csv
import io
import json
import math
import re
from decimal import Decimal
from pathlib import Path

from .dataset import load_manifest

AUDITION_COLUMNS = ["Name", "Start", "Duration", "Time Format", "Type", "Description"]
LABEL_FORMATS = ("auto", "audition", "audacity")


def _validate_events(events: list[dict], duration: float) -> list[dict]:
    if not events:
        raise ValueError("No bolt_drop regions found. Use --normal only after reviewing the entire normal recording")
    for event in events:
        start, end = event["start_s"], event["end_s"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError("Require 0 <= start < end <= audio duration; point labels are unsupported")
    events.sort(key=lambda event: event["start_s"])
    if any(current["start_s"] < previous["end_s"] for previous, current in zip(events, events[1:])):
        raise ValueError("Overlapping bolt_drop regions; merge labels for the same event first")
    return events


def _decimal_seconds(value: str, prefix: str) -> Decimal:
    """Audition decimal time: seconds, minutes:seconds, or hours:minutes:seconds."""
    value = value.strip()
    if not re.fullmatch(r"\d+(?::\d{1,2}){0,2}(?:\.\d+)?", value):
        raise ValueError(f"{prefix}: invalid decimal time {value!r}; use seconds, m:ss.sss or h:mm:ss.sss")
    parts = [Decimal(part) for part in value.split(":")]
    if any(part >= 60 for part in parts[1:]):
        raise ValueError(f"{prefix}: seconds and non-leading minutes must be below 60")
    seconds = Decimal(0)
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _read_audition(path: Path, duration: float) -> tuple[list[dict], list[dict]]:
    text = path.read_text(encoding="utf-8-sig").lstrip("\r\n")
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = "\t" if "\t" in first_line else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("No Audition marker header found")
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    required = {"Name", "Start", "Duration", "Time Format"}
    if len(set(reader.fieldnames)) != len(reader.fieldnames) or not required <= set(reader.fieldnames):
        raise ValueError(f"{path.name}: require unique Audition columns Name, Start, Duration, Time Format")
    events, markers = [], []
    for row in reader:
        prefix = f"{path.name}:{reader.line_num}"
        if None in row:
            raise ValueError(f"{prefix}: too many fields; check CSV delimiter and quoting")
        if not any(value and value.strip() for value in row.values()):
            continue
        row = {key: value.strip() if value is not None else "" for key, value in row.items()}
        if row["Name"] != "bolt_drop":
            raise ValueError(f"{prefix}: marker Name must be bolt_drop, got {row['Name']!r}")
        if row["Time Format"].casefold() != "decimal":
            raise ValueError(f"{prefix}: export Audition Time Format as decimal; samples and frame timecode are unsupported")
        start_decimal = _decimal_seconds(row["Start"], prefix)
        span_decimal = _decimal_seconds(row["Duration"], prefix)
        if span_decimal <= 0:
            raise ValueError(f"{prefix}: Duration must be positive; select a range rather than a point marker")
        end_decimal = start_decimal + span_decimal
        start, end = float(start_decimal), float(end_decimal)
        # Audition's millisecond rounding of Start and Duration can make an
        # end-of-file marker exceed the exact WAV duration by up to 1 ms.
        limit = Decimal(str(duration))
        clamped = limit < end_decimal <= limit + Decimal("0.001")
        if clamped:
            end = duration
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"{prefix}: marker outside audio duration ({duration:.9f}s)")
        events.append({"start_s": start, "end_s": end})
        markers.append({**row, "source_line": reader.line_num, "start_s": start, "end_s": end,
                        "end_clamped_to_audio": clamped})
    return _validate_events(events, duration), markers


def read_audition_labels(path: Path, duration: float) -> list[dict]:
    """Read Audition markers including BOM, Tab/comma separation and decimal times."""
    return _read_audition(path, duration)[0]


def read_label_file(path: Path, duration: float, label_format: str = "auto") -> tuple[list[dict], dict]:
    if label_format not in LABEL_FORMATS:
        raise ValueError(f"Unknown label format: {label_format}")
    if label_format == "auto":
        first_line = next((line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()), "")
        header = next(csv.reader([first_line], delimiter="\t" if "\t" in first_line else ","), [])
        label_format = "audition" if {"Name", "Start"} <= {name.strip() for name in header} else "audacity"
    if label_format == "audition":
        events, markers = _read_audition(path, duration)
        return events, {"tool": "adobe_audition_markers", "format": "audition", "markers": markers}
    return read_audacity_labels(path, duration), {"tool": "audacity_region_labels", "format": "audacity"}


def read_audacity_labels(path: Path, duration: float) -> list[dict]:
    """Read standard Audacity region labels: start TAB end TAB bolt_drop."""
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split("\t")
        prefix = f"{path.name}:{number}"
        if len(columns) != 3 or columns[2].strip() != "bolt_drop":
            raise ValueError(f"{prefix}: require start<TAB>end<TAB>bolt_drop; export standard region labels")
        start, end = float(columns[0]), float(columns[1])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"{prefix}: require 0 <= start < end <= audio duration; point labels are unsupported")
        events.append({"start_s": start, "end_s": end})
    return _validate_events(events, duration)


def import_labels(manifest: Path, recording_id: str, group_id: str, split: str,
                  labels: Path | None = None, normal: bool = False, replace: bool = False,
                  label_format: str = "auto"):
    """Explicit whole-recording review; preserve other rows, metadata and original audio."""
    if (labels is None) == (not normal):
        raise ValueError("Choose exactly one of a labels file or --normal")
    if not group_id.strip() or group_id == "REVIEW_INDEPENDENT_TRIAL":
        raise ValueError("Provide the actual independent trial/session group_id")
    if split not in ("train", "val", "test"):
        raise ValueError("Choose train, val or test")
    records = load_manifest(manifest)
    selected = next((row for row in records if row["recording_id"] == recording_id), None)
    if selected is None:
        raise ValueError(f"Unknown recording_id: {recording_id}")
    if selected["reviewed"] and not replace:
        raise ValueError("Recording already reviewed; use --replace to deliberately replace its annotations")
    events, annotation = ([], {"tool": "manual_normal_review"}) if normal else read_label_file(
        labels, selected["duration_s"], label_format)
    original = manifest.read_text(encoding="utf-8-sig")
    rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    row = next(row for row in rows if row["recording_id"] == recording_id)
    row.update(group_id=group_id, split=split, reviewed=True, events=events)
    row["annotation"] = {**annotation, "source": str(labels.resolve()) if labels else None}
    # Validate the complete proposed manifest before committing it, beside the
    # original so relative audio paths continue to resolve identically.
    from uuid import uuid4

    temporary = manifest.with_name(f"{manifest.name}.{uuid4().hex}.part")
    try:
        temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        load_manifest(temporary)
        backup = manifest.with_suffix(manifest.suffix + ".bak")
        backup.write_text(original, encoding="utf-8")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return {"recording_id": recording_id, "events": events, "reviewed": True,
            "group_id": group_id, "split": split, "label_format": annotation.get("format"),
            "manifest": str(manifest), "backup": str(backup)}


def _audition_timestamp(seconds: Decimal) -> str:
    minutes, remaining = divmod(seconds, Decimal(60))
    whole, _, fraction = format(remaining, "f").partition(".")
    return f"{int(minutes)}:{whole.zfill(2)}.{fraction.ljust(3, '0')}"


def export_labels(events: list[dict], path: Path, label_format: str = "auto"):
    """Export model proposals for correction, never mark them as reviewed automatically."""
    if label_format not in LABEL_FORMATS:
        raise ValueError(f"Unknown label format: {label_format}")
    if label_format == "auto":
        label_format = "audition" if path.suffix.lower() == ".csv" else "audacity"
    path.parent.mkdir(parents=True, exist_ok=True)
    if label_format == "audition":
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\r\n")
            writer.writerow(AUDITION_COLUMNS)
            for event in events:
                start, end = Decimal(str(event["start_s"])), Decimal(str(event["end_s"]))
                writer.writerow(["bolt_drop", _audition_timestamp(start), _audition_timestamp(end - start),
                                 "decimal", "Cue", "Model proposal; requires manual review"])
        return
    path.write_text("".join(f"{event['start_s']:.17g}\t{event['end_s']:.17g}\tbolt_drop\n" for event in events),
                    encoding="utf-8")
