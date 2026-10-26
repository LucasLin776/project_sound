from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .features import FeatureExtractor, file_sha256


def load_manifest(path: Path, require_ready=False) -> list[dict]:
    """One JSON object per recording; all event intervals must be reviewed."""
    records, ids, groups, hashes, paths = [], set(), {}, {}, set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        prefix = f"{path.name}:{line_number}"
        required = {"recording_id", "audio", "group_id", "split", "reviewed", "events"}
        if not required <= row.keys():
            raise ValueError(f"{prefix}: missing {sorted(required - row.keys())}")
        for key in ("recording_id", "audio", "group_id"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{prefix}: {key} must be a nonempty string")
        if row["recording_id"] in ids:
            raise ValueError(f"{prefix}: duplicate recording_id")
        ids.add(row["recording_id"])
        if row["split"] not in ("train", "val", "test", "unassigned") or type(row["reviewed"]) is not bool:
            raise ValueError(f"{prefix}: invalid split or reviewed flag")
        if require_ready and (not row["reviewed"] or row["split"] == "unassigned"):
            raise ValueError(f"{prefix}: review event intervals and assign train/val/test before extraction")
        if require_ready and row["group_id"] == "REVIEW_INDEPENDENT_TRIAL":
            raise ValueError(f"{prefix}: replace placeholder group_id with the actual independent trial/session")
        audio = Path(row["audio"])
        audio = (audio if audio.is_absolute() else path.parent / audio).resolve(strict=True)
        if audio in paths:
            raise ValueError(f"{prefix}: recording path is duplicated")
        paths.add(audio)
        info = sf.info(audio)
        if info.frames <= 0:
            raise ValueError(f"{prefix}: empty audio")
        if not isinstance(row["events"], list):
            raise ValueError(f"{prefix}: events must be a list")
        intervals = []
        for event in row["events"]:
            start, end = float(event["start_s"]), float(event["end_s"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= info.duration):
                raise ValueError(f"{prefix}: event interval outside audio duration")
            intervals.append((start, end))
        intervals.sort()
        if any(intervals[i][0] < intervals[i - 1][1] for i in range(1, len(intervals))):
            raise ValueError(f"{prefix}: overlapping event intervals; merge them first")
        digest = file_sha256(audio)
        for value, mapping, kind in [(row["group_id"], groups, "group_id"), (digest, hashes, "identical audio")]:
            previous = mapping.setdefault(value, row["split"])
            if previous != row["split"]:
                raise ValueError(f"{prefix}: {kind} crosses splits ({previous}, {row['split']})")
        records.append({**row, "path": audio, "duration_s": info.duration,
                        "sha256": digest, "intervals": intervals})
    if not records:
        raise ValueError("Manifest contains no recordings")
    return records


def window_labels(starts, ends, intervals, guard_seconds):
    """Any annotated overlap is positive; neighboring uncertain negatives are excluded."""
    labels = np.zeros(len(starts), dtype=np.int64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        if any(start < right and end > left for left, right in intervals):
            labels[index] = 1
        elif any(start < right + guard_seconds and end > left - guard_seconds for left, right in intervals):
            labels[index] = -1
    return labels


def extract_manifest(manifest: Path, output: Path, extractor: FeatureExtractor):
    records = load_manifest(manifest, require_ready=True)
    chunks, targets, starts, ends, recording_ids, group_ids, splits = [], [], [], [], [], [], []
    metadata = []
    for row in records:
        print(f"Extracting {row['recording_id']}: {row['path'].name}", flush=True)
        result = extractor.extract(row["path"])
        labels = window_labels(result["start_s"], result["end_s"], row["intervals"],
                               extractor.settings.negative_guard_seconds)
        # Retain uncertain windows for recording-level evaluation: production
        # inference sees these windows too. Only supervised fitting excludes them.
        count = len(labels)
        chunks.append(result["features"])
        targets.append(labels)
        starts.extend(result["start_s"])
        ends.extend(result["end_s"])
        recording_ids.extend([row["recording_id"]] * count)
        group_ids.extend([row["group_id"]] * count)
        splits.extend([row["split"]] * count)
        metadata.append({"recording_id": row["recording_id"], "group_id": row["group_id"],
                         "split": row["split"], "sha256": row["sha256"],
                         "audio": str(row["path"]), "duration_s": row["duration_s"],
                         "annotation": row.get("annotation"), "notes": row.get("notes", ""),
                         "quality": result["quality"], "event_intervals": row["intervals"],
                         "total_windows": count, "supervised_windows": int((labels >= 0).sum()),
                         "uncertain_windows": int((labels < 0).sum())})
    if any(row["supervised_windows"] == 0 for row in metadata):
        raise ValueError("A recording has no usable windows; check annotation guard intervals")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(
            stream, X=np.concatenate(chunks), y=np.concatenate(targets), start_s=starts, end_s=ends,
            recording_id=np.asarray(recording_ids), group_id=np.asarray(group_ids), split=np.asarray(splits),
            feature_spec=json.dumps(extractor.spec), records_json=json.dumps(metadata, ensure_ascii=False),
            negative_guard_seconds=extractor.settings.negative_guard_seconds,
        )
    return {"recordings": len(records), "windows": sum(len(target) for target in targets), "output": str(output)}


def initialize_manifest(audio_dir: Path, output: Path):
    """Inventory WAV recordings; labels and independent trial groupings require review."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    audio_dir = audio_dir.resolve(strict=True)
    paths = sorted(path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not paths:
        raise ValueError("No WAV recordings found")
    rows = [{"recording_id": f"recording_{index:05d}", "audio": str(path),
             "group_id": "REVIEW_INDEPENDENT_TRIAL", "split": "unassigned", "reviewed": False,
             "events": [], "notes": "Review the whole file, event intervals, independent trial group and split."}
            for index, path in enumerate(paths, 1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return {"recordings": len(rows), "output": str(output), "notice": "Pending manual annotation; not training data yet."}
