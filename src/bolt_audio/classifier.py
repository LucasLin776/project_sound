from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .evaluation import event_metrics
from .reporting import write_json


def merge_windows(starts, ends, scores, threshold, gap, min_windows):
    groups = []
    for start, end, score in zip(starts, ends, scores):
        if score < threshold:
            continue
        if groups and start <= groups[-1]["end_s"] + gap:
            groups[-1]["end_s"] = max(groups[-1]["end_s"], float(end))
            groups[-1]["max_score"] = max(groups[-1]["max_score"], float(score))
            groups[-1]["positive_windows"] += 1
        else:
            groups.append({"start_s": float(start), "end_s": float(end),
                           "max_score": float(score), "positive_windows": 1})
    return [group for group in groups if group["positive_windows"] >= min_windows]


def metrics(y, predicted, scores=None):
    result = {"count": len(y), "confusion_matrix_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
              "precision": float(precision_score(y, predicted, zero_division=0)),
              "recall": float(recall_score(y, predicted, zero_division=0)),
              "f1": float(f1_score(y, predicted, zero_division=0))}
    if scores is not None and len(set(y)) == 2:
        result["average_precision"] = float(average_precision_score(y, scores))
    return result


def recording_metrics(y, scores, ids, starts, ends, threshold, gap, min_windows):
    truth, predicted = [], []
    for recording in np.unique(ids):
        mask = ids == recording
        order = np.argsort(starts[mask])
        events = merge_windows(starts[mask][order], ends[mask][order], scores[mask][order],
                               threshold, gap, min_windows)
        truth.append(int(y[mask].max()))
        predicted.append(int(bool(events)))
    return metrics(truth, predicted)


def temporal_metrics(scores, ids, starts, ends, records, threshold, gap, min_windows, event_iou):
    predictions = {}
    for recording in np.unique(ids):
        mask = ids == recording
        order = np.argsort(starts[mask])
        predictions[str(recording)] = merge_windows(starts[mask][order], ends[mask][order], scores[mask][order],
                                                   threshold, gap, min_windows)
    return event_metrics(predictions, records, event_iou)


class LinearClassifier:
    """JSON persistence avoids pickle; only numerical preprocessing and classifier parameters are loaded."""

    def __init__(self, artifact: dict):
        if artifact.get("schema_version") != 1 or artifact.get("model_type") != "standard_scaler_logistic_regression":
            raise ValueError("Unsupported classifier artifact")
        self.artifact = artifact
        self.mean = np.asarray(artifact["mean"], dtype=np.float64)
        self.scale = np.asarray(artifact["scale"], dtype=np.float64)
        self.coefficient = np.asarray(artifact["coefficient"], dtype=np.float64)
        self.intercept = float(artifact["intercept"])
        self.threshold = float(artifact["threshold"])
        if not (self.mean.ndim == 1 and len(self.mean) > 0 and self.mean.shape == self.scale.shape == self.coefficient.shape):
            raise ValueError("Invalid classifier dimensions")
        if not (np.isfinite([self.mean, self.scale, self.coefficient]).all()
                and np.isfinite(self.intercept) and (self.scale > 0).all() and 0 <= self.threshold <= 1):
            raise ValueError("Invalid classifier numerical parameters")
        gap = artifact.get("merge_gap_seconds", -1)
        count = artifact.get("min_positive_windows", 0)
        if not np.isfinite(gap) or gap < 0 or type(count) is not int or count < 1:
            raise ValueError("Invalid classifier alarm settings")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict_scores(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError("Features do not match the trained classifier")
        return expit(((features - self.mean) / self.scale) @ self.coefficient + self.intercept)


def train_classifier(cache_path: Path, output: Path, gap=0.25, min_windows=1, event_iou=0.3):
    if not np.isfinite(event_iou) or not 0 < event_iou <= 1:
        raise ValueError("event_iou must be finite and in (0, 1]")
    with np.load(cache_path, allow_pickle=False) as cache:
        X, y = cache["X"], cache["y"]
        splits, ids, groups = cache["split"], cache["recording_id"], cache["group_id"]
        starts, ends = cache["start_s"], cache["end_s"]
        spec = json.loads(str(cache["feature_spec"]))
        records = json.loads(str(cache["records_json"]))
    if (X.ndim != 2 or not np.isfinite(X).all() or not {0, 1} <= set(np.unique(y))
            or not set(np.unique(y)) <= {-1, 0, 1}):
        raise ValueError("Require finite features with positive and negative labels")
    if len({len(X), len(y), len(splits), len(ids), len(groups), len(starts), len(ends)}) != 1:
        raise ValueError("Feature cache columns have mismatched lengths")
    if not set(splits) <= {"train", "val", "test"}:
        raise ValueError("Cache contains unassigned splits")
    for keys in (groups, ids):
        if any(len(set(splits[keys == value])) > 1 for value in np.unique(keys)):
            raise ValueError("Group or recording leaks across dataset splits")
    hash_splits = {}
    for record in records:
        previous = hash_splits.setdefault(record["sha256"], record["split"])
        if previous != record["split"]:
            raise ValueError("Identical audio leaks across dataset splits")
    train, val = (splits == "train") & (y >= 0), splits == "val"
    for name, mask in [("train", train), ("val", val)]:
        if set(y[mask & (y >= 0)]) != {0, 1}:
            raise ValueError(f"The {name} split must have both positive and negative windows")
    val_truth = [int(y[val & (ids == recording)].max()) for recording in np.unique(ids[val])]
    if set(val_truth) != {0, 1}:
        raise ValueError("Validation must contain both event recordings and fully negative recordings")
    scaler = StandardScaler().fit(X[train])
    # Equal recording contributions avoid overweighting long recordings with many correlated windows.
    train_ids = ids[train]
    weights = np.asarray([1 / np.count_nonzero(train_ids == value) for value in train_ids])
    weights *= len(weights) / weights.sum()
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(scaler.transform(X[train]), y[train], sample_weight=weights)
    scores = model.predict_proba(scaler.transform(X))[:, 1]
    candidates = np.unique(np.r_[0.0, np.quantile(scores[val], np.linspace(0, 1, 201)), 1.0])
    best = None
    val_records = [record for record in records if record["split"] == "val"]
    for threshold in candidates:
        report = recording_metrics(y[val], scores[val], ids[val], starts[val], ends[val], threshold, gap, min_windows)
        temporal = temporal_metrics(scores[val], ids[val], starts[val], ends[val], val_records,
                                    threshold, gap, min_windows, event_iou)
        rank = (temporal["f1"], report["f1"], temporal["precision"], report["precision"], float(threshold))
        if best is None or rank > best[0]:
            best = (rank, float(threshold))
    threshold = best[1]
    reports = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        if mask.any():
            supervised = mask & (y >= 0)
            reports[split] = {
                "window": metrics(y[supervised], scores[supervised] >= threshold, scores[supervised])
                if supervised.any() else {"count": 0},
                "recording": recording_metrics(y[mask], scores[mask], ids[mask], starts[mask], ends[mask],
                                               threshold, gap, min_windows),
                "event": temporal_metrics(scores[mask], ids[mask], starts[mask], ends[mask],
                                          [record for record in records if record["split"] == split],
                                          threshold, gap, min_windows, event_iou),
                "group_count": len(np.unique(groups[mask])),
            }
    artifact = {
        "schema_version": 1, "model_type": "standard_scaler_logistic_regression",
        "feature_spec": spec, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "threshold": threshold, "merge_gap_seconds": gap, "min_positive_windows": min_windows,
        "threshold_selection": "maximum validation event F1; recording F1 then precision break ties; test unused",
        "event_iou": event_iou,
        "uncertain_windows": "Excluded from fitting/window metrics; included in recording-level alarm evaluation",
        "deployment_validated": False, "reports": reports, "training_records": records,
        "notice": "Prototype. Validation selected the threshold; test is the independent estimate. Overlapping windows are correlated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "classifier.json", artifact)
    write_json(output / "metrics.json", reports)
    return artifact
