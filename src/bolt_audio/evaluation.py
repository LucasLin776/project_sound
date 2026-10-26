from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def event_metrics(predictions: dict[str, list[dict]], records: list[dict], iou_threshold: float) -> dict:
    """One-to-one temporal IoU matching; a correct file alone is not a correct event."""
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("event_iou must be finite and in (0, 1]")
    tp = fp = fn = 0
    seconds = normal_seconds = 0.0
    normal_false_alarms = 0
    onset_errors, offset_errors = [], []
    details = []
    for record in records:
        predicted = predictions.get(record["recording_id"], [])
        truth = record["event_intervals"]
        duration = record["duration_s"] if "duration_s" in record else record["quality"]["duration_seconds"]
        seconds += duration
        if not truth:
            normal_seconds += duration
            normal_false_alarms += len(predicted)
        overlaps = np.zeros((len(truth), len(predicted)), dtype=float)
        for left, (start, end) in enumerate(truth):
            for right, event in enumerate(predicted):
                intersection = max(0.0, min(end, event["end_s"]) - max(start, event["start_s"]))
                union = end - start + event["end_s"] - event["start_s"] - intersection
                overlaps[left, right] = intersection / union
        matches = []
        if overlaps.size:
            # Cardinality has priority over summed IoU, so one long prediction
            # cannot count as multiple true events.
            valid = overlaps >= iou_threshold
            utility = valid * (1 + overlaps / (min(overlaps.shape) + 1))
            rows, columns = linear_sum_assignment(-utility)
            for left, right in zip(rows, columns):
                if valid[left, right]:
                    matches.append({"truth_index": int(left), "prediction_index": int(right),
                                    "iou": float(overlaps[left, right])})
                    onset_errors.append(abs(truth[left][0] - predicted[right]["start_s"]))
                    offset_errors.append(abs(truth[left][1] - predicted[right]["end_s"]))
        tp += len(matches)
        fp += len(predicted) - len(matches)
        fn += len(truth) - len(matches)
        details.append({"recording_id": record["recording_id"], "truth": truth,
                        "predictions": predicted, "matches": matches})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "matching": "one-to-one temporal IoU", "iou_threshold": iou_threshold,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "recorded_hours": seconds / 3600,
        "unmatched_predictions_per_recorded_hour": fp * 3600 / seconds if seconds else None,
        "normal_recording_hours": normal_seconds / 3600,
        "false_alarms_per_normal_recorded_hour": normal_false_alarms * 3600 / normal_seconds if normal_seconds else None,
        "matched_onset_mae_seconds": float(np.mean(onset_errors)) if onset_errors else None,
        "matched_offset_mae_seconds": float(np.mean(offset_errors)) if offset_errors else None,
        "recordings": details,
    }
