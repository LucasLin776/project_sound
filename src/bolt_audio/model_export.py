"""Export numerical parameters and their matching PANNs checkpoint without training."""
from __future__ import annotations

import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .classifier import LinearClassifier
from .features import ACOUSTIC_NAMES, file_sha256
from .reporting import write_json

CHECKPOINT_NAME = "Cnn14_mAP=0.431.pth"
RUNTIME_KEYS = ("schema_version", "model_type", "feature_spec", "mean", "scale", "coefficient", "intercept",
                "threshold", "merge_gap_seconds", "min_positive_windows", "smoke_test_only",
                "deployment_validated", "evaluation_scope")


def export_model(classifier_path: Path, output: Path, checkpoint: Path | None = None):
    classifier_path = classifier_path.resolve(strict=True)
    classifier = LinearClassifier.load(classifier_path)
    source = classifier.artifact
    spec = source["feature_spec"]
    feature_set = spec.get("feature_set")
    if (spec.get("schema_version") != 1 or feature_set not in ("panns", "fusion", "acoustic")
            or spec.get("sample_rate") != 32000 or spec.get("gain_normalization") is not False
            or spec.get("channel_handling") != "mean_to_mono"):
        raise ValueError("不支持导出此模型的特征格式")
    if spec.get("acoustic_features") != ([] if feature_set == "panns" else ACOUSTIC_NAMES):
        raise ValueError("时频特征定义不一致")
    if len(classifier.mean) != {"fusion": 2060, "panns": 2048, "acoustic": 12}[feature_set]:
        raise ValueError("模型参数维数与特征格式不一致")
    window, hop = float(spec["window_seconds"]), float(spec["hop_seconds"])
    if not all(math.isfinite(x) for x in (window, hop)) or not (1 <= window <= 10 and 0 < hop <= window):
        raise ValueError("模型窗口或步长无效")
    if feature_set != "acoustic":
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("导出深度特征模型需要匹配的Cnn14预训练权重")
        if file_sha256(checkpoint) != spec.get("checkpoint_sha256"):
            raise ValueError("Cnn14权重与训练时的SHA-256不一致")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有模型目录，请换一个版本名：{output}")
    output.mkdir(parents=True)
    artifact = {key: source[key] for key in RUNTIME_KEYS if key in source}
    write_json(output / "classifier.json", artifact)
    files = {"classifier.json": {"sha256": file_sha256(output / "classifier.json")}}
    if feature_set != "acoustic":
        destination = output / CHECKPOINT_NAME
        shutil.copy2(checkpoint, destination)
        if file_sha256(destination) != spec["checkpoint_sha256"]:
            raise ValueError("导出的Cnn14权重复制校验失败")
        files[CHECKPOINT_NAME] = {"sha256": spec["checkpoint_sha256"]}
    write_json(output / "bundle.json", {"format": "wind-bolt-model", "schema_version": 1,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_classifier_sha256": file_sha256(classifier_path), "files": files,
               "feature_set": feature_set, "threshold": classifier.threshold, "training_started": False})
    return {"status": "exported", "model_dir": str(output), "files": list(files) + ["bundle.json"],
            "feature_set": feature_set, "training_started": False}
