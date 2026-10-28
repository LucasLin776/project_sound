"""手动运行小规模训练冒烟测试；--check-only 只检查数据与环境，不启动训练。"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4


def preflight(settings, manifest: Path, *, full_data=False, allow_related_splits=False) -> tuple[list[dict], dict]:
    """Read configuration, metadata and WAV headers; do not instantiate a model or fit anything."""
    import numpy as np
    from bolt_audio.audio import window_starts
    from bolt_audio.dataset import load_manifest, window_labels

    rows = load_manifest(manifest.resolve(strict=True), require_ready=True, allow_related_splits=allow_related_splits)
    if not allow_related_splits:
        sources = {}
        for row in rows:
            for field in ("source_ids", "source_group_ids", "source_sha256"):
                for source in row.get(field, []):
                    previous = sources.setdefault((field, source), row["split"])
                    if previous != row["split"]:
                        raise ValueError(f"原始来源跨集合：{field}={source}（{previous}，{row['split']}）")
    expected = {"train", "val", "test"} if full_data else {"train"}
    if {row["split"] for row in rows} != expected:
        raise ValueError("全量清单必须同时包含训练集、验证集、测试集" if full_data else "冒烟清单只能包含训练集录音")
    classes = [bool(row["events"]) for row in rows]
    if set(classes) != {False, True}:
        raise ValueError("测试清单必须同时包含正例和反例")
    for split in expected:
        if {bool(row["events"]) for row in rows if row["split"] == split} != {False, True}:
            raise ValueError(f"{split}集合必须同时包含正例和反例")
    dependencies = {name: version(name) for name in
                    ("numpy", "scipy", "soundfile", "librosa", "scikit-learn", "torch", "torchlibrosa")}
    checkpoint = None
    if settings.feature_set != "acoustic":
        checkpoint = settings.checkpoint_dir / "Cnn14_mAP=0.431.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024 * 1024:
            raise FileNotFoundError(f"PANNs权重文件缺失或不完整：{checkpoint}")
    all_labels, split_labels = [], {split: [] for split in expected}
    window, hop = round(settings.window_seconds * settings.sample_rate), round(settings.hop_seconds * settings.sample_rate)
    for row in rows:
        if row["duration_s"] > settings.max_audio_seconds:
            raise ValueError(f"录音超过配置的时长上限：{row['recording_id']}")
        frames = round(row["duration_s"] * settings.sample_rate)
        starts = np.asarray(window_starts(frames, window, hop), dtype=np.float64) / settings.sample_rate
        ends = np.minimum(starts + settings.window_seconds, row["duration_s"])
        labels = window_labels(starts, ends, row["intervals"], settings.negative_guard_seconds).tolist()
        all_labels.extend(labels)
        split_labels[row["split"]].extend(labels)
    if not {0, 1} <= set(all_labels):
        raise ValueError("所选窗口必须同时包含正例和反例")
    for split, labels in split_labels.items():
        if not {0, 1} <= set(labels):
            raise ValueError(f"{split}集合的窗口必须同时包含正例和反例")
    plan = {
        "status": "ready", "check_only": True, "training_started": False, "model_initialized": False,
        "manifest": str(manifest.resolve()), "python": sys.executable, "dependencies": dependencies,
        "feature_set": settings.feature_set, "device": settings.device,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "recordings": len(rows), "positive_recordings": sum(classes),
        "negative_recordings": len(rows) - sum(classes),
        "multi_event_recordings": sum(len(row["events"]) > 1 for row in rows),
        "allow_related_splits": bool(allow_related_splits),
        "recording_ids": [row["recording_id"] for row in rows],
        "planned_windows": len(all_labels),
        "planned_window_labels": {str(label): all_labels.count(label) for label in (-1, 0, 1)},
        "expected_feature_dimensions": {"fusion": 2060, "panns": 2048, "acoustic": 12}[settings.feature_set],
        "threshold": None if full_data else .5, "performance_evaluation": bool(full_data),
        "split_counts": {split: {"recordings": sum(row["split"] == split for row in rows),
                                 "positive": sum(row["split"] == split and bool(row["events"]) for row in rows),
                                 "negative": sum(row["split"] == split and not row["events"] for row in rows),
                                 "windows": len(split_labels[split])}
                         for split in ("train", "val", "test") if split in expected},
    }
    return rows, plan


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def print_plan(plan: dict) -> None:
    print("数据与环境预检通过。", flush=True)
    print(f"数据清单：{plan['manifest']}")
    print(f"Python解释器：{plan['python']}")
    features = {"fusion": "PANNs深度特征 + 时频统计", "panns": "PANNs深度特征", "acoustic": "时频统计"}
    print(f"特征方案：{features[plan['feature_set']]}，设备：{plan['device']}")
    print(f"录音总数：{plan['recordings']}，正例：{plan['positive_recordings']}，反例：{plan['negative_recordings']}")
    print(f"含多个故障时间段的录音：{plan['multi_event_recordings']}条")
    print("来源隔离：允许相关来源跨集合（仅供历史流程测试）" if plan["allow_related_splits"]
          else "来源隔离：已启用，原始来源不得跨集合")
    for split, counts in plan["split_counts"].items():
        name = {"train": "训练集", "val": "验证集", "test": "测试集"}[split]
        print(f"{name}：{counts['recordings']}条，正例{counts['positive']}条，反例{counts['negative']}条")
    print(f"预计窗口数：{plan['planned_windows']}，每个窗口特征维数：{plan['expected_feature_dimensions']}")
    print("阈值设置：由验证集选择" if plan["performance_evaluation"] else "阈值设置：固定为0.5")
    print("预检未初始化特征模型，未提取特征，未启动训练。", flush=True)


def run_smoke(settings, manifest: Path, rows: list[dict], plan: dict, output: Path) -> dict:
    """Called only by the user without --check-only; check extraction, fitting and model reload."""
    import numpy as np
    from bolt_audio.classifier import LinearClassifier, train_classifier
    from bolt_audio.dataset import extract_manifest
    from bolt_audio.features import FeatureExtractor
    from bolt_audio.reporting import write_json

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(output / "RUNNING.json", {"status": "running", "smoke_test_only": True})
    write_json(output / "plan.json", plan)
    completed = []
    report = {"status": "running", "smoke_test_only": True, "performance_evaluated": False,
              "manifest": str(manifest.resolve()), "output": str(output.resolve()), "steps_completed": completed}
    try:
        print(f"[1/3] 正在从{len(rows)}条录音提取特征……", flush=True)
        cache = output / "features.npz"
        extractor = FeatureExtractor(settings)
        extraction = extract_manifest(manifest, cache, extractor)
        del extractor
        gc.collect()
        with np.load(cache, allow_pickle=False) as values:
            X, y = values["X"], values["y"]
        if X.shape != (plan["planned_windows"], plan["expected_feature_dimensions"]) or not np.isfinite(X).all():
            raise ValueError("特征缓存维数不符合预期，或含有无效数值")
        completed.append("feature_extraction")

        print("[2/3] 正在拟合标准化参数和分类器，阈值固定为0.5……", flush=True)
        model_dir = output / "model"
        artifact = train_classifier(cache, model_dir, settings.merge_gap_seconds,
                                    settings.min_positive_windows, smoke_test=True)
        completed.append("training")

        print("[3/3] 正在重新加载模型文件并检查预测分数……", flush=True)
        model_path = model_dir / "classifier.json"
        reloaded = LinearClassifier.load(model_path, allow_smoke_test=True)
        scores = reloaded.predict_scores(X)
        if scores.shape != (len(y),) or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("重新加载的分类器返回了无效分数")
        if not artifact["smoke_test_only"]:
            raise ValueError("模型缺少冒烟测试标记")
        completed.append("model_reload")

        report.update(status="passed", recordings=len(rows), windows=extraction["windows"],
                      feature_dimensions=X.shape[1], threshold=artifact["threshold"],
                      classifier=str(model_path.resolve()),
                      processing_seconds=time.perf_counter() - started,
                      notice="Pipeline completed. Prediction correctness is not a pass/fail condition.")
        write_json(output / "smoke_report.json", report)
        (output / "RUNNING.json").unlink()
        print(f"冒烟测试通过\n测试报告：{output / 'smoke_report.json'}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      processing_seconds=time.perf_counter() - started)
        write_json(output / "smoke_report.json", report)
        raise


def main(argv=None) -> int:
    configure_console()
    project = Path(__file__).resolve().parents[1]
    # Resolve this checkout before an editable installation pointing to an older directory.
    sys.path.insert(0, str(project / "src"))
    from bolt_audio.config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project / "configs/default.json", help="项目配置文件")
    parser.add_argument("--manifest", type=Path, default=project / "data/manifests/bolt_aug_v2_eq_smoke.jsonl", help="冒烟数据清单")
    parser.add_argument("--output-dir", type=Path, help="新输出目录，默认位于artifacts/smoke_runs下")
    parser.add_argument("--check-only", action="store_true", help="只读预检，不加载模型、不提取特征、不训练")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config.resolve())
    manifest = args.manifest.resolve(strict=True)
    rows, plan = preflight(settings, manifest)
    print_plan(plan)
    if args.check_only:
        return 0
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output = (args.output_dir or settings.project_dir / "artifacts/smoke_runs" / run_id).resolve()
    run_smoke(settings, manifest, rows, plan, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
