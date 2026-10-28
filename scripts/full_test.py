"""手动执行当前全部数据的训练、验证和测试；--check-only 不启动模型或训练。"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from smoke_train import configure_console, preflight, print_plan

SPLIT_NAMES = {"train": "训练集", "val": "验证集", "test": "测试集"}


def print_metrics(reports: dict) -> None:
    print("\n三个集合的本次测试结果：", flush=True)
    for split in ("train", "val", "test"):
        report = reports[split]
        recording, event = report["recording"], report["event"]
        print(f"{SPLIT_NAMES[split]}：录音级准确率 {recording['accuracy']:.2%}，"
              f"精确率 {recording['precision']:.2%}，召回率 {recording['recall']:.2%}，"
              f"F1 {recording['f1']:.4f}，事件级F1 {event['f1']:.4f}", flush=True)


def run_full(settings, manifest: Path, rows: list[dict], plan: dict, output: Path) -> dict:
    import numpy as np
    from bolt_audio.classifier import LinearClassifier, train_classifier
    from bolt_audio.dataset import extract_manifest
    from bolt_audio.features import FeatureExtractor
    from bolt_audio.reporting import write_json

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    completed = []
    allow_related = plan["allow_related_splits"]
    purpose = "pipeline_test" if allow_related else "dataset_split_training"
    report = {"status": "running", "purpose": purpose, "recordings": len(rows),
              "allow_related_splits": allow_related,
              "source_manifest": str(manifest), "output": str(output), "steps_completed": completed}
    write_json(output / "RUNNING.json", report)
    write_json(output / "plan.json", plan)
    try:
        print(f"\n[1/3] 正在提取全部{len(rows)}条录音的特征……", flush=True)
        cache = output / "features.npz"
        extractor = FeatureExtractor(settings)
        extraction = extract_manifest(manifest, cache, extractor, allow_related_splits=allow_related)
        del extractor
        gc.collect()
        with np.load(cache, allow_pickle=False) as values:
            X, y = values["X"], values["y"]
        if X.shape != (plan["planned_windows"], plan["expected_feature_dimensions"]) or not np.isfinite(X).all():
            raise ValueError("特征缓存维数不符合预期，或存在无效数值")
        completed.append("feature_extraction")

        print("\n[2/3] 正在使用训练集拟合分类器，并用验证集选择阈值……", flush=True)
        model_dir = output / "model"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            artifact = train_classifier(cache, model_dir, settings.merge_gap_seconds,
                                        settings.min_positive_windows, allow_related_splits=allow_related)
        if captured:
            write_json(output / "training_warnings.json", [
                {"category": warning.category.__name__, "message": str(warning.message)} for warning in captured])
            print(f"训练产生{len(captured)}条依赖库提示，详细内容已写入 training_warnings.json。", flush=True)
        if set(artifact["reports"]) != {"train", "val", "test"}:
            raise ValueError("模型报告未覆盖训练集、验证集和测试集")
        completed.append("training_and_validation")
        print(f"本次选择的报警阈值：{artifact['threshold']:.6f}", flush=True)
        print_metrics(artifact["reports"])

        print("\n[3/3] 正在重新加载模型，检查全部窗口的预测输出……", flush=True)
        model_path = model_dir / "classifier.json"
        classifier = LinearClassifier.load(model_path)
        scores = classifier.predict_scores(X)
        if scores.shape != (len(y),) or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("重新加载的模型返回了无效分数")
        completed.append("model_reload")
        del X, y, scores
        gc.collect()

        report.update(status="passed", windows=extraction["windows"], split_counts=plan["split_counts"],
                      classifier=str(model_path), threshold=artifact["threshold"], metrics=artifact["reports"],
                      processing_seconds=time.perf_counter() - started)
        write_json(output / "full_report.json", report)
        (output / "RUNNING.json").unlink()
        print(f"\n全量流程测试完成\n测试报告：{output / 'full_report.json'}\n"
              f"模型文件：{model_path}\n部署推理已移至独立项目；请导出模型包后运行。", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), processing_seconds=time.perf_counter() - started)
        write_json(output / "full_report.json", report)
        print(f"本次执行失败，错误详情已记录：{output / 'full_report.json'}", file=sys.stderr, flush=True)
        raise


def main(argv=None) -> int:
    configure_console()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from bolt_audio.config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project / "configs/default.json", help="项目配置文件")
    parser.add_argument("--manifest", type=Path, default=project / "data/manifests/bolt_dataset_v2_all.jsonl", help="完整划分清单")
    parser.add_argument("--output-dir", type=Path, help="新输出目录，默认位于artifacts/full_runs下")
    parser.add_argument("--check-only", action="store_true", help="只读检查三组数据、环境和权重，不提取特征、不训练")
    parser.add_argument("--allow-related-splits", action="store_true", help="仅用于复现旧批次流程测试，允许相关来源跨集合")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config.resolve())
    manifest = args.manifest.resolve(strict=True)
    rows, plan = preflight(settings, manifest, full_data=True, allow_related_splits=args.allow_related_splits)
    print_plan(plan)
    if args.check_only:
        return 0
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    output = (args.output_dir or settings.project_dir / "artifacts/full_runs" / run_id).resolve()
    run_full(settings, manifest, rows, plan, output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"全量流程测试未完成：{error}", file=sys.stderr)
        raise SystemExit(1) from None
