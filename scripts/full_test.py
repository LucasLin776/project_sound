"""手动执行当前全部数据的训练、验证和测试；--check-only 不启动模型或训练。"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import traceback
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from smoke_train import configure_console, preflight, print_plan, validate_detection_result

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
    from bolt_audio.detector import BoltDetector
    from bolt_audio.features import FeatureExtractor
    from bolt_audio.reporting import write_json

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    completed = []
    report = {"status": "running", "purpose": "pipeline_test", "recordings": len(rows),
              "source_manifest": str(manifest), "output": str(output), "steps_completed": completed}
    write_json(output / "RUNNING.json", report)
    write_json(output / "plan.json", plan)
    try:
        print(f"\n[1/4] 正在提取全部{len(rows)}条录音的特征……", flush=True)
        cache = output / "features.npz"
        extractor = FeatureExtractor(settings)
        extraction = extract_manifest(manifest, cache, extractor, allow_related_splits=True)
        del extractor
        gc.collect()
        with np.load(cache, allow_pickle=False) as values:
            X, y = values["X"], values["y"]
        if X.shape != (plan["planned_windows"], plan["expected_feature_dimensions"]) or not np.isfinite(X).all():
            raise ValueError("特征缓存维数不符合预期，或存在无效数值")
        completed.append("feature_extraction")

        print("\n[2/4] 正在使用训练集拟合分类器，并用验证集选择阈值……", flush=True)
        model_dir = output / "model"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            artifact = train_classifier(cache, model_dir, settings.merge_gap_seconds,
                                        settings.min_positive_windows, allow_related_splits=True)
        if captured:
            write_json(output / "training_warnings.json", [
                {"category": warning.category.__name__, "message": str(warning.message)} for warning in captured])
            print(f"训练产生{len(captured)}条依赖库提示，详细内容已写入 training_warnings.json。", flush=True)
        if set(artifact["reports"]) != {"train", "val", "test"}:
            raise ValueError("模型报告未覆盖训练集、验证集和测试集")
        completed.append("training_and_validation")
        print(f"本次选择的报警阈值：{artifact['threshold']:.6f}", flush=True)
        print_metrics(artifact["reports"])

        print("\n[3/4] 正在重新加载模型，检查全部窗口的预测输出……", flush=True)
        model_path = model_dir / "classifier.json"
        classifier = LinearClassifier.load(model_path)
        scores = classifier.predict_scores(X)
        if scores.shape != (len(y),) or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError("重新加载的模型返回了无效分数")
        completed.append("model_reload")
        del X, y, scores
        gc.collect()

        print(f"\n[4/4] 正在逐条检测全部{len(rows)}条WAV并生成结果……", flush=True)
        detector = BoltDetector(replace(settings, classifier_path=model_path))
        predictions = []
        with (output / "detection_summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["编号", "文件", "集合", "真实类别", "检测结果", "检测分数", "事件区间_秒", "处理耗时_秒", "结果文件"])
            for index, row in enumerate(rows, 1):
                result = detector.detect(row["path"], log_path=output / "logs" / f"{row['recording_id']}.json")
                validate_detection_result(result)
                destination = output / "predictions" / f"{row['recording_id']}.json"
                write_json(destination, result)
                intervals = "; ".join(f"{event['start_s']:.3f}~{event['end_s']:.3f}" for event in result["events"]) or "无"
                truth = "正例" if row["events"] else "反例"
                prediction = "有螺栓掉落" if result["bolt_detected"] else "无螺栓掉落"
                writer.writerow([row["recording_id"], row["path"].name, SPLIT_NAMES[row["split"]], truth, prediction,
                                 f"{result['score']:.6f}", intervals, f"{result['processing_seconds']:.3f}", str(destination)])
                stream.flush()
                predictions.append({"recording_id": row["recording_id"], "split": row["split"],
                                    "source_has_event": bool(row["events"]), "bolt_detected": result["bolt_detected"],
                                    "result": str(destination)})
                print(f"[逐条检测 {index}/{len(rows)}] {SPLIT_NAMES[row['split']]} {row['path'].name}："
                      f"标注{truth}，预测{prediction}，分数{result['score']:.4f}", flush=True)
        completed.append("all_wav_detection")
        report.update(status="passed", windows=extraction["windows"], split_counts=plan["split_counts"],
                      classifier=str(model_path), threshold=artifact["threshold"], metrics=artifact["reports"],
                      predictions=predictions, processing_seconds=time.perf_counter() - started)
        write_json(output / "full_report.json", report)
        (output / "RUNNING.json").unlink()
        print(f"\n全量流程测试完成\n测试报告：{output / 'full_report.json'}\n"
              f"逐条结果汇总：{output / 'detection_summary.csv'}\n模型文件：{model_path}", flush=True)
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
    parser.add_argument("--manifest", type=Path, default=project / "data/manifests/bolt_aug_v2_eq_all.jsonl", help="完整划分清单")
    parser.add_argument("--output-dir", type=Path, help="新输出目录，默认位于artifacts/full_runs下")
    parser.add_argument("--check-only", action="store_true", help="只读检查三组数据、环境和权重，不提取特征、不训练")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config.resolve())
    manifest = args.manifest.resolve(strict=True)
    rows, plan = preflight(settings, manifest, full_data=True)
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
