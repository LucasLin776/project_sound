from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings


def dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="风机螺栓声音模型：标注、训练、评估与模型导出")
    parser.add_argument("--config", type=Path, help="默认使用configs/default.json")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("init-manifest", help="建立待标注录音清单")
    inventory.add_argument("--audio-dir", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    annotate = commands.add_parser("import-labels", help="导入Audition/Audacity标注或确认正常录音")
    annotate.add_argument("--manifest", required=True, type=Path)
    annotate.add_argument("--recording-id", required=True)
    source = annotate.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", type=Path)
    source.add_argument("--normal", action="store_true")
    annotate.add_argument("--group-id", required=True)
    annotate.add_argument("--split", required=True, choices=["train", "val", "test"])
    annotate.add_argument("--replace", action="store_true")
    annotate.add_argument("--label-format", default="auto", choices=["auto", "audition", "audacity"])
    check = commands.add_parser("check-data", help="检查标注和来源隔离")
    check.add_argument("--manifest", required=True, type=Path)
    extract = commands.add_parser("extract", help="从已标注录音提取训练特征")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    train = commands.add_parser("train", help="训练分类器并在验证集选择阈值")
    train.add_argument("--features", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--event-iou", type=float, default=0.3)
    export = commands.add_parser("export-model", help="导出独立推理模型包，不训练")
    export.add_argument("--classifier", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--checkpoint", type=Path, help="默认使用本项目checkpoints内的Cnn14权重")
    diagnose = commands.add_parser("diagnose", help="查看音频质量和特征，不执行故障推理")
    diagnose.add_argument("--audio", required=True, type=Path)
    diagnose.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    diagnose.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    settings = Settings.load(args.config, feature_set=getattr(args, "feature_set", None))
    if args.command == "init-manifest":
        from .dataset import initialize_manifest
        dump(initialize_manifest(args.audio_dir, args.output))
    elif args.command == "import-labels":
        from .annotations import import_labels
        dump(import_labels(args.manifest.resolve(strict=True), args.recording_id, args.group_id, args.split,
                           args.labels, args.normal, args.replace, args.label_format))
    elif args.command == "check-data":
        from .dataset import load_manifest
        rows = load_manifest(args.manifest.resolve())
        dump({"recordings": len(rows), "reviewed": sum(row["reviewed"] for row in rows),
              "ready_for_extraction": all(row["reviewed"] and row["split"] != "unassigned"
                                          and row["group_id"] != "REVIEW_INDEPENDENT_TRIAL" for row in rows),
              "recording_details": [{"id": row["recording_id"], "duration_s": row["duration_s"],
                                     "split": row["split"], "events": row["events"]} for row in rows]})
    elif args.command == "extract":
        from .dataset import extract_manifest, load_manifest
        from .features import FeatureExtractor
        load_manifest(args.manifest.resolve(strict=True), require_ready=True)
        dump(extract_manifest(args.manifest.resolve(strict=True), args.output, FeatureExtractor(settings)))
    elif args.command == "train":
        from .classifier import train_classifier
        artifact = train_classifier(args.features.resolve(strict=True), args.output_dir,
                                    settings.merge_gap_seconds, settings.min_positive_windows, args.event_iou)
        dump({"threshold": artifact["threshold"], "reports": artifact["reports"]})
    elif args.command == "export-model":
        from .model_export import CHECKPOINT_NAME, export_model
        dump(export_model(args.classifier, args.output_dir,
                          args.checkpoint or settings.checkpoint_dir / CHECKPOINT_NAME))
    else:
        from .features import FeatureExtractor
        from .reporting import write_json
        extractor = FeatureExtractor(settings)
        data = extractor.extract(args.audio.resolve(strict=True))
        result = {"audio_name": args.audio.name, "quality": data["quality"],
                  "feature_shape": list(data["features"].shape), "feature_spec": extractor.spec,
                  "notice": "音频质量与特征诊断；故障检测请使用独立推理项目"}
        write_json(args.output, result)
        dump(result)


if __name__ == "__main__":
    main()
