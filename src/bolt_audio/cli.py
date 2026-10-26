from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from .config import Settings


def dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="Wind turbine bolt audio project")
    parser.add_argument("--config", type=Path, help="Defaults to configs/default.json")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("init-manifest", help="Inventory WAV files into an unreviewed annotation manifest")
    inventory.add_argument("--audio-dir", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    annotate = commands.add_parser("import-labels", help="Import reviewed Audition/Audacity labels or confirm normal audio")
    annotate.add_argument("--manifest", required=True, type=Path)
    annotate.add_argument("--recording-id", required=True)
    annotation_source = annotate.add_mutually_exclusive_group(required=True)
    annotation_source.add_argument("--labels", type=Path, help="Audition CSV markers or Audacity region labels (UTF-8)")
    annotation_source.add_argument("--normal", action="store_true", help="Confirm the whole recording has no bolt events")
    annotate.add_argument("--group-id", required=True, help="Actual independent trial/session, shared by related files")
    annotate.add_argument("--split", required=True, choices=["train", "val", "test"])
    annotate.add_argument("--replace", action="store_true", help="Deliberately replace an already reviewed annotation")
    annotate.add_argument("--label-format", default="auto", choices=["auto", "audition", "audacity"],
                          help="Default: recognize the label format from its contents")
    check = commands.add_parser("check-data", help="Check annotations, audio and group separation")
    check.add_argument("--manifest", required=True, type=Path)
    extract = commands.add_parser("extract", help="Extract training features from reviewed recordings")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    train = commands.add_parser("train", help="Train a classifier and select threshold on validation recordings")
    train.add_argument("--features", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--event-iou", type=float, default=0.3, help="Temporal IoU needed to match an event (default: 0.3)")
    infer = commands.add_parser("infer", help="Detect bolt events with a trained classifier")
    infer.add_argument("--audio", required=True, type=Path)
    infer.add_argument("--classifier", type=Path)
    infer.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--log-output", type=Path, help="Detailed log (default: logs/inference/<request_id>.json)")
    infer.add_argument("--labels-output", type=Path, help="Export proposals: .csv for Audition, .txt for Audacity")
    infer.add_argument("--label-format", default="auto", choices=["auto", "audition", "audacity"],
                       help="Proposal format; auto uses the labels-output extension")
    diagnose = commands.add_parser("diagnose", help="Detailed feature analysis; can run before target training")
    diagnose.add_argument("--audio", required=True, type=Path)
    diagnose.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    diagnose.add_argument("--output", required=True, type=Path)
    serve = commands.add_parser("serve", help="Start the persistent inference API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings.load(args.config, feature_set=getattr(args, "feature_set", None),
                             classifier_path=getattr(args, "classifier", None))
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
        load_manifest(args.manifest.resolve(), require_ready=True)
        dump(extract_manifest(args.manifest.resolve(), args.output, FeatureExtractor(settings)))
    elif args.command == "train":
        from .classifier import train_classifier
        artifact = train_classifier(args.features, args.output_dir,
                                    settings.merge_gap_seconds, settings.min_positive_windows, args.event_iou)
        dump({"classifier": str((args.output_dir / "classifier.json").resolve()),
              "threshold": artifact["threshold"], "reports": artifact["reports"]})
    elif args.command == "infer":
        from .detector import BoltDetector, UntrainedDetectorError
        from .reporting import write_json
        request_id = uuid4().hex
        log_path = args.log_output or settings.project_dir / "logs/inference" / f"{request_id}.json"
        try:
            result = BoltDetector(settings).detect(args.audio.resolve(strict=True), log_path=log_path,
                                                   request_id=request_id)
        except UntrainedDetectorError as error:
            print(f"{error}\nDetailed log: {log_path}", file=sys.stderr)
            raise SystemExit(2) from error
        write_json(args.output, result)
        if args.labels_output:
            from .annotations import export_labels
            export_labels(result["events"], args.labels_output, args.label_format)
        dump(result)
        print(f"Detailed log: {log_path.resolve()}", file=sys.stderr)
    elif args.command == "diagnose":
        from .detector import BoltDetector
        from .reporting import write_json
        result = BoltDetector(settings).diagnose(args.audio.resolve(strict=True))
        write_json(args.output, result)
        dump(result)
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
