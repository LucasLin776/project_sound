from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    check = commands.add_parser("check-data", help="Check annotations, audio and group separation")
    check.add_argument("--manifest", required=True, type=Path)
    extract = commands.add_parser("extract", help="Extract training features from reviewed recordings")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    train = commands.add_parser("train", help="Train a classifier and select threshold on validation recordings")
    train.add_argument("--features", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    infer = commands.add_parser("infer", help="Detect with a trained classifier, or run feature diagnostics")
    infer.add_argument("--audio", required=True, type=Path)
    infer.add_argument("--classifier", type=Path)
    infer.add_argument("--feature-set", choices=["panns", "fusion", "acoustic"])
    infer.add_argument("--output", required=True, type=Path)
    serve = commands.add_parser("serve", help="Start the persistent inference API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings.load(args.config, feature_set=getattr(args, "feature_set", None),
                             classifier_path=getattr(args, "classifier", None))
    if args.command == "init-manifest":
        from .dataset import initialize_manifest
        dump(initialize_manifest(args.audio_dir, args.output))
    elif args.command == "check-data":
        from .dataset import load_manifest
        rows = load_manifest(args.manifest.resolve())
        dump({"recordings": len(rows), "reviewed": sum(row["reviewed"] for row in rows),
              "ready_for_extraction": all(row["reviewed"] and row["split"] != "unassigned" for row in rows),
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
                                    settings.merge_gap_seconds, settings.min_positive_windows)
        dump({"classifier": str((args.output_dir / "classifier.json").resolve()),
              "threshold": artifact["threshold"], "reports": artifact["reports"]})
    elif args.command == "infer":
        from .detector import BoltDetector
        result = BoltDetector(settings).detect(args.audio.resolve(strict=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        dump(result)
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
