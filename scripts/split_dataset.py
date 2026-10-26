"""将当前增强批次按约70%/15%/15%划分；只整理清单，不启动训练。"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")
RATIOS = (.7, .15, .15)
PREFIX = "bolt_aug_v2_eq"


def apportion(count: int) -> list[int]:
    scaled = [count * ratio for ratio in RATIOS]
    result = [math.floor(value) for value in scaled]
    order = sorted(range(3), key=lambda i: (scaled[i] - result[i], -i), reverse=True)
    for index in order[:count - sum(result)]:
        result[index] += 1
    return result


def create_splits(rows: list[dict], logs: list[dict], seed=20261026) -> tuple[list[dict], dict]:
    """按录音类别分层；同一混合正例及其配对背景负例放入同一集合。"""
    indexed = {row["recording_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("原始清单存在重复录音编号")
    by_pair = {}
    for log in logs:
        if "pair_id" in log["augmentation"]:
            by_pair.setdefault(log["augmentation"]["pair_id"], []).append(log["recording_id"])
    pairs = []
    paired_ids = set()
    for key in sorted(by_pair):
        ids = by_pair[key]
        if len(ids) != 2 or len(set(ids)) != 2 or not set(ids) <= indexed.keys():
            raise ValueError(f"混合配对记录不完整：{key}")
        if {bool(indexed[value]["events"]) for value in ids} != {False, True}:
            raise ValueError(f"每个配对必须包含一条正例和一条反例：{key}")
        if paired_ids & set(ids):
            raise ValueError("同一录音被分配到了多个混合配对")
        pairs.append(ids)
        paired_ids.update(ids)
    positive = [row["recording_id"] for row in rows if row["events"]]
    negative = [row["recording_id"] for row in rows if not row["events"]]
    targets = {True: apportion(len(positive)), False: apportion(len(negative))}
    if min(*targets[True], *targets[False]) < 1:
        raise ValueError("当前正反例数量不足，无法保证三个集合各有正反例")
    pair_counts = apportion(len(pairs))
    if any(pair_counts[i] > min(targets[True][i], targets[False][i]) for i in range(3)):
        raise ValueError("当前配对数量无法满足此分层比例")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    assigned = {}
    offset = 0
    for split, count in zip(SPLITS, pair_counts):
        for pair in pairs[offset:offset + count]:
            for recording_id in pair:
                assigned[recording_id] = split
        offset += count
    for label, candidates in ((True, positive), (False, negative)):
        remaining = sorted(set(candidates) - paired_ids)
        rng.shuffle(remaining)
        offset = 0
        for index, split in enumerate(SPLITS):
            count = targets[label][index] - pair_counts[index]
            for recording_id in remaining[offset:offset + count]:
                assigned[recording_id] = split
            offset += count
        if offset != len(remaining):
            raise ValueError("剩余录音的划分数量不一致")
    if set(assigned) != indexed.keys():
        raise ValueError("划分未覆盖全部录音")
    output = [{**row, "split": assigned[row["recording_id"]],
               "source_split": row.get("source_split", row["split"]),
               "source_notes": row.get("source_notes", row.get("notes", "")),
               "notes": "当前增强批次的全量流程测试。", "pipeline_test": True,
               "split_seed": seed} for row in rows]
    summary = {
        "status": "prepared", "purpose": "pipeline_test", "seed": seed,
        "ratios": dict(zip(SPLITS, RATIOS)), "recordings": len(output),
        "method": "stratified_recording_split_with_positive_negative_pairs_kept_together",
        "pair_count": len(pairs), "original_group_ids_preserved": True, "training_started": False,
        "splits": {split: {"total": sum(row["split"] == split for row in output),
                           "positive": sum(row["split"] == split and bool(row["events"]) for row in output),
                           "negative": sum(row["split"] == split and not row["events"] for row in output),
                           "recording_ids": [row["recording_id"] for row in output if row["split"] == split]}
                   for split in SPLITS},
    }
    return output, summary


def write_splits(rows: list[dict], summary: dict, source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def rebase(value: str) -> str:
        original = Path(value)
        original = (original if original.is_absolute() else source_dir / original).resolve(strict=True)
        try:
            return Path(os.path.relpath(original, output_dir)).as_posix()
        except ValueError:
            return str(original)

    rebased = [{**row, "audio": rebase(row["audio"]),
                "labels": rebase(row["labels"]) if row.get("labels") else None,
                "plots": {kind: rebase(value) for kind, value in row.get("plots", {}).items()}}
               for row in rows]
    for name, selected in [("all", rebased)] + [(split, [row for row in rebased if row["split"] == split])
                                               for split in SPLITS]:
        path = output_dir / f"{PREFIX}_{name}.jsonl"
        temporary = path.with_suffix(".jsonl.part")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        temporary.replace(path)
    path = output_dir / f"{PREFIX}_split.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    from bolt_audio.dataset import load_manifest
    from smoke_train import configure_console

    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "data/raw/bolt_aug_v2_eq/train.jsonl", help="完整原始清单")
    parser.add_argument("--output-dir", type=Path, default=project / "data/manifests", help="划分清单输出目录")
    parser.add_argument("--seed", type=int, default=20261026, help="固定随机种子")
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    load_manifest(source, require_ready=True)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    logs = [json.loads(line) for line in (source.parent / "augmentation_log.jsonl").read_text(encoding="utf-8").splitlines()]
    selected, summary = create_splits(rows, logs, args.seed)
    write_splits(selected, summary, source.parent, args.output_dir.resolve())
    print("数据集划分完成，训练尚未启动。")
    for split, counts in summary["splits"].items():
        name = {"train": "训练集", "val": "验证集", "test": "测试集"}[split]
        print(f"{name}：{counts['total']}条，正例{counts['positive']}条，反例{counts['negative']}条")
    print(f"清单目录：{args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
