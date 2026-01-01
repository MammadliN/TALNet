"""Run WSSED experiments for AnuraSet and FNJV.

This script mirrors the TALNet training flow but swaps the AudioSet-specific
parts with PAM-focused data preparation. It prepares stratified recording-level
splits on AnuraSet, generates bag/instance segments, and writes manifest files
that can be consumed by downstream training/evaluation jobs. FNJV is loaded
purely for final testing and never participates in training or validation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .constants import TARGET_SPECIES
from .datasets import (
    ClipAnnotation,
    SplitAssignments,
    build_anuraset_splits,
    build_recording_stats,
    describe_split,
    df_to_annotations,
    filter_split,
    generate_segments,
    load_anuraset,
    load_fnjv,
)

EXPERIMENT_GRID: List[Tuple[int, int]] = []
for bag in [60, 120, 300, 600]:
    for instance in [1, 3, 5, 10, 15, 30, 60]:
        EXPERIMENT_GRID.append((instance, bag))


def _segments_to_df(segments: Iterable[ClipAnnotation], instance_len: int, bag_len: int) -> pd.DataFrame:
    generated = generate_segments(segments, instance_length=instance_len, bag_length=bag_len)
    return pd.DataFrame(
        [
            {
                "file_name": seg.file_name,
                "file_path": str(seg.file_path),
                "start_second": seg.start_second,
                "end_second": seg.end_second,
                "label": seg.label,
                "bag_id": seg.bag_id,
            }
            for seg in generated
        ]
    )


def _save_split(df: pd.DataFrame, out_dir: pathlib.Path, name: str) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{name}.csv"
    df.to_csv(outfile, index=False)
    return describe_split(df)


def _write_manifest(
    out_dir: pathlib.Path,
    split_stats: Dict[str, Dict[str, int]],
    recording_stats: pd.DataFrame,
    config: Tuple[int, int],
    val_present: bool,
) -> None:
    manifest = {
        "target_species": TARGET_SPECIES,
        "instance_length_seconds": config[0],
        "bag_length_seconds": config[1],
        "splits": split_stats,
        "validation_enabled": val_present,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    recording_stats.to_csv(out_dir / "recording_label_counts.csv")


def _prepare_anuraset(
    annotations_root: pathlib.Path,
    output_root: pathlib.Path,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[SplitAssignments, pd.DataFrame]:
    anura_df = load_anuraset(annotations_root)
    splits = build_anuraset_splits(anura_df, val_size=val_size, test_size=test_size, random_state=random_state)
    recording_stats = build_recording_stats(anura_df)
    output_root.mkdir(parents=True, exist_ok=True)
    anura_df.to_csv(output_root / "anuraset_normalized.csv", index=False)
    recording_stats.to_csv(output_root / "anuraset_recording_label_counts.csv")
    return splits, anura_df


def _prepare_fnjv(metadata_path: pathlib.Path, output_root: pathlib.Path) -> pd.DataFrame:
    fnjv_df = load_fnjv(metadata_path)
    output_root.mkdir(parents=True, exist_ok=True)
    fnjv_df.to_csv(output_root / "fnjv_filtered.csv", index=False)
    return fnjv_df


def main():
    parser = argparse.ArgumentParser(description="Prepare AnuraSet/FNJV WSSED experiments")
    parser.add_argument("--anuraset-root", type=pathlib.Path, required=True, help="Directory with AnuraSet CSVs")
    parser.add_argument("--fnjv-metadata", type=pathlib.Path, required=True, help="Path to metadata_filtered_filled.csv")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("../../workspace/pam"))
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of recordings for validation (0 to disable)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of recordings for AnuraSet test")
    parser.add_argument("--random-seed", type=int, default=13)
    args = parser.parse_args()

    splits, anura_df = _prepare_anuraset(args.anuraset_root, args.output_dir, args.val_size, args.test_size, args.random_seed)
    fnjv_df = _prepare_fnjv(args.fnjv_metadata, args.output_dir)

    print("AnuraSet split sizes (recordings):")
    print({
        "train": len(splits.train),
        "val": len(splits.val),
        "test": len(splits.test),
    })
    print("FNJV usable recordings:", fnjv_df["file_name"].nunique())

    for instance_len, bag_len in EXPERIMENT_GRID:
        cfg_name = f"instance{instance_len}_bag{bag_len}"
        cfg_dir = args.output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # Build split-specific annotations
        train_df = filter_split(anura_df, splits.train)
        val_df = filter_split(anura_df, splits.val) if splits.val else pd.DataFrame(columns=train_df.columns)
        test_df = filter_split(anura_df, splits.test)

        stats = {}
        stats["train"] = _save_split(_segments_to_df(df_to_annotations(train_df), instance_len, bag_len), cfg_dir, "train")
        if not val_df.empty:
            stats["val"] = _save_split(_segments_to_df(df_to_annotations(val_df), instance_len, bag_len), cfg_dir, "val")
        else:
            stats["val"] = {}
        stats["test_anuraset"] = _save_split(
            _segments_to_df(df_to_annotations(test_df), instance_len, bag_len), cfg_dir, "test_anuraset"
        )
        stats["test_fnjv"] = _save_split(
            _segments_to_df(
                [
                    ClipAnnotation(
                        file_name=row.file_name,
                        file_path=pathlib.Path(row.get("file_path", row.file_name)),
                        start_second=float(row.get("start_second", 0.0)),
                        end_second=float(row.get("end_second", bag_len)),
                        label=row.label,
                    )
                    for row in fnjv_df.itertuples()
                ],
                instance_len,
                bag_len,
            ),
            cfg_dir,
            "test_fnjv",
        )

        _write_manifest(cfg_dir, stats, build_recording_stats(anura_df), (instance_len, bag_len), bool(splits.val))
        print(
            f"Prepared {cfg_name} | train recordings per label {stats['train']} | "
            f"val {stats['val']} | test AnuraSet {stats['test_anuraset']} | FNJV {stats['test_fnjv']}"
        )


if __name__ == "__main__":
    main()
