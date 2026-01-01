"""Data loading and splitting utilities for AnuraSet and FNJV.

The functions in this module replace the previous AudioSet-centric logic with
AnuraSet/FNJV pipelines tailored for Passive Acoustic Monitoring (PAM).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .constants import OTHERS, TARGET_SPECIES, map_to_target


@dataclass
class ClipAnnotation:
    """Representation of a single annotated segment."""

    file_name: str
    file_path: pathlib.Path
    start_second: float
    end_second: float
    label: str

    @property
    def duration(self) -> float:
        return float(self.end_second - self.start_second)


@dataclass
class SplitAssignments:
    """Recording-level split information."""

    train: List[str]
    val: List[str]
    test: List[str]


def _load_csv(csv_path: pathlib.Path, target_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"file_name", "file_path", "start_second", "end_second", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {csv_path}")

    df["label"] = df["label"].apply(map_to_target)
    if target_only:
        df = df[df["label"].isin(TARGET_SPECIES)]
    return df


def load_anuraset(annotations_root: pathlib.Path) -> pd.DataFrame:
    """Load AnuraSet annotations combining strong labels.

    Parameters
    ----------
    annotations_root:
        Directory containing ``10_species.csv`` and ``all_labels_combined.csv``.

    Returns
    -------
    pd.DataFrame
        Normalized annotations with canonicalized labels and OTHERS mapping.
    """

    ten_species = annotations_root / "10_species.csv"
    all_labels = annotations_root / "all_labels_combined.csv"
    if not ten_species.exists() or not all_labels.exists():
        raise FileNotFoundError("Expected both 10_species.csv and all_labels_combined.csv in AnuraSet annotations")

    # Use the exhaustive list to ensure we keep OTHERS examples.
    df_all = _load_csv(all_labels, target_only=False)
    df_ten = _load_csv(ten_species, target_only=True)

    # Keep all rows from the exhaustive file but replace the label for target rows with the
    # target-only mapping to prevent accidental OTHERS assignment for rare target codes.
    df = df_all.copy()
    df.loc[df_ten.index, "label"] = df_ten["label"]
    return df


def build_recording_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-recording label counts for reporting."""

    grouped = df.groupby(["file_name", "label"]).size().reset_index(name="n_segments")
    return grouped.pivot_table(index="file_name", columns="label", values="n_segments", fill_value=0)


def _stratified_group_split(
    df: pd.DataFrame,
    group_column: str,
    stratify_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[List[str], List[str]]:
    """Perform a stratified split while keeping groups intact.

    We approximate stratification by using the dominant label per group as the
    stratification target. This favors keeping scarce classes distributed while
    guaranteeing that segments from a single recording do not leak across splits.
    """

    # Determine dominant label for each group
    label_counts = df.groupby([group_column, stratify_column]).size().reset_index(name="count")
    dominant = label_counts.sort_values([group_column, "count"], ascending=[True, False]).drop_duplicates(group_column)
    groups = dominant[group_column]
    stratify_labels = dominant[stratify_column]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dominant, stratify_labels, groups=groups))
    train_groups = dominant.iloc[train_idx][group_column].tolist()
    test_groups = dominant.iloc[test_idx][group_column].tolist()
    return train_groups, test_groups


def build_anuraset_splits(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 13,
) -> SplitAssignments:
    """Create train/validation/test splits grouped by recording.

    Parameters
    ----------
    df:
        Normalized AnuraSet annotations.
    val_size:
        Fraction of recordings to allocate to validation. Set to 0 to skip validation.
    test_size:
        Fraction of recordings to allocate to test.
    random_state:
        Seed for reproducibility.
    """

    remaining = df.copy()
    train_groups, test_groups = _stratified_group_split(
        remaining, group_column="file_name", stratify_column="label", test_size=test_size, random_state=random_state
    )

    if val_size > 0:
        df_train = remaining[remaining["file_name"].isin(train_groups)]
        train_groups, val_groups = _stratified_group_split(
            df_train, group_column="file_name", stratify_column="label", test_size=val_size, random_state=random_state
        )
    else:
        val_groups = []

    return SplitAssignments(train=train_groups, val=val_groups, test=test_groups)


def filter_split(df: pd.DataFrame, split: Sequence[str]) -> pd.DataFrame:
    """Return only rows whose recording is listed in the split."""

    return df[df["file_name"].isin(split)].copy()


def load_fnjv(metadata_csv: pathlib.Path) -> pd.DataFrame:
    """Load FNJV metadata and filter to target species.

    Records with ``Code == 'IGNORE'`` are dropped; any rows with codes not in
    the target list are excluded from evaluation.
    """

    df = pd.read_csv(metadata_csv)
    required_cols = {"Arquivo do registro", "Code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {metadata_csv}")

    df = df[df["Code"].str.upper() != "IGNORE"].copy()
    df["label"] = df["Code"].apply(map_to_target)
    df = df[df["label"].isin(TARGET_SPECIES)]
    df.rename(columns={"Arquivo do registro": "file_name"}, inplace=True)
    return df


@dataclass
class Segment:
    """A temporal segment extracted from a recording."""

    file_name: str
    file_path: pathlib.Path
    start_second: float
    end_second: float
    label: str
    bag_id: str


def generate_segments(
    annotations: Iterable[ClipAnnotation],
    instance_length: float,
    bag_length: float,
) -> List[Segment]:
    """Split recordings into bag/instance segments.

    Each bag is a contiguous region of ``bag_length`` seconds, padded or clipped
    based on the source annotation boundaries. Instances inside a bag are
    non-overlapping and aligned to bag start times.
    """

    segments: List[Segment] = []
    for ann in annotations:
        clip_duration = ann.duration
        bag_count = max(1, int(round(clip_duration / bag_length + 0.499)))
        for bag_idx in range(bag_count):
            bag_start = ann.start_second + bag_idx * bag_length
            bag_end = min(ann.end_second, bag_start + bag_length)
            bag_id = f"{ann.file_name}_bag{bag_idx:04d}_{int(bag_length)}s"

            instance_count = max(1, int(round((bag_end - bag_start) / instance_length + 0.499)))
            for inst_idx in range(instance_count):
                inst_start = bag_start + inst_idx * instance_length
                inst_end = min(bag_end, inst_start + instance_length)
                segments.append(
                    Segment(
                        file_name=ann.file_name,
                        file_path=ann.file_path,
                        start_second=inst_start,
                        end_second=inst_end,
                        label=ann.label,
                        bag_id=bag_id,
                    )
                )
    return segments


def df_to_annotations(df: pd.DataFrame) -> List[ClipAnnotation]:
    """Convert an annotation dataframe to ``ClipAnnotation`` list."""

    return [
        ClipAnnotation(
            file_name=row.file_name,
            file_path=pathlib.Path(row.file_path),
            start_second=float(row.start_second),
            end_second=float(row.end_second),
            label=row.label,
        )
        for row in df.itertuples()
    ]


def describe_split(df: pd.DataFrame) -> Dict[str, int]:
    """Return a dictionary with counts per label within a dataframe."""

    return df.groupby("label")["file_name"].nunique().to_dict()
