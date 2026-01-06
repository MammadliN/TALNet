"""End-to-end training entrypoint for AnuraSet with TALNet.

This script keeps the TALNet architecture intact while adapting the
data pipeline to the AnuraSet layout. Configure the three variables
below (``ANURASET_ROOT``, ``POOLING_MODE``, ``BAG_SECONDS``) and run

    python main.py

to execute the full training and evaluation pipeline.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import librosa

from Net import Net


# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------

# Root directory containing metadata.csv and the four site folders
# (INCT20955, INCT4, INCT41, INCT17)
ANURASET_ROOT = Path("/workspace/AnuraSet")

# Pooling choice: one of ["max", "ave", "lin", "exp", "att"]
POOLING_MODE = "max"

# Non-overlapping bag length in seconds: choose from [3, 10, 15, 30, 60]
BAG_SECONDS = 10


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

RESERVED_COLUMNS = {
    "sample_name",
    "fname",
    "min_t",
    "max_t",
    "site",
    "date",
    "species_number",
    "subset",
}


def _discover_label_columns(metadata: pd.DataFrame) -> List[str]:
    """Return label column names in original order."""

    return [c for c in metadata.columns if c not in RESERVED_COLUMNS]


def _select_label_columns(
    metadata: pd.DataFrame, target_species: Sequence[str]
) -> List[str]:
    label_columns = _discover_label_columns(metadata)
    if not target_species:
        return label_columns

    filtered = [c for c in label_columns if c in target_species]
    if not filtered:
        raise ValueError(
            "None of the requested target species were found in metadata columns."
        )
    missing = [c for c in target_species if c not in filtered]
    if missing:
        print(
            "Warning: the following target species were not present in metadata and will"
            f" be ignored: {missing}"
        )
    return filtered


def _ensure_audio_path(root: Path, row: pd.Series) -> Path:
    """Locate the wav file for a metadata row."""

    candidates = [
        root / str(row["site"]) / f"{row['fname']}.wav",
        root / str(row["site"]) / str(row["fname"]),
        root / str(row["site"]) / str(row.get("sample_name", "")),
    ]
    for path in candidates:
        if path.suffix == "":
            path = path.with_suffix(".wav")
        if path.exists():
            return path
    # Fall back to first guess even if it does not exist so the caller gets
    # a clear error when attempting to load.
    return candidates[0].with_suffix(".wav")


class AnuraSetDataset(Dataset):
    def __init__(
        self,
        root: Path,
        bag_seconds: int,
        metadata: pd.DataFrame,
        label_columns: Sequence[str],
        subset: Optional[str] = None,
        sample_rate: int = 32000,
        fps: float = 40.0,
        n_mels: int = 64,
        n_fft: int = 1024,
        clip_duration: int = 60,
    ) -> None:
        super().__init__()
        self.root = root
        self.bag_seconds = bag_seconds
        self.sample_rate = sample_rate
        self.hop_length = int(round(sample_rate / fps))
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.clip_duration = clip_duration
        self.label_columns = list(label_columns)
        self.subset = subset

        if subset:
            self.metadata = metadata[metadata["subset"] == subset].copy()
        else:
            self.metadata = metadata.copy()

        self.items = self._build_index()

    def _build_index(self) -> List[Dict]:
        entries: List[Dict] = []
        grouped = self.metadata.groupby(["site", "fname"], sort=False)
        for (site, fname), group in grouped:
            if group.empty:
                continue
            total_duration = max(group["max_t"].max(), float(self.clip_duration))
            n_bags = int(total_duration // self.bag_seconds)
            audio_path = _ensure_audio_path(self.root, group.iloc[0])

            for idx in range(n_bags):
                start_time = idx * self.bag_seconds
                end_time = start_time + self.bag_seconds
                in_window = (group["min_t"] >= start_time) & (group["max_t"] <= end_time)
                label_block = group.loc[in_window, self.label_columns]
                if label_block.empty:
                    labels = np.zeros(len(self.label_columns), dtype=np.float32)
                else:
                    labels = label_block.values.astype(np.float32).max(axis=0)
                entries.append(
                    {
                        "audio_path": audio_path,
                        "start": float(start_time),
                        "labels": labels,
                        "subset": group["subset"].iloc[0],
                        "site": site,
                        "fname": fname,
                    }
                )
        return entries

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        y, _ = librosa.load(
            item["audio_path"],
            sr=self.sample_rate,
            offset=item["start"],
            duration=self.bag_seconds,
            mono=True,
        )

        expected_len = int(self.sample_rate * self.bag_seconds)
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)), mode="constant")

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalize per-clip
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        features = mel_db.T  # time x mel
        target_frames = int(math.ceil(self.bag_seconds * (self.sample_rate / self.hop_length)))
        features = librosa.util.fix_length(features, size=target_frames, axis=0)

        x = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return x, labels


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------


@dataclass
class TalnetArgs:
    embedding_size: int = 1024
    n_conv_layers: int = 10
    n_pool_layers: int = 5
    kernel_size: Tuple[int, int] = (3, 3)
    batch_norm: bool = True
    dropout: float = 0.0
    pooling: str = "max"
    output_size: int = 527


def build_model(pooling: str, output_size: int, device: torch.device) -> Net:
    args = TalnetArgs(pooling=pooling, output_size=output_size)
    model = Net(args).to(device)
    return model


def compute_micro_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred >= threshold).int()
    target_bin = (target >= 0.5).int()
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else float(2 * tp) / float(denom)


def run_epoch(
    model: Net,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    total_batches: Optional[int] = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    epoch_loss = 0.0
    batch_counter = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(batch_x)[0]
        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        loss = criterion(outputs, batch_y)

        if training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        batch_counter += 1
        all_preds.append(outputs.detach().cpu())
        all_targets.append(batch_y.detach().cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    f1 = compute_micro_f1(preds, targets)
    batches_used = total_batches if total_batches is not None else batch_counter
    avg_loss = epoch_loss / max(1, batches_used)
    return avg_loss, f1


def create_dataloaders(
    root: Path,
    bag_seconds: int,
    batch_size: int,
    target_species: Sequence[str],
    num_workers: int = 0,
    use_batch_generator: bool = False,
) -> Tuple[
    AnuraSetDataset,
    Tuple[Sequence, int, bool],
    Optional[Tuple[Sequence, int, bool]],
]:
    metadata = pd.read_csv(root / "metadata.csv")
    label_columns = _select_label_columns(metadata, target_species)

    train_ds = AnuraSetDataset(root, bag_seconds, metadata, label_columns, subset="train")
    test_subset = "test" if "test" in metadata["subset"].unique() else None

    def _build_loader(ds: AnuraSetDataset, shuffle: bool) -> Tuple[Sequence, int, bool]:
        if use_batch_generator:
            total_batches = int(math.ceil(len(ds) / batch_size))

            def generator():
                indices = np.arange(len(ds))
                if shuffle:
                    np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    batch_idx = indices[start : start + batch_size]
                    batch = [ds[i] for i in batch_idx]
                    xs, ys = zip(*batch)
                    yield torch.stack(xs), torch.stack(ys)

            return generator, total_batches, True
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader, len(loader), False

    train_loader = _build_loader(train_ds, shuffle=True)
    test_loader: Optional[Tuple[Sequence, int, bool]] = None
    if test_subset:
        test_ds = AnuraSetDataset(root, bag_seconds, metadata, label_columns, subset=test_subset)
        test_loader = _build_loader(test_ds, shuffle=False)

    return train_ds, train_loader, test_loader


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using AnuraSet root: {ANURASET_ROOT}")
    print(f"Pooling: {POOLING_MODE} | Bag seconds: {BAG_SECONDS}")
    print(f"Device: {device} | Batch generator: {args.use_batch_generator}")
    print(f"Target species: {args.target_species if args.target_species else 'ALL'}")

    train_ds, train_loader_info, test_loader_info = create_dataloaders(
        ANURASET_ROOT,
        BAG_SECONDS,
        batch_size=args.batch_size,
        target_species=args.target_species,
        num_workers=args.num_workers,
        use_batch_generator=args.use_batch_generator,
    )

    output_size = len(train_ds.label_columns)
    model = build_model(POOLING_MODE, output_size, device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loader, train_batches, train_is_gen = train_loader_info
        train_data = train_loader() if train_is_gen else train_loader

        train_loss, train_f1 = run_epoch(
            model, train_data, optimizer, criterion, device, total_batches=train_batches
        )
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, train micro-F1={train_f1:.4f}")

        if test_loader_info is not None:
            test_loader, test_batches, test_is_gen = test_loader_info
            test_data = test_loader() if test_is_gen else test_loader
            with torch.no_grad():
                val_loss, val_f1 = run_epoch(
                    model, test_data, None, criterion, device, total_batches=test_batches
                )
            print(f"           test loss={val_loss:.4f}, test micro-F1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), "best_talnet.pt")
        else:
            if train_f1 > best_f1:
                best_f1 = train_f1
                torch.save(model.state_dict(), "best_talnet.pt")

    print("Training finished. Best micro-F1: {:.4f}".format(best_f1))


if __name__ == "__main__":
    TARGET_SPECIES = [
        "DENMIN",
        "LEPLAT",
        "PHYCUV",
        "SPHSUR",
        "SCIPER",
        "BOABIS",
        "BOAFAB",
        "LEPPOD",
        "PHYALB",
    ]

    parser = argparse.ArgumentParser(description="Train TALNet on AnuraSet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--use_batch_generator",
        action="store_true",
        help="Use the lightweight Python batch generator instead of PyTorch DataLoader",
    )
    parser.add_argument(
        "--target_species",
        nargs="*",
        default=TARGET_SPECIES,
        help=(
            "Target species columns to include; provide none to use all available species"
        ),
    )
    main(parser.parse_args())

