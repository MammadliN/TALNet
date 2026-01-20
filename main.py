"""End-to-end training entrypoint for AnuraSet with TALNet.

This script keeps the TALNet architecture intact while adapting the
data pipeline to the AnuraSet layout. Configure the defaults inside the
``if __name__ == "__main__"`` block or override them via CLI flags and run
``python main.py`` to execute the full training and evaluation pipeline.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import librosa
import matplotlib.pyplot as plt
import soundfile as sf

from Net import Net


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


@dataclass
class RunConfig:
    root: Path
    fnjv_root: Path
    fnjv_metadata: Path
    pooling: str
    bag_seconds: int
    target_species: Sequence[str]
    dataset_name: str
    threshold_method: str
    output_dir: Path
    sample_rate: int
    fps: float
    n_mels: int
    n_fft: int
    embedding_size: int


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


def _load_audio_segment(
    path: Path, sample_rate: int, start: float, duration: float
) -> np.ndarray:
    """Load a mono audio segment, preferring soundfile to avoid audioread fallbacks."""

    start_frame = int(round(start * sample_rate))
    frame_count = int(round(duration * sample_rate))

    try:
        info = sf.info(path)
        native_sr = info.samplerate
        total_frames = info.frames

        if start_frame >= total_frames:
            return np.zeros(frame_count, dtype=np.float32)

        frames_to_read = min(frame_count, total_frames - start_frame)
        audio, sr = sf.read(
            path,
            start=start_frame,
            frames=frames_to_read,
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        if len(audio) < frame_count:
            audio = np.pad(audio, (0, frame_count - len(audio)), mode="constant")
        return audio
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(
            f"soundfile failed for {path} ({exc}); falling back to librosa.load",
            RuntimeWarning,
        )
        audio, _ = librosa.load(
            path, sr=sample_rate, offset=start, duration=duration, mono=True
        )
        if len(audio) < frame_count:
            audio = np.pad(audio, (0, frame_count - len(audio)), mode="constant")
        return audio


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
        file_filter: Optional[Set[Tuple[str, str]]] = None,
        sample_rate: int = 22000,
        fps: float = 40.0,
        n_mels: int = 64,
        n_fft: int = 1100,
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
        if file_filter:
            file_df = pd.DataFrame(list(file_filter), columns=["site", "fname"])
            self.metadata = self.metadata.merge(file_df, on=["site", "fname"], how="inner")

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
        y = _load_audio_segment(
            item["audio_path"],
            sample_rate=self.sample_rate,
            start=item["start"],
            duration=self.bag_seconds,
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


class FNJVDataset(Dataset):
    def __init__(
        self,
        root: Path,
        metadata: pd.DataFrame,
        label_columns: Sequence[str],
        sample_rate: int,
        fps: float,
        n_mels: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        self.root = root
        self.sample_rate = sample_rate
        self.hop_length = int(round(sample_rate / fps))
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.label_columns = list(label_columns)
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        audio_path = self.root / row["Arquivo do registro"]
        y = _load_audio_segment(audio_path, self.sample_rate, start=0.0, duration=60.0)

        expected_len = int(self.sample_rate * 60.0)
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
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        features = mel_db.T
        target_frames = int(math.ceil(60.0 * (self.sample_rate / self.hop_length)))
        features = librosa.util.fix_length(features, size=target_frames, axis=0)

        labels = np.zeros(len(self.label_columns), dtype=np.float32)
        code = row["Code"]
        if code in self.label_columns:
            labels[self.label_columns.index(code)] = 1.0

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y


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


def build_model(pooling: str, output_size: int, device: torch.device, embedding_size: int) -> Net:
    args = TalnetArgs(pooling=pooling, output_size=output_size, embedding_size=embedding_size)
    model = Net(args).to(device)
    return model


def _binarize_predictions(
    preds: np.ndarray, thresholds: np.ndarray | float
) -> np.ndarray:
    if isinstance(thresholds, np.ndarray):
        return (preds >= thresholds).astype(np.int32)
    return (preds >= thresholds).astype(np.int32)


def compute_classification_metrics(
    preds: torch.Tensor, targets: torch.Tensor, thresholds: np.ndarray | float
) -> Dict[str, float]:
    preds_np = preds.detach().cpu().numpy()
    targets_np = (targets.detach().cpu().numpy() >= 0.5).astype(np.int32)
    pred_bin = _binarize_predictions(preds_np, thresholds)

    tp = (pred_bin * targets_np).sum()
    fp = (pred_bin * (1 - targets_np)).sum()
    fn = ((1 - pred_bin) * targets_np).sum()
    micro_precision = 0.0 if (tp + fp) == 0 else float(tp) / float(tp + fp)
    micro_recall = 0.0 if (tp + fn) == 0 else float(tp) / float(tp + fn)
    micro_f1 = (
        0.0
        if (2 * tp + fp + fn) == 0
        else float(2 * tp) / float(2 * tp + fp + fn)
    )

    macro_precision = []
    macro_recall = []
    macro_f1 = []
    for idx in range(preds_np.shape[1]):
        pred_c = pred_bin[:, idx]
        targ_c = targets_np[:, idx]
        tp_c = (pred_c * targ_c).sum()
        fp_c = (pred_c * (1 - targ_c)).sum()
        fn_c = ((1 - pred_c) * targ_c).sum()
        prec_c = 0.0 if (tp_c + fp_c) == 0 else float(tp_c) / float(tp_c + fp_c)
        rec_c = 0.0 if (tp_c + fn_c) == 0 else float(tp_c) / float(tp_c + fn_c)
        f1_c = (
            0.0
            if (2 * tp_c + fp_c + fn_c) == 0
            else float(2 * tp_c) / float(2 * tp_c + fp_c + fn_c)
        )
        macro_precision.append(prec_c)
        macro_recall.append(rec_c)
        macro_f1.append(f1_c)

    macro_precision_val = float(np.mean(macro_precision)) if macro_precision else 0.0
    macro_recall_val = float(np.mean(macro_recall)) if macro_recall else 0.0
    macro_f1_val = float(np.mean(macro_f1)) if macro_f1 else 0.0

    ntrue = targets_np.sum(axis=1)
    npred = pred_bin.sum(axis=1)
    ncorr = (pred_bin & targets_np).sum(axis=1)
    nmiss = ntrue - ncorr
    nfa = npred - ncorr
    denom = ntrue.sum()
    er = 0.0 if denom == 0 else float(np.maximum(nmiss, nfa).sum()) / float(denom)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision_val,
        "macro_recall": macro_recall_val,
        "macro_f1": macro_f1_val,
        "er": er,
    }


def _best_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.size == 0:
        return 0.5
    best_thres = float("inf")
    best_f1 = 0.0
    instances = [(-np.inf, False)] + sorted(zip(scores.tolist(), targets.tolist()))
    tp = 0
    denom = targets.sum()
    for i in range(len(instances) - 1, 0, -1):
        if instances[i][1]:
            tp += 1
        denom += 1
        if instances[i][0] > instances[i - 1][0]:
            f1 = 0.0 if denom == 0 else (2.0 * tp) / denom
            if f1 > best_f1:
                best_thres = (instances[i][0] + instances[i - 1][0]) / 2.0
                best_f1 = f1
    return best_thres if np.isfinite(best_thres) else 0.5


def find_best_global_threshold(preds: torch.Tensor, targets: torch.Tensor) -> float:
    scores = preds.detach().cpu().numpy().reshape(-1)
    truth = targets.detach().cpu().numpy().reshape(-1).astype(bool)
    return _best_threshold(scores, truth)


def find_best_per_class_thresholds(preds: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    scores = preds.detach().cpu().numpy()
    truth = targets.detach().cpu().numpy().astype(bool)
    thresholds = np.zeros(scores.shape[1], dtype=np.float32)
    for idx in range(scores.shape[1]):
        thresholds[idx] = _best_threshold(scores[:, idx], truth[:, idx])
    return thresholds


def run_epoch(
    model: Net,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    total_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
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
    metrics = compute_classification_metrics(preds, targets, thresholds=0.5)
    batches_used = total_batches if total_batches is not None else batch_counter
    avg_loss = epoch_loss / max(1, batches_used)
    return avg_loss, metrics


def collect_outputs(
    model: Net,
    loader,
    device: torch.device,
    total_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_globals: List[torch.Tensor] = []
    all_frames: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        global_prob = torch.clamp(outputs[0], 1e-7, 1 - 1e-7)
        frame_prob = torch.clamp(outputs[1], 1e-7, 1 - 1e-7)
        all_globals.append(global_prob.detach().cpu())
        all_frames.append(frame_prob.detach().cpu())
        all_targets.append(batch_y.detach().cpu())
    return (
        torch.cat(all_globals, dim=0),
        torch.cat(all_frames, dim=0),
        torch.cat(all_targets, dim=0),
    )


def build_stratified_split(
    metadata: pd.DataFrame,
    label_columns: Sequence[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    train_meta = metadata[metadata["subset"] == "train"].copy()
    file_labels = (
        train_meta.groupby(["site", "fname"], sort=False)[list(label_columns)]
        .max()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    val_files: Set[Tuple[str, str]] = set()
    for col in label_columns:
        positives = file_labels[file_labels[col] > 0]
        if positives.empty:
            continue
        candidates = list(
            zip(positives["site"].tolist(), positives["fname"].tolist())
        )
        rng.shuffle(candidates)
        target_count = max(1, int(round(val_ratio * len(candidates))))
        selected = [c for c in candidates if c not in val_files][:target_count]
        val_files.update(selected)
    all_files = set(zip(file_labels["site"], file_labels["fname"]))
    train_files = all_files - val_files
    return train_files, val_files


def create_dataloaders(
    root: Path,
    fnjv_root: Path,
    fnjv_metadata: Path,
    bag_seconds: int,
    batch_size: int,
    target_species: Sequence[str],
    dataset_name: str,
    sample_rate: int,
    fps: float,
    n_mels: int,
    n_fft: int,
    num_workers: int = 0,
    use_batch_generator: bool = False,
) -> Tuple[
    AnuraSetDataset,
    AnuraSetDataset,
    Tuple[Sequence, int, bool],
    Tuple[Sequence, int, bool],
    Optional[Tuple[Sequence, int, bool]],
]:
    anuraset_meta = pd.read_csv(root / "metadata.csv")
    label_columns = _select_label_columns(anuraset_meta, target_species)
    if dataset_name == "FNJV":
        fnjv_meta = pd.read_csv(fnjv_metadata)
        fnjv_meta = fnjv_meta[fnjv_meta["Code"].notna()]
        fnjv_meta = fnjv_meta[fnjv_meta["Code"].str.upper() != "IGNORE"]
        fnjv_codes = sorted({code for code in fnjv_meta["Code"] if code in label_columns})
        if not fnjv_codes:
            raise ValueError("FNJV metadata has no matching species codes with AnuraSet.")
        label_columns = fnjv_codes
        fnjv_meta = fnjv_meta[fnjv_meta["Code"].isin(label_columns)]
        train_ds = FNJVDataset(
            fnjv_root,
            fnjv_meta,
            label_columns,
            sample_rate=sample_rate,
            fps=fps,
            n_mels=n_mels,
            n_fft=n_fft,
        )
        val_ds = AnuraSetDataset(
            root,
            bag_seconds,
            anuraset_meta,
            label_columns,
            subset="train",
            sample_rate=sample_rate,
            fps=fps,
            n_mels=n_mels,
            n_fft=n_fft,
        )
        test_subset = "test" if "test" in anuraset_meta["subset"].unique() else None
        test_ds = (
            AnuraSetDataset(
                root,
                bag_seconds,
                anuraset_meta,
                label_columns,
                subset=test_subset,
                sample_rate=sample_rate,
                fps=fps,
                n_mels=n_mels,
                n_fft=n_fft,
            )
            if test_subset
            else None
        )
    else:
        train_files, val_files = build_stratified_split(anuraset_meta, label_columns)
        train_ds = AnuraSetDataset(
            root,
            bag_seconds,
            anuraset_meta,
            label_columns,
            subset="train",
            file_filter=train_files,
            sample_rate=sample_rate,
            fps=fps,
            n_mels=n_mels,
            n_fft=n_fft,
        )
        val_ds = AnuraSetDataset(
            root,
            bag_seconds,
            anuraset_meta,
            label_columns,
            subset="train",
            file_filter=val_files,
            sample_rate=sample_rate,
            fps=fps,
            n_mels=n_mels,
            n_fft=n_fft,
        )
        test_subset = "test" if "test" in anuraset_meta["subset"].unique() else None
        test_ds = (
            AnuraSetDataset(
                root,
                bag_seconds,
                anuraset_meta,
                label_columns,
                subset=test_subset,
                sample_rate=sample_rate,
                fps=fps,
                n_mels=n_mels,
                n_fft=n_fft,
            )
            if test_subset
            else None
        )

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
    val_loader = _build_loader(val_ds, shuffle=False)
    test_loader: Optional[Tuple[Sequence, int, bool]] = None
    if test_ds is not None:
        test_loader = _build_loader(test_ds, shuffle=False)

    return train_ds, val_ds, train_loader, val_loader, test_loader


def main(args: argparse.Namespace, run_config: RunConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using AnuraSet root: {run_config.root}")
    if run_config.dataset_name == "FNJV":
        print(f"Using FNJV root: {run_config.fnjv_root}")
    print(f"Pooling: {run_config.pooling} | Bag seconds: {run_config.bag_seconds}")
    print(f"Device: {device} | Batch generator: {args.use_batch_generator}")
    print(
        f"Target species: {run_config.target_species if run_config.target_species else 'ALL'}"
    )
    print(f"Dataset selection: {run_config.dataset_name}")
    print(f"Threshold method: {run_config.threshold_method}")

    train_ds, val_ds, train_loader_info, val_loader_info, test_loader_info = create_dataloaders(
        run_config.root,
        run_config.fnjv_root,
        run_config.fnjv_metadata,
        run_config.bag_seconds,
        batch_size=args.batch_size,
        target_species=run_config.target_species,
        dataset_name=run_config.dataset_name,
        sample_rate=run_config.sample_rate,
        fps=run_config.fps,
        n_mels=run_config.n_mels,
        n_fft=run_config.n_fft,
        num_workers=args.num_workers,
        use_batch_generator=args.use_batch_generator,
    )

    output_size = len(train_ds.label_columns)
    model = build_model(run_config.pooling, output_size, device, run_config.embedding_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_f1 = -1.0
    best_macro = -1.0
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    best_micro_path = run_config.output_dir / "best_model_micro.pt"
    best_macro_path = run_config.output_dir / "best_model_macro.pt"
    plot_path = run_config.output_dir / "training_metrics.png"
    history: Dict[str, Dict[str, List[float]]] = {
        "train": {
            "loss": [],
            "micro_f1": [],
            "macro_f1": [],
            "micro_precision": [],
            "micro_recall": [],
            "macro_precision": [],
            "macro_recall": [],
            "er": [],
        },
        "val": {
            "loss": [],
            "micro_f1": [],
            "macro_f1": [],
            "micro_precision": [],
            "micro_recall": [],
            "macro_precision": [],
            "macro_recall": [],
            "er": [],
        },
    }

    def _append_history(split: str, loss: float, metrics: Dict[str, float]) -> None:
        history[split]["loss"].append(loss)
        history[split]["micro_f1"].append(metrics["micro_f1"])
        history[split]["macro_f1"].append(metrics["macro_f1"])
        history[split]["micro_precision"].append(metrics["micro_precision"])
        history[split]["micro_recall"].append(metrics["micro_recall"])
        history[split]["macro_precision"].append(metrics["macro_precision"])
        history[split]["macro_recall"].append(metrics["macro_recall"])
        history[split]["er"].append(metrics["er"])

    def _plot_history(output_path: Path) -> None:
        epochs = np.arange(1, len(history["train"]["loss"]) + 1)
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes = axes.flatten()

        def plot_metric(
            ax, metric: str, title: str, ylim: Optional[Tuple[float, float]] = None
        ) -> None:
            ax.plot(epochs, history["train"][metric], label="train")
            ax.plot(epochs, history["val"][metric], label="val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " "))
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plot_metric(axes[0], "loss", "Loss")
        plot_metric(axes[1], "micro_f1", "Micro F1", ylim=(0.0, 1.0))
        plot_metric(axes[2], "macro_f1", "Macro F1", ylim=(0.0, 1.0))
        plot_metric(axes[3], "micro_precision", "Micro Precision", ylim=(0.0, 1.0))
        plot_metric(axes[4], "micro_recall", "Micro Recall", ylim=(0.0, 1.0))
        plot_metric(axes[5], "macro_precision", "Macro Precision", ylim=(0.0, 1.0))
        plot_metric(axes[6], "macro_recall", "Macro Recall", ylim=(0.0, 1.0))
        plot_metric(axes[7], "er", "Error Rate")

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    for epoch in range(1, args.epochs + 1):
        train_loader, train_batches, train_is_gen = train_loader_info
        train_data = train_loader() if train_is_gen else train_loader

        train_loss, train_metrics = run_epoch(
            model, train_data, optimizer, criterion, device, total_batches=train_batches
        )
        _append_history("train", train_loss, train_metrics)
        print(
            "Epoch {epoch}: train loss={loss:.4f}, micro-P={mp:.4f}, micro-R={mr:.4f}, "
            "micro-F1={mf1:.4f}, macro-P={Mp:.4f}, macro-R={Mr:.4f}, macro-F1={Mf1:.4f}, "
            "ER={er:.4f}".format(
                epoch=epoch,
                loss=train_loss,
                mp=train_metrics["micro_precision"],
                mr=train_metrics["micro_recall"],
                mf1=train_metrics["micro_f1"],
                Mp=train_metrics["macro_precision"],
                Mr=train_metrics["macro_recall"],
                Mf1=train_metrics["macro_f1"],
                er=train_metrics["er"],
            )
        )

        val_loader, val_batches, val_is_gen = val_loader_info
        val_data = val_loader() if val_is_gen else val_loader
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model, val_data, None, criterion, device, total_batches=val_batches
            )
        _append_history("val", val_loss, val_metrics)
        print(
            "           val loss={loss:.4f}, micro-P={mp:.4f}, micro-R={mr:.4f}, "
            "micro-F1={mf1:.4f}, macro-P={Mp:.4f}, macro-R={Mr:.4f}, macro-F1={Mf1:.4f}, "
            "ER={er:.4f}".format(
                loss=val_loss,
                mp=val_metrics["micro_precision"],
                mr=val_metrics["micro_recall"],
                mf1=val_metrics["micro_f1"],
                Mp=val_metrics["macro_precision"],
                Mr=val_metrics["macro_recall"],
                Mf1=val_metrics["macro_f1"],
                er=val_metrics["er"],
            )
        )
        if val_metrics["micro_f1"] > best_f1:
            best_f1 = val_metrics["micro_f1"]
            torch.save(model.state_dict(), best_micro_path)
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_macro_path)
        _plot_history(plot_path)

    print("Training finished. Best micro-F1: {:.4f}".format(best_f1))
    print("Training finished. Best macro-F1: {:.4f}".format(best_macro))

    if test_loader_info is None:
        print("No test split available; skipping localization evaluation.")
        return

    val_loader, _, val_is_gen = val_loader_info
    val_data = val_loader() if val_is_gen else val_loader
    test_loader, _, test_is_gen = test_loader_info
    test_data = test_loader() if test_is_gen else test_loader

    def evaluate_model(tag: str, model_path: Path) -> None:
        if not model_path.exists():
            print(f"[{tag}] model file not found: {model_path}")
            return
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        val_global, val_frame, val_targets = collect_outputs(model, val_data, device)
        test_global, test_frame, test_targets = collect_outputs(model, test_data, device)

        val_frame_targets = val_targets.unsqueeze(1).repeat(1, val_frame.size(1), 1)
        test_frame_targets = test_targets.unsqueeze(1).repeat(1, test_frame.size(1), 1)

        global_threshold = find_best_global_threshold(val_global, val_targets)
        per_class_tagging_thresholds = find_best_per_class_thresholds(val_global, val_targets)
        per_class_segment_thresholds = find_best_per_class_thresholds(
            val_frame.reshape(-1, val_frame.size(-1)),
            val_frame_targets.reshape(-1, val_frame_targets.size(-1)),
        )

        methods = ["global", "per_class", "segment"]
        if run_config.threshold_method != "all":
            methods = [run_config.threshold_method]

        print(f"[{tag}] Evaluation results:")
        for method in methods:
            if method == "global":
                tag_metrics = compute_classification_metrics(
                    test_global, test_targets, thresholds=global_threshold
                )
                frame_metrics = compute_classification_metrics(
                    test_frame.reshape(-1, test_frame.size(-1)),
                    test_frame_targets.reshape(-1, test_frame_targets.size(-1)),
                    thresholds=global_threshold,
                )
                print(
                    "[global threshold] tagging micro-P={mp:.4f} micro-R={mr:.4f} micro-F1={mf1:.4f} "
                    "macro-P={Mp:.4f} macro-R={Mr:.4f} macro-F1={Mf1:.4f} ER={er:.4f}".format(
                        mp=tag_metrics["micro_precision"],
                        mr=tag_metrics["micro_recall"],
                        mf1=tag_metrics["micro_f1"],
                        Mp=tag_metrics["macro_precision"],
                        Mr=tag_metrics["macro_recall"],
                        Mf1=tag_metrics["macro_f1"],
                        er=tag_metrics["er"],
                    )
                )
                print(
                    "[global threshold] localization micro-P={mp:.4f} micro-R={mr:.4f} micro-F1={mf1:.4f} "
                    "macro-P={Mp:.4f} macro-R={Mr:.4f} macro-F1={Mf1:.4f} ER={er:.4f}".format(
                        mp=frame_metrics["micro_precision"],
                        mr=frame_metrics["micro_recall"],
                        mf1=frame_metrics["micro_f1"],
                        Mp=frame_metrics["macro_precision"],
                        Mr=frame_metrics["macro_recall"],
                        Mf1=frame_metrics["macro_f1"],
                        er=frame_metrics["er"],
                    )
                )
            else:
                tag_metrics = compute_classification_metrics(
                    test_global, test_targets, thresholds=per_class_tagging_thresholds
                )
                if method == "per_class":
                    frame_metrics = compute_classification_metrics(
                        test_frame.reshape(-1, test_frame.size(-1)),
                        test_frame_targets.reshape(-1, test_frame_targets.size(-1)),
                        thresholds=per_class_tagging_thresholds,
                    )
                    label = "per-class threshold"
                else:
                    frame_metrics = compute_classification_metrics(
                        test_frame.reshape(-1, test_frame.size(-1)),
                        test_frame_targets.reshape(-1, test_frame_targets.size(-1)),
                        thresholds=per_class_segment_thresholds,
                    )
                    label = "segment-level threshold"
                print(
                    "[{label}] tagging micro-P={mp:.4f} micro-R={mr:.4f} micro-F1={mf1:.4f} "
                    "macro-P={Mp:.4f} macro-R={Mr:.4f} macro-F1={Mf1:.4f} ER={er:.4f}".format(
                        label=label,
                        mp=tag_metrics["micro_precision"],
                        mr=tag_metrics["micro_recall"],
                        mf1=tag_metrics["micro_f1"],
                        Mp=tag_metrics["macro_precision"],
                        Mr=tag_metrics["macro_recall"],
                        Mf1=tag_metrics["macro_f1"],
                        er=tag_metrics["er"],
                    )
                )
                print(
                    "[{label}] localization micro-P={mp:.4f} micro-R={mr:.4f} micro-F1={mf1:.4f} "
                    "macro-P={Mp:.4f} macro-R={Mr:.4f} macro-F1={Mf1:.4f} ER={er:.4f}".format(
                        label=label,
                        mp=frame_metrics["micro_precision"],
                        mr=frame_metrics["micro_recall"],
                        mf1=frame_metrics["micro_f1"],
                        Mp=frame_metrics["macro_precision"],
                        Mr=frame_metrics["macro_recall"],
                        Mf1=frame_metrics["macro_f1"],
                        er=frame_metrics["er"],
                    )
                )

    evaluate_model("best-micro", best_micro_path)
    evaluate_model("best-macro", best_macro_path)


if __name__ == "__main__":
    ANURASET_ROOT = Path(r"/ds-iml/Bioacoustics/AnuraSet/raw_data")
    FNJV_ROOT = Path(r"/ds-iml/Bioacoustics/FNJV/458")
    FNJV_METADATA = FNJV_ROOT / "metadata_filtered_filled.csv"
    DEFAULT_POOLING = "max"
    DEFAULT_BAG_SECONDS = 10
    DATASET_NAME = "AnuraSet"
    THRESHOLD_METHOD = "all"
    SAMPLE_RATE = 22000
    FPS = 40.0
    N_MELS = 64
    N_FFT = 1100
    EMBEDDING_SIZE = 1024

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument(
        "--all_species",
        action="store_true",
        help="Ignore target species list and train on all available labels",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override AnuraSet root containing metadata.csv and site folders",
    )
    parser.add_argument(
        "--fnjv_root",
        type=Path,
        default=None,
        help="Override FNJV root containing metadata.csv and site folders",
    )
    parser.add_argument(
        "--fnjv_metadata",
        type=Path,
        default=None,
        help="Override FNJV metadata CSV (metadata_filtered_filled.csv)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DATASET_NAME,
        choices=["AnuraSet", "FNJV"],
        help="Select which dataset to use for training",
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default=THRESHOLD_METHOD,
        choices=["global", "per_class", "segment", "all"],
        help="Threshold strategy for tagging/localization evaluation",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=None,
        choices=["max", "ave", "lin", "exp", "att", "softmax", "autopool", "powerpool", "betaexp"],
        help="Pooling mode to use for MIL reduction",
    )
    parser.add_argument(
        "--bag_seconds",
        type=int,
        default=None,
        choices=[3, 10, 15, 30, 60],
        help="Non-overlapping bag duration in seconds",
    )
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--n_mels", type=int, default=N_MELS)
    parser.add_argument("--n_fft", type=int, default=N_FFT)
    parser.add_argument("--embedding_size", type=int, default=EMBEDDING_SIZE)

    cli_args = parser.parse_args()
    resolved_root = cli_args.root or ANURASET_ROOT
    resolved_fnjv_root = cli_args.fnjv_root or FNJV_ROOT
    resolved_fnjv_metadata = cli_args.fnjv_metadata or FNJV_METADATA
    resolved_pooling = cli_args.pooling or DEFAULT_POOLING
    resolved_bag_seconds = cli_args.bag_seconds or DEFAULT_BAG_SECONDS
    resolved_species = [] if cli_args.all_species else cli_args.target_species

    output_dir = Path("outputs") / (
        f"{cli_args.dataset_name}_{resolved_pooling}_{resolved_bag_seconds}sec_"
        f"{cli_args.epochs}epoch_{cli_args.batch_size}batch"
    )

    run_config = RunConfig(
        root=resolved_root,
        fnjv_root=resolved_fnjv_root,
        fnjv_metadata=resolved_fnjv_metadata,
        pooling=resolved_pooling,
        bag_seconds=resolved_bag_seconds,
        target_species=resolved_species,
        dataset_name=cli_args.dataset_name,
        threshold_method=cli_args.threshold_method,
        output_dir=output_dir,
        sample_rate=cli_args.sample_rate,
        fps=cli_args.fps,
        n_mels=cli_args.n_mels,
        n_fft=cli_args.n_fft,
        embedding_size=cli_args.embedding_size,
    )

    main(cli_args, run_config)
