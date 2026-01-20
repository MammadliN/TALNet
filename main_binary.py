"""Binary (frog/background) training entrypoint using FNJV for training and AnuraSet for eval."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

from Net import Net


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
    anuraset_root: Path
    fnjv_root: Path
    fnjv_metadata: Path
    pooling: str
    bag_seconds: int
    output_dir: Path
    sample_rate: int
    fps: float
    n_mels: int
    n_fft: int
    embedding_size: int


def _load_audio_segment(path: Path, sample_rate: int, start: float, duration: float) -> np.ndarray:
    start_frame = int(round(start * sample_rate))
    frame_count = int(round(duration * sample_rate))
    try:
        info = sf.info(path)
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
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"soundfile failed for {path} ({exc}); falling back to librosa.load",
            RuntimeWarning,
        )
        audio, _ = librosa.load(path, sr=sample_rate, offset=start, duration=duration, mono=True)
        if len(audio) < frame_count:
            audio = np.pad(audio, (0, frame_count - len(audio)), mode="constant")
        return audio


def _ensure_audio_path(root: Path, row: pd.Series) -> Path:
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
    return candidates[0].with_suffix(".wav")


class AnuraSetBinaryDataset(Dataset):
    def __init__(
        self,
        root: Path,
        bag_seconds: int,
        metadata: pd.DataFrame,
        label_columns: Sequence[str],
        subset: Optional[str],
        sample_rate: int,
        fps: float,
        n_mels: int,
        n_fft: int,
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
                frog = 0.0
                if not label_block.empty:
                    frog = float(label_block.values.astype(np.float32).max())
                background = 1.0 - frog
                entries.append(
                    {
                        "audio_path": audio_path,
                        "start": float(start_time),
                        "labels": np.array([background, frog], dtype=np.float32),
                    }
                )
        return entries

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[index]
        y = _load_audio_segment(
            item["audio_path"], sample_rate=self.sample_rate, start=item["start"], duration=self.bag_seconds
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
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        features = mel_db.T
        target_frames = int(math.ceil(self.bag_seconds * (self.sample_rate / self.hop_length)))
        features = librosa.util.fix_length(features, size=target_frames, axis=0)
        x = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(item["labels"], dtype=torch.float32)
        return x, labels


class FNJVBIN(Dataset):
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
        x = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
        return x, labels


@dataclass
class TalnetArgs:
    embedding_size: int = 1024
    n_conv_layers: int = 10
    n_pool_layers: int = 5
    kernel_size: Tuple[int, int] = (3, 3)
    batch_norm: bool = True
    dropout: float = 0.0
    pooling: str = "max"
    output_size: int = 2


def build_model(pooling: str, output_size: int, device: torch.device, embedding_size: int) -> Net:
    args = TalnetArgs(pooling=pooling, output_size=output_size, embedding_size=embedding_size)
    return Net(args).to(device)


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


def compute_classification_metrics(
    preds: torch.Tensor, targets: torch.Tensor, thresholds: np.ndarray | float
) -> Dict[str, float]:
    preds_np = preds.detach().cpu().numpy()
    targets_np = (targets.detach().cpu().numpy() >= 0.5).astype(np.int32)
    if isinstance(thresholds, np.ndarray):
        pred_bin = (preds_np >= thresholds).astype(np.int32)
    else:
        pred_bin = (preds_np >= thresholds).astype(np.int32)

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
        "er": er,
    }


def run_epoch(
    model: Net,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    epoch_loss = 0.0
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
        all_preds.append(outputs.detach().cpu())
        all_targets.append(batch_y.detach().cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_classification_metrics(preds, targets, thresholds=0.5)
    avg_loss = epoch_loss / max(1, len(all_preds))
    return avg_loss, metrics


def collect_outputs(
    model: Net,
    loader,
    device: torch.device,
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


def main(args: argparse.Namespace, run_config: RunConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config.output_dir.mkdir(parents=True, exist_ok=True)

    fnjv_meta = pd.read_csv(run_config.fnjv_metadata)
    fnjv_meta = fnjv_meta[fnjv_meta["Code"].notna()]
    fnjv_meta = fnjv_meta[fnjv_meta["Code"].str.upper() != "IGNORE"]
    fnjv_codes = sorted(set(fnjv_meta["Code"]))

    anuraset_meta = pd.read_csv(run_config.anuraset_root / "metadata.csv")
    label_columns = [c for c in anuraset_meta.columns if c not in RESERVED_COLUMNS]
    fnjv_codes = [c for c in fnjv_codes if c in label_columns]
    if not fnjv_codes:
        raise ValueError("No FNJV species codes found in AnuraSet metadata.")

    fnjv_meta = fnjv_meta[fnjv_meta["Code"].isin(fnjv_codes)]
    train_ds = FNJVBIN(
        run_config.fnjv_root,
        fnjv_meta,
        fnjv_codes,
        sample_rate=run_config.sample_rate,
        fps=run_config.fps,
        n_mels=run_config.n_mels,
        n_fft=run_config.n_fft,
    )

    anuraset_meta = anuraset_meta[anuraset_meta[fnjv_codes].sum(axis=1) > 0]
    val_ds = AnuraSetBinaryDataset(
        run_config.anuraset_root,
        run_config.bag_seconds,
        anuraset_meta,
        fnjv_codes,
        subset="train",
        sample_rate=run_config.sample_rate,
        fps=run_config.fps,
        n_mels=run_config.n_mels,
        n_fft=run_config.n_fft,
    )
    test_ds = AnuraSetBinaryDataset(
        run_config.anuraset_root,
        run_config.bag_seconds,
        anuraset_meta,
        fnjv_codes,
        subset="test",
        sample_rate=run_config.sample_rate,
        fps=run_config.fps,
        n_mels=run_config.n_mels,
        n_fft=run_config.n_fft,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        run_config.pooling,
        output_size=2,
        device=device,
        embedding_size=run_config.embedding_size,
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_micro = -1.0
    best_micro_path = run_config.output_dir / "best_model_micro.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, None, criterion, device
        )
        print(
            f"Epoch {epoch}: train loss={train_loss:.4f}, train micro-F1={train_metrics['micro_f1']:.4f}"
        )
        print(
            f"           val loss={val_loss:.4f}, val micro-F1={val_metrics['micro_f1']:.4f}"
        )
        if val_metrics["micro_f1"] > best_micro:
            best_micro = val_metrics["micro_f1"]
            torch.save(model.state_dict(), best_micro_path)

    print(f"Training finished. Best micro-F1: {best_micro:.4f}")

    model.load_state_dict(torch.load(best_micro_path, map_location=device))
    model.eval()
    val_global, val_frame, val_targets = collect_outputs(model, val_loader, device)
    test_global, test_frame, test_targets = collect_outputs(model, test_loader, device)

    val_frame_targets = val_targets.unsqueeze(1).repeat(1, val_frame.size(1), 1)
    test_frame_targets = test_targets.unsqueeze(1).repeat(1, test_frame.size(1), 1)

    global_threshold = find_best_global_threshold(val_global, val_targets)
    per_class_thresholds = find_best_per_class_thresholds(val_frame.reshape(-1, 2), val_frame_targets.reshape(-1, 2))

    tag_metrics = compute_classification_metrics(test_global, test_targets, thresholds=global_threshold)
    frame_metrics = compute_classification_metrics(
        test_frame.reshape(-1, 2), test_frame_targets.reshape(-1, 2), thresholds=per_class_thresholds
    )
    print(
        "[binary] tagging micro-P={mp:.4f} micro-R={mr:.4f} micro-F1={mf1:.4f} ER={er:.4f}".format(
            mp=tag_metrics["micro_precision"],
            mr=tag_metrics["micro_recall"],
            mf1=tag_metrics["micro_f1"],
            er=tag_metrics["er"],
        )
    )
    print(
        "[binary] localization micro-P={mp:.4f} micro-R={mr:.4f} micro-F1={mf1:.4f} ER={er:.4f}".format(
            mp=frame_metrics["micro_precision"],
            mr=frame_metrics["micro_recall"],
            mf1=frame_metrics["micro_f1"],
            er=frame_metrics["er"],
        )
    )


if __name__ == "__main__":
    SAMPLE_RATE = 22000
    FPS = 40.0
    N_MELS = 64
    N_FFT = 1100
    EMBEDDING_SIZE = 1024

    parser = argparse.ArgumentParser(description="Binary frog/background training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "ave", "lin", "exp", "att", "softmax", "autopool", "powerpool", "betaexp"],
    )
    parser.add_argument("--bag_seconds", type=int, default=10)
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--n_mels", type=int, default=N_MELS)
    parser.add_argument("--n_fft", type=int, default=N_FFT)
    parser.add_argument("--embedding_size", type=int, default=EMBEDDING_SIZE)
    parser.add_argument(
        "--anuraset_root",
        type=Path,
        default=Path(r"/ds-iml/Bioacoustics/AnuraSet/raw_data"),
    )
    parser.add_argument(
        "--fnjv_root",
        type=Path,
        default=Path(r"/ds-iml/Bioacoustics/FNJV/458"),
    )
    parser.add_argument(
        "--fnjv_metadata",
        type=Path,
        default=Path(r"/ds-iml/Bioacoustics/FNJV/458/metadata_filtered_filled.csv"),
    )
    args = parser.parse_args()

    output_dir = Path("outputs") / (
        f"binary_{args.pooling}_{args.bag_seconds}sec_{args.epochs}epoch_{args.batch_size}batch"
    )

    run_config = RunConfig(
        anuraset_root=args.anuraset_root,
        fnjv_root=args.fnjv_root,
        fnjv_metadata=args.fnjv_metadata,
        pooling=args.pooling,
        bag_seconds=args.bag_seconds,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        fps=args.fps,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        embedding_size=args.embedding_size,
    )
    main(args, run_config)
