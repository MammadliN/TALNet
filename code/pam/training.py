"""Train TALNet on PAM WSSED manifests with AnuraSet/FNJV labels."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import wave
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .constants import PAM_CLASSES
from .talnet import PamTALNet


def _mel_filterbank(n_fft: int, n_mels: int, sample_rate: int, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or sample_rate / 2.0

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = fft_bins[m - 1], fft_bins[m], fft_bins[m + 1]
        if f_m_minus == f_m or f_m == f_m_plus:
            continue
        for k in range(f_m_minus, f_m):
            filterbank[m - 1, k] = (k - f_m_minus) / float(f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filterbank[m - 1, k] = (f_m_plus - k) / float(f_m_plus - f_m)
    return filterbank


def _dtype_from_sampwidth(width: int):
    if width == 1:
        return np.int8
    if width == 2:
        return np.int16
    if width == 3:
        # 24-bit PCM: load into 32-bit container
        return np.int32
    if width == 4:
        return np.int32
    raise ValueError(f"Unsupported sample width: {width}")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / float(orig_sr)
    target_length = int(round(duration * target_sr))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _load_wave(path: pathlib.Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        buffer = wf.readframes(n_frames)
    dtype = _dtype_from_sampwidth(sample_width)
    audio = np.frombuffer(buffer, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    max_val = float(np.iinfo(dtype).max)
    if max_val > 0:
        audio /= max_val
    return audio, sample_rate


def _slice_and_pad(audio: np.ndarray, sample_rate: int, start_second: float, duration: float) -> np.ndarray:
    start_idx = int(max(0, round(start_second * sample_rate)))
    end_idx = start_idx + int(round(duration * sample_rate))
    segment = audio[start_idx:end_idx]
    desired = end_idx - start_idx
    if len(segment) < desired:
        segment = np.pad(segment, (0, desired - len(segment)), mode="constant")
    elif len(segment) > desired:
        segment = segment[:desired]
    return segment.astype(np.float32)


def _logmel_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    n_fft = n_fft or 1 << (win_length - 1).bit_length()
    window = np.hanning(win_length).astype(np.float32)
    step = hop_length
    frames: List[np.ndarray] = []
    for start in range(0, max(len(audio) - win_length + 1, 1), step):
        frame = audio[start : start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)), mode="constant")
        spectrum = np.fft.rfft(frame * window, n=n_fft)
        frames.append(np.abs(spectrum) ** 2)
    power_spec = np.stack(frames, axis=1) if frames else np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
    mel_filters = _mel_filterbank(n_fft, n_mels, sample_rate)
    mel_spec = np.dot(mel_filters, power_spec)
    mel_spec = np.maximum(mel_spec, 1e-10)
    logmel = 10.0 * np.log10(mel_spec)
    return logmel.T.astype(np.float32)


@dataclass
class BagRecord:
    bag_id: str
    file_path: pathlib.Path
    start_second: float
    duration: float
    labels: List[str]


class PamBagDataset(Dataset):
    """Dataset that loads bag-level segments and computes log-mel features."""

    def __init__(
        self,
        manifest_csv: pathlib.Path,
        audio_root: Optional[pathlib.Path] = None,
        sample_rate: int = 32000,
        n_mels: int = 64,
        win_length: float = 0.025,
        hop_length: float = 0.010,
        bag_seconds: Optional[float] = None,
    ) -> None:
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_csv}")
        df = pd.read_csv(manifest_csv)
        if df.empty:
            self.records: List[BagRecord] = []
        else:
            self.records = self._build_records(df, audio_root, bag_seconds)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = int(round(win_length * sample_rate))
        self.hop_length = int(round(hop_length * sample_rate))
        self.label_to_index = {label: idx for idx, label in enumerate(PAM_CLASSES)}

    def _build_records(
        self, df: pd.DataFrame, audio_root: Optional[pathlib.Path], bag_seconds: Optional[float]
    ) -> List[BagRecord]:
        records: List[BagRecord] = []
        for bag_id, group in df.groupby("bag_id"):
            first_path = pathlib.Path(group.iloc[0].file_path)
            file_path = first_path if first_path.is_absolute() or audio_root is None else audio_root / first_path
            bag_len = bag_seconds or self._infer_bag_seconds(bag_id) or float(group.end_second.max() - group.start_second.min())
            labels = sorted(set(group.label))
            records.append(
                BagRecord(
                    bag_id=bag_id,
                    file_path=file_path,
                    start_second=float(group.start_second.min()),
                    duration=bag_len,
                    labels=labels,
                )
            )
        return records

    @staticmethod
    def _infer_bag_seconds(bag_id: str) -> Optional[float]:
        if "_bag" in bag_id and bag_id.endswith("s"):
            try:
                tail = bag_id.split("_bag", 1)[1]
                seconds_part = tail.split("s")[-2].split("_")[-1]
                return float(seconds_part)
            except Exception:
                return None
        return None

    def __len__(self) -> int:
        return len(self.records)

    def _one_hot(self, labels: Sequence[str]) -> np.ndarray:
        vec = np.zeros(len(PAM_CLASSES), dtype=np.float32)
        for label in labels:
            if label in self.label_to_index:
                vec[self.label_to_index[label]] = 1.0
        return vec

    def _load_features(self, record: BagRecord) -> np.ndarray:
        if record.file_path.suffix.lower() in {".npy", ".npz"}:
            arr = np.load(record.file_path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[list(arr.files)[0]]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D time-frequency array in {record.file_path}")
            return arr.astype(np.float32)

        audio, sample_rate = _load_wave(record.file_path)
        if sample_rate != self.sample_rate:
            audio = _resample(audio, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate
        segment = _slice_and_pad(audio, sample_rate, record.start_second, record.duration)
        return _logmel_from_audio(segment, sample_rate, self.n_mels, self.win_length, self.hop_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[idx]
        feat = self._load_features(record)
        labels = self._one_hot(record.labels)
        return torch.from_numpy(feat), torch.from_numpy(labels), record.bag_id


def pad_and_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, str]]):
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), []
    features, labels, bag_ids = zip(*batch)
    max_len = max(feat.shape[0] for feat in features)
    freq_bins = features[0].shape[1]
    padded = torch.zeros(len(features), max_len, freq_bins, dtype=torch.float32)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0], :] = feat
    label_tensor = torch.stack(labels).float()
    return padded, label_tensor, list(bag_ids)


class PamTALNetTrainer:
    """Lightweight trainer wrapping TALNet for PAM WSSED."""

    def __init__(self, model_args: argparse.Namespace, device: Optional[torch.device] = None):
        kernel_size = tuple(int(x) for x in str(model_args.kernel_size).split("x"))
        model_args.kernel_size = kernel_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PamTALNet(model_args, n_classes=len(PAM_CLASSES)).to(self.device)
        self.criterion = nn.BCELoss()

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        running_loss = 0.0
        total = 0
        for feats, labels, _ in loader:
            feats = feats.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            global_prob = self.model(feats)[0]
            loss = self.criterion(global_prob, labels)
            loss.backward()
            optimizer.step()
            batch_size = feats.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
        return running_loss / max(1, total)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        if len(loader.dataset) == 0:
            return {"loss": 0.0, "micro_f1": 0.0, "mAP": 0.0}
        total_loss = 0.0
        total = 0
        all_targets: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for feats, labels, _ in loader:
                feats = feats.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(feats)[0]
                loss = self.criterion(outputs, labels)
                batch_size = feats.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size
                all_targets.append(labels.cpu().numpy())
                all_probs.append(outputs.cpu().numpy())
        targets = np.vstack(all_targets)
        probs = np.vstack(all_probs)
        micro_f1 = f1_score(targets, probs >= 0.5, average="micro") if targets.size > 0 else 0.0
        try:
            mAP = average_precision_score(targets, probs, average="macro")
        except ValueError:
            mAP = 0.0
        return {"loss": total_loss / max(1, total), "micro_f1": micro_f1, "mAP": float(mAP)}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=2) if val_loader else None
        best_state = None
        best_val = -math.inf
        best_metrics: Optional[Dict[str, float]] = None
        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer)
            train_metrics = {"loss": train_loss}
            val_metrics = None
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                if val_metrics["micro_f1"] > best_val:
                    best_val = val_metrics["micro_f1"]
                    best_state = self.model.state_dict()
                    best_metrics = val_metrics
                if scheduler:
                    scheduler.step(val_metrics["micro_f1"])
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics or {}})
        if best_state is not None:
            self.model.load_state_dict(best_state)
        final_val = best_metrics or (history[-1]["val"] if history and history[-1]["val"] else None)
        return history[-1]["train"], final_val

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "classes": PAM_CLASSES}, path)


def _default_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        embedding_size=args.embedding_size,
        n_conv_layers=args.n_conv_layers,
        n_pool_layers=args.n_pool_layers,
        kernel_size=str(args.kernel_size),
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        pooling=args.pooling,
    )


def _build_loader(
    manifest: pathlib.Path,
    audio_root: Optional[pathlib.Path],
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
):
    dataset = PamBagDataset(manifest, audio_root=audio_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_and_collate)


def _load_optional_manifest(manifest: pathlib.Path) -> Optional[pathlib.Path]:
    return manifest if manifest.exists() and manifest.stat().st_size > 0 else None


def _run_cli():
    parser = argparse.ArgumentParser(description="Train TALNet on PAM WSSED manifests")
    parser.add_argument("--manifest-root", type=pathlib.Path, required=True, help="Directory containing experiment manifests")
    parser.add_argument("--config", type=str, required=True, help="Configuration folder name (e.g., instance1_bag60)")
    parser.add_argument("--audio-root", type=pathlib.Path, default=None, help="Root directory for audio files")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-size", type=int, default=1024)
    parser.add_argument("--n-conv-layers", dest="n_conv_layers", type=int, default=10)
    parser.add_argument("--n-pool-layers", dest="n_pool_layers", type=int, default=5)
    parser.add_argument("--kernel-size", type=str, default="3")
    parser.add_argument("--batch-norm", dest="batch_norm", action="store_true", default=True)
    parser.add_argument("--no-batch-norm", dest="batch_norm", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling", type=str, default="lin", choices=["max", "ave", "lin", "exp", "att"])
    args = parser.parse_args()

    config_dir = args.manifest_root / args.config
    train_manifest = config_dir / "train.csv"
    val_manifest = _load_optional_manifest(config_dir / "val.csv")
    test_manifest = _load_optional_manifest(config_dir / "test_anuraset.csv")
    fnjv_manifest = _load_optional_manifest(config_dir / "test_fnjv.csv")

    model_args = _default_model_args(args)
    trainer = PamTALNetTrainer(model_args)

    train_loader = _build_loader(train_manifest, args.audio_root, args.batch_size, args.num_workers, shuffle=True)
    val_loader = _build_loader(val_manifest, args.audio_root, args.batch_size, args.num_workers, shuffle=False) if val_manifest else None
    history_train, history_val = trainer.fit(train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    results: Dict[str, Dict[str, float]] = {"train": history_train}
    if history_val:
        results["val"] = history_val
    if test_manifest:
        test_loader = _build_loader(test_manifest, args.audio_root, args.batch_size, args.num_workers, shuffle=False)
        results["test_anuraset"] = trainer.evaluate(test_loader)
    if fnjv_manifest:
        fnjv_loader = _build_loader(fnjv_manifest, args.audio_root, args.batch_size, args.num_workers, shuffle=False)
        results["test_fnjv"] = trainer.evaluate(fnjv_loader)

    with open(config_dir / "pam_talnet_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    trainer.save(config_dir / "pam_talnet.pt")


if __name__ == "__main__":
    _run_cli()
