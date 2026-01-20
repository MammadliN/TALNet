"""Dataset utilities for AnuraSet and FNJV."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

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


def discover_label_columns(metadata: pd.DataFrame) -> List[str]:
    return [c for c in metadata.columns if c not in RESERVED_COLUMNS]


def select_label_columns(metadata: pd.DataFrame, target_species: Sequence[str]) -> List[str]:
    label_columns = discover_label_columns(metadata)
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


def load_audio_segment(path: Path, sample_rate: int, start: float, duration: float) -> np.ndarray:
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


def ensure_audio_path(root: Path, row: pd.Series) -> Path:
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
            audio_path = ensure_audio_path(self.root, group.iloc[0])
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
        y = load_audio_segment(
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


class FNJVDataset(Dataset):
    def __init__(
        self,
        root: Path,
        metadata: pd.DataFrame,
        label_columns: Sequence[str],
        sample_rate: int = 22000,
        fps: float = 40.0,
        n_mels: int = 64,
        n_fft: int = 1100,
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
        y = load_audio_segment(audio_path, self.sample_rate, start=0.0, duration=60.0)
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
