"""Threshold selection utilities."""

from __future__ import annotations

import numpy as np
import torch


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
