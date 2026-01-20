"""Metric helpers for tagging/localization."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


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
