"""Visualization helpers for training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
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
