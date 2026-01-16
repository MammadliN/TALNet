"""Launch comparative TALNet sweeps across 6 V100 GPUs.

- GPUs 0/1/2 run the target-species grid (all poolings x bag lengths).
- GPUs 3/4/5 run the all-species grid (all poolings x bag lengths).
"""

from __future__ import annotations

import concurrent.futures
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Experiment grid
POOLINGS: Sequence[str] = ["max", "ave", "lin", "exp", "att"]
BAG_LENGTHS: Sequence[int] = [3, 10, 15, 30, 60]
TARGET_SPECIES: Sequence[str] = [
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

# GPU assignments
TARGET_GPUS: Sequence[str] = ["0", "1", "2"]
ALL_SPECIES_GPUS: Sequence[str] = ["3", "4", "5"]

DEFAULT_ROOT = Path(r"/ds-iml/Bioacoustics/AnuraSet/raw_data")
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Task:
    name: str
    gpu: str
    pooling: str
    bag_seconds: int
    all_species: bool
    target_species: Sequence[str]

    def command(self) -> List[str]:
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "main.py"),
            "--pooling",
            self.pooling,
            "--bag_seconds",
            str(self.bag_seconds),
            "--root",
            str(DEFAULT_ROOT),
            "--batch_size",
            "8",
            "--num_workers",
            "4",
            "--epochs",
            "300",
        ]
        if self.all_species:
            cmd.append("--all_species")
        else:
            cmd.extend(["--target_species", *self.target_species])
        return cmd


def build_tasks(
    gpu_ids: Sequence[str],
    poolings: Sequence[str],
    bag_lengths: Sequence[int],
    use_all_species: bool,
    target_species: Sequence[str],
) -> List[Task]:
    tasks: List[Task] = []
    gpu_cycle = itertools.cycle(gpu_ids)
    for pooling, bag_seconds in itertools.product(poolings, bag_lengths):
        tasks.append(
            Task(
                name=f"{'all' if use_all_species else 'target'}_{pooling}_{bag_seconds}s",
                gpu=next(gpu_cycle),
                pooling=pooling,
                bag_seconds=bag_seconds,
                all_species=use_all_species,
                target_species=target_species,
            )
        )
    return tasks


def run_task(task: Task) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = task.gpu
    print(f"[dispatch] GPU {task.gpu}: starting {task.name}")
    result = subprocess.run(task.command(), cwd=PROJECT_DIR, env=env)
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"[dispatch] GPU {task.gpu}: finished {task.name} -> {status}")
    return result.returncode


def run_tasks_for_gpu(gpu_id: str, tasks: Sequence[Task]) -> Dict[str, int]:
    results: Dict[str, int] = {}
    for task in tasks:
        results[task.name] = run_task(task)
    return results


def dispatch(tasks: Iterable[Task]) -> None:
    tasks_by_gpu: Dict[str, List[Task]] = {}
    for task in tasks:
        tasks_by_gpu.setdefault(task.gpu, []).append(task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks_by_gpu)) as executor:
        future_map = {
            executor.submit(run_tasks_for_gpu, gpu, gpu_tasks): gpu
            for gpu, gpu_tasks in tasks_by_gpu.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            gpu = future_map[future]
            try:
                results = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[dispatch] GPU {gpu} worker failed: {exc}")
                continue
            for name, code in results.items():
                if code != 0:
                    print(f"[dispatch] Task {name} on GPU {gpu} failed with code {code}")


def main() -> None:
    target_tasks = build_tasks(TARGET_GPUS, POOLINGS, BAG_LENGTHS, False, TARGET_SPECIES)
    all_species_tasks = build_tasks(ALL_SPECIES_GPUS, POOLINGS, BAG_LENGTHS, True, TARGET_SPECIES)

    print("[info] Launching target-species grid across GPUs 0/1/2...")
    print("[info] Launching all-species grid across GPUs 3/4/5...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(dispatch, target_tasks),
            executor.submit(dispatch, all_species_tasks),
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
