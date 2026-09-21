#!/usr/bin/env python3
"""Create MedJEPA training diagnostic plots from a training_history.json file.

This script is intentionally independent from the training script so it can be run
while a long training job is still running, after a crash, or on old runs.

Examples:
  python plot_medjepa_training_history.py \
    --run-dir /pfss/.../luis/runs/my_run

  python plot_medjepa_training_history.py \
    --history-json /pfss/.../metrics/training_history.json \
    --output-dir /pfss/.../plots

  # Update plots every 10 minutes:
  python plot_medjepa_training_history.py --run-dir /pfss/.../runs/my_run --watch-seconds 600
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MedJEPA training diagnostics from training_history.json.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory containing metrics/training_history.json and plots/. Optional if --history-json is given.")
    parser.add_argument("--history-json", type=Path, default=None,
                        help="Path to training_history.json. Overrides --run-dir/metrics/training_history.json.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for PNG plots. Defaults to --run-dir/plots or history_json.parent/../plots.")
    parser.add_argument("--watch-seconds", type=int, default=0,
                        help="If >0, regenerate plots every N seconds until interrupted.")
    parser.add_argument("--max-read-retries", type=int, default=5,
                        help="Retries if JSON is temporarily invalid while training writes it.")
    parser.add_argument("--retry-sleep", type=float, default=1.0,
                        help="Seconds to sleep between JSON read retries.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for saved PNG files.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.history_json is not None:
        history_json = args.history_json
    elif args.run_dir is not None:
        history_json = args.run_dir / "metrics" / "training_history.json"
    else:
        raise SystemExit("Provide either --run-dir or --history-json.")

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.run_dir is not None:
        output_dir = args.run_dir / "plots"
    else:
        # history_json = .../metrics/training_history.json -> .../plots
        parent = history_json.parent
        output_dir = parent.parent / "plots" if parent.name == "metrics" else parent / "plots"

    return history_json, output_dir


def load_history(path: Path, retries: int, retry_sleep: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
            return data
        except Exception as exc:  # JSON may be half-written while training is running.
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"Could not read valid history JSON from {path}: {last_error}")


def as_float_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.asarray([], dtype=float)
    out: list[float] = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            x = math.nan
        out.append(x)
    return np.asarray(out, dtype=float)


def history_len(history: dict[str, Any]) -> int:
    if "lejepa" in history and isinstance(history["lejepa"], list):
        return len(history["lejepa"])
    lengths = [len(v) for v in history.values() if isinstance(v, list)]
    return max(lengths) if lengths else 0


def plot_series(ax: plt.Axes, epochs: np.ndarray, history: dict[str, Any], key: str, label: str | None = None) -> bool:
    y = as_float_array(history.get(key, []))
    if len(y) == 0:
        return False
    n = min(len(epochs), len(y))
    if n == 0:
        return False
    ax.plot(epochs[:n], y[:n], label=label or key)
    return True


def savefig_atomic(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the final image extension as the actual suffix so that matplotlib
    # can infer/write the correct file format. For example, use
    # training_history.tmp.png instead of training_history.png.tmp.
    suffix = path.suffix.lstrip(".")
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)

    plt.savefig(tmp, dpi=dpi, format=suffix, bbox_inches="tight")
    os.replace(tmp, path)


def plot_training_history(history: dict[str, Any], output_dir: Path, dpi: int = 200) -> None:
    n_epochs = history_len(history)
    if n_epochs == 0:
        raise ValueError("No epoch history found. Is training_history.json still empty?")
    epochs = np.arange(1, n_epochs + 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Main objective history.
    plt.figure(figsize=(16, 4))
    for i, key in enumerate(["lejepa", "invariance", "sigreg", "lr"]):
        ax = plt.subplot(1, 4, i + 1)
        plot_series(ax, epochs, history, key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_atomic(output_dir / "training_history.png", dpi=dpi)
    plt.close()

    # 2) Collapse diagnostics in one compact 2x3 grid.
    plt.figure(figsize=(15, 8))
    for i, key in enumerate(["lejepa", "invariance", "sigreg", "raw_proj_std", "loss_proj_std", "emb_std"]):
        ax = plt.subplot(2, 3, i + 1)
        plot_series(ax, epochs, history, key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_atomic(output_dir / "collapse_diagnostics.png", dpi=dpi)
    plt.close()

    # 3) Projection/embedding diagnostics.
    plt.figure(figsize=(15, 8))
    diag_groups = [
        ("standard deviation", ["raw_proj_std", "loss_proj_std", "emb_std"]),
        ("mean vector norm", ["raw_proj_norm", "loss_proj_norm", "emb_norm"]),
        ("effective rank", ["raw_proj_effective_rank", "loss_proj_effective_rank", "emb_effective_rank"]),
    ]
    for row, (title, keys) in enumerate(diag_groups):
        ax = plt.subplot(3, 1, row + 1)
        any_line = False
        for key in keys:
            any_line = plot_series(ax, epochs, history, key) or any_line
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if any_line:
            ax.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(output_dir / "training_diagnostics.png", dpi=dpi)
    savefig_atomic(output_dir / "projection_embedding_diagnostics.png", dpi=dpi)
    plt.close()

    # 4) Epoch time.
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plot_series(ax, epochs, history, "epoch_time_sec")
    ax.set_title("epoch_time_sec")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_atomic(output_dir / "epoch_times.png", dpi=dpi)
    plt.close()

    # 5) Timing components.
    timing_keys = [
        "data_wait_and_augmentation_time_sec",
        "h2d_transfer_time_sec",
        "forward_loss_time_sec",
        "backward_optimizer_time_sec",
        "metrics_bookkeeping_time_sec",
    ]
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    any_line = False
    for key in timing_keys:
        any_line = plot_series(ax, epochs, history, key, label=key.replace("_time_sec", "")) or any_line
    ax.set_title("Epoch timing components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(alpha=0.3)
    if any_line:
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(output_dir / "epoch_timing_components.png", dpi=dpi)
    plt.close()

    # 6) Timing fractions.
    fraction_keys = [
        "data_wait_and_augmentation_fraction",
        "h2d_transfer_fraction",
        "forward_loss_fraction",
        "backward_optimizer_fraction",
        "metrics_bookkeeping_fraction",
    ]
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    any_line = False
    for key in fraction_keys:
        any_line = plot_series(ax, epochs, history, key, label=key.replace("_fraction", "")) or any_line
    ax.set_title("Epoch timing fractions")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of epoch wall time")
    ax.grid(alpha=0.3)
    if any_line:
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(output_dir / "epoch_timing_fractions.png", dpi=dpi)
    plt.close()


def run_once(args: argparse.Namespace) -> None:
    history_json, output_dir = resolve_paths(args)
    history = load_history(history_json, retries=args.max_read_retries, retry_sleep=args.retry_sleep)
    plot_training_history(history, output_dir=output_dir, dpi=args.dpi)
    print(f"Wrote plots for {history_len(history)} epochs to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.watch_seconds and args.watch_seconds > 0:
        print(f"Watching history and updating plots every {args.watch_seconds} seconds. Press Ctrl+C to stop.")
        while True:
            try:
                run_once(args)
            except Exception as exc:
                print(f"WARNING: plot update failed: {exc}")
            time.sleep(args.watch_seconds)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
