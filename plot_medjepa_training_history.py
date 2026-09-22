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

from core.plotting import history_len, plot_training_history

import argparse
import json
import time
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MedJEPA training diagnostics from training_history.json.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing metrics/training_history.json and plots/. Optional if --history-json is given.",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Path to training_history.json. Overrides --run-dir/metrics/training_history.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to --run-dir/plots or history_json.parent/../plots.",
    )
    parser.add_argument(
        "--watch-seconds", type=int, default=0, help="If >0, regenerate plots every N seconds until interrupted."
    )
    parser.add_argument(
        "--max-read-retries",
        type=int,
        default=5,
        help="Retries if JSON is temporarily invalid while training writes it.",
    )
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds to sleep between JSON read retries.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved PNG files.")
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
