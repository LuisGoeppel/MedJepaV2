#!/usr/bin/env python3
"""Simple supervised MG baseline with weighted cross-entropy.

- Reads train/val/test split CSVs and a raw uint16 memmap .bin.
- Trains a timm backbone, default vit_small_patch8_224, for collapsed BI-RADS.
- Saves metrics, plots, confusion matrix, and best checkpoint.

Example:
python -m supervised.run_baseline \
  --full-csv /path/mg-only-all.csv \
  --bin /path/mg-only-all.bin \
  --train-csv /path/mg_train.csv \
  --val-csv /path/mg_val.csv \
  --test-csv /path/mg_test.csv \
  --output-dir /path/output \
  --image-size 224
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")


from core.config import save_json
from core.data import (
    CLASS_NAMES,
    set_seed,
    read_csv_clean as read_csv,
    prepare_baseline_split as prepare_split,
    validate_split_frames,
    BaselineDataset as MGMammoDataset,
    validate_bin,
)
from core.models import build_supervised_model
from core.supervised import compute_class_weights, evaluate, fit_supervised, make_baseline_loader
from core.plotting import plot_baseline_history, plot_confusion, plot_class_distribution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple supervised MG weighted-CE baseline.")
    p.add_argument("--full-csv", required=True, type=str, help="Full mg-only-all.csv; used for memmap row mapping.")
    p.add_argument("--bin", required=True, type=str, help="Raw image memmap .bin.")
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--val-csv", required=True, type=str)
    p.add_argument("--test-csv", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument("--backbone", default="vit_small_patch8_224", type=str)
    p.add_argument("--image-size", default=224, type=int)
    p.add_argument("--image-height", default=512, type=int)
    p.add_argument("--image-width", default=512, type=int)
    p.add_argument("--memmap-dtype", default="uint16", type=str)
    p.add_argument("--normalize-mode", default="uint16", choices=["uint16", "per_image_percentile"])
    p.add_argument("--percentile-low", default=1.0, type=float)
    p.add_argument("--percentile-high", default=99.0, type=float)

    p.add_argument("--epochs", default=100, type=int)
    p.add_argument(
        "--patience", default=20, type=int, help="Early stopping patience on val balanced accuracy. 0 disables."
    )
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--eval-batch-size", default=256, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--learning-rate", default=3e-4, type=float)
    p.add_argument("--min-learning-rate", default=1e-5, type=float)
    p.add_argument("--weight-decay", default=5e-2, type=float)
    p.add_argument("--dropout", default=0.0, type=float, help="timm drop_rate.")
    p.add_argument("--drop-path-rate", default=0.1, type=float)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--grad-clip-norm", default=0.0, type=float)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights if available.")

    p.add_argument(
        "--train-max-samples", default=0, type=int, help="0 = use all train rows. If >0, subsample train rows."
    )
    p.add_argument(
        "--minority-inclusive",
        action="store_true",
        help="When --train-max-samples > 0, include all follow_up/biopsy first and fill with routine.",
    )

    p.add_argument("--hflip-p", default=0.5, type=float, help="Random horizontal flip probability for train only.")
    p.add_argument("--no-amp", action="store_true", help="Disable BF16 autocast on CUDA.")
    return p.parse_args()


def maybe_subsample_train(df: pd.DataFrame, max_samples: int, minority_inclusive: bool, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if not minority_inclusive:
        idx = rng.choice(len(df), size=max_samples, replace=False)
        return df.iloc[idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    minority = df[df["collapsed_birads"].isin(["follow_up", "biopsy"])]
    if len(minority) > max_samples:
        # Keep a balanced minority subset if the requested size is too small.
        parts = []
        per = max_samples // 2
        for cls in ["follow_up", "biopsy"]:
            sub = df[df["collapsed_birads"] == cls]
            take = min(len(sub), per)
            parts.append(sub.sample(n=take, random_state=seed))
        out = pd.concat(parts, ignore_index=False)
        if len(out) < max_samples:
            rest = df.drop(index=out.index)
            out = pd.concat([out, rest.sample(n=max_samples - len(out), random_state=seed)], ignore_index=False)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    remaining = max_samples - len(minority)
    routine = df[df["collapsed_birads"] == "routine"]
    routine_sample = routine.sample(n=min(remaining, len(routine)), random_state=seed)
    out = pd.concat([minority, routine_sample], ignore_index=False)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def lr_for_epoch(args: argparse.Namespace, epoch_zero_based: int) -> float:
    if args.warmup_epochs > 0 and epoch_zero_based < args.warmup_epochs:
        return args.learning_rate * float(epoch_zero_based + 1) / float(args.warmup_epochs)
    progress = (epoch_zero_based - args.warmup_epochs + 1) / max(1, args.epochs - args.warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def main() -> None:
    args = parse_args()
    args.amp = not bool(args.no_amp)
    set_seed(args.seed, benchmark=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_json = args
    save_json(vars(args_json), output_dir / "config.json")

    print("Loading CSVs...", flush=True)
    full_df = read_csv(args.full_csv)
    train_df = prepare_split(read_csv(args.train_csv), full_df, "train")
    val_df = prepare_split(read_csv(args.val_csv), full_df, "val")
    test_df = prepare_split(read_csv(args.test_csv), full_df, "test")
    validate_split_frames(train_df, val_df, test_df)
    train_df = maybe_subsample_train(train_df, args.train_max_samples, args.minority_inclusive, args.seed)

    split_summary = {
        name: {
            "rows": int(len(df)),
            "collapsed_birads_counts": {
                str(k): int(v) for k, v in df["collapsed_birads"].value_counts().to_dict().items()
            },
        }
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    print(json.dumps(split_summary, indent=2), flush=True)
    save_json(split_summary, output_dir / "split_summary.json")
    train_df.to_csv(output_dir / "train_used.csv", index=False)
    plot_class_distribution(
        *(df.rename(columns={"target": "target_collapsed"}) for df in (train_df, val_df, test_df)),
        output_dir / "class_distribution.png",
    )

    validate_bin(args.bin, len(full_df), args.image_height, args.image_width, args.memmap_dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    shape = (args.image_height, args.image_width)
    n_full = len(full_df)
    train_ds = MGMammoDataset(
        train_df,
        args.bin,
        n_full,
        shape,
        args.memmap_dtype,
        args.image_size,
        True,
        args.normalize_mode,
        args.percentile_low,
        args.percentile_high,
        args.hflip_p,
    )
    val_ds = MGMammoDataset(
        val_df,
        args.bin,
        n_full,
        shape,
        args.memmap_dtype,
        args.image_size,
        False,
        args.normalize_mode,
        args.percentile_low,
        args.percentile_high,
        0.0,
    )
    test_ds = MGMammoDataset(
        test_df,
        args.bin,
        n_full,
        shape,
        args.memmap_dtype,
        args.image_size,
        False,
        args.normalize_mode,
        args.percentile_low,
        args.percentile_high,
        0.0,
    )
    train_loader = make_baseline_loader(train_ds, args.batch_size, args.num_workers, train=True, drop_last=True)
    val_loader = make_baseline_loader(val_ds, args.eval_batch_size, args.num_workers, train=False)
    test_loader = make_baseline_loader(test_ds, args.eval_batch_size, args.num_workers, train=False)

    class_weights = compute_class_weights(train_df["target"].to_numpy(), dtype=np.float64).to(device)
    print("Class weights:", {CLASS_NAMES[i]: float(class_weights[i].cpu()) for i in range(3)}, flush=True)

    model = build_supervised_model(
        args.backbone,
        args.image_size,
        args.pretrained,
        len(CLASS_NAMES),
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def checkpoint(path, epoch, metrics):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "class_weights": class_weights.detach().cpu(),
                "args": vars(args_json),
                "val_metrics": metrics,
            },
            path,
        )

    def save_history(rows):
        history = {key: [row[key] for row in rows] for key in rows[0]}
        save_json(history, output_dir / "training_history.json")
        plot_baseline_history(history, output_dir / "training_curves.png")

    fit = fit_supervised(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=args.epochs,
        amp=args.amp,
        criterion=criterion,
        output_dir=output_dir,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        lr_for_epoch=lambda epoch: lr_for_epoch(args, epoch),
        checkpoint_names={"balanced_accuracy": "best_by_val_balanced_accuracy.pt"},
        save_checkpoint=checkpoint,
        history_writer=save_history,
        eval_progress=True,
    )
    history = {key: [row[key] for row in fit["history"]] for key in fit["history"][0]}
    best_epoch = fit["best_epochs"]["balanced_accuracy"]
    best_val = fit["best_metrics"]["balanced_accuracy"]
    best_val_bal_acc = best_val["balanced_accuracy"]
    plot_confusion(
        best_val["confusion_matrix"], "Validation confusion matrix", output_dir / "confusion_matrix_val_best.png"
    )

    print("Loading best checkpoint and evaluating test set...", flush=True)
    best = torch.load(output_dir / "best_by_val_balanced_accuracy.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, args.amp, show_progress=True)
    plot_confusion(test_metrics["confusion_matrix"], "Test confusion matrix", output_dir / "confusion_matrix_test.png")

    summary = {
        "best_epoch_val_balanced_accuracy": int(best_epoch),
        "best_val_balanced_accuracy": float(best_val_bal_acc),
        "test": test_metrics,
        "final_epoch": len(history["train_loss"]),
        "class_names": CLASS_NAMES,
        "split_summary": split_summary,
        "config": vars(args_json),
    }
    save_json(summary, output_dir / "metrics.json")
    print("Test summary:", flush=True)
    print(
        json.dumps(
            {k: test_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2
        ),
        flush=True,
    )
    print(f"Wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
