#!/usr/bin/env python3
"""Simple supervised MG baseline with weighted cross-entropy.

- Reads train/val/test split CSVs and a raw uint16 memmap .bin.
- Trains a timm backbone, default vit_small_patch8_224, for collapsed BI-RADS.
- Saves metrics, plots, confusion matrix, and best checkpoint.

Example:
python run_mg_supervised_weighted_ce_simple.py \
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
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import timm
except ImportError as exc:
    raise ImportError("Missing dependency: timm. Install with: pip install timm") from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
except ImportError as exc:
    raise ImportError("Missing dependency: scikit-learn. Install with: pip install scikit-learn") from exc


CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class ArgsForJson:
    full_csv: str
    bin: str
    train_csv: str
    val_csv: str
    test_csv: str
    output_dir: str
    backbone: str
    image_size: int
    image_height: int
    image_width: int
    memmap_dtype: str
    normalize_mode: str
    epochs: int
    patience: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    dropout: float
    drop_path_rate: float
    warmup_epochs: int
    seed: int
    train_max_samples: int
    minority_inclusive: bool
    hflip_p: float
    grad_clip_norm: float
    amp: bool
    pretrained: bool


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
    p.add_argument("--patience", default=20, type=int, help="Early stopping patience on val balanced accuracy. 0 disables.")
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

    p.add_argument("--train-max-samples", default=0, type=int, help="0 = use all train rows. If >0, subsample train rows.")
    p.add_argument("--minority-inclusive", action="store_true",
                   help="When --train-max-samples > 0, include all follow_up/biopsy first and fill with routine.")

    p.add_argument("--hflip-p", default=0.5, type=float, help="Random horizontal flip probability for train only.")
    p.add_argument("--no-amp", action="store_true", help="Disable BF16 autocast on CUDA.")
    return p.parse_args()


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_birads_numeric(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    for k in ["1", "2", "3", "4", "5", "6"]:
        if s == k or s.startswith(k + ".") or s.startswith(k + " ") or f"({k})" in s:
            return int(k)
    try:
        val = int(float(s))
        return val if val in {1, 2, 3, 4, 5, 6} else None
    except Exception:
        return None


def collapse_label_from_row(row: pd.Series) -> str:
    # Prefer already-derived collapsed_birads when present.
    if "collapsed_birads" in row and not pd.isna(row["collapsed_birads"]):
        s = str(row["collapsed_birads"]).strip().lower()
        s = s.replace("-", "_").replace(" ", "_")
        if s in CLASS_TO_ID:
            return s

    # Then support the actionability strings used in mg-only-all.csv.
    if "birads" in row and not pd.isna(row["birads"]):
        s = str(row["birads"]).strip().lower()
        if "suspicious" in s or "malignan" in s or "biopsy" in s:
            return "biopsy"
        if "follow" in s or "probably benign" in s:
            return "follow_up"
        if "routine" in s or "healthy" in s:
            return "routine"

    # Fallback to original_birads / numeric birads.
    for col in ["original_birads", "birads_numeric", "birads"]:
        if col in row:
            n = parse_birads_numeric(row[col])
            if n in (1, 2):
                return "routine"
            if n == 3:
                return "follow_up"
            if n in (4, 5, 6):
                return "biopsy"
    return "unknown"


def prepare_split(df: pd.DataFrame, full_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["collapsed_birads"] = df.apply(collapse_label_from_row, axis=1)
    df = df[df["collapsed_birads"].isin(CLASS_NAMES)].copy()
    df["target"] = df["collapsed_birads"].map(CLASS_TO_ID).astype(int)

    if "original_index" in df.columns:
        df["original_index"] = df["original_index"].astype(int)
        return df.reset_index(drop=True)

    # The full file has unique exam rows. Prefer exam mapping when possible.
    if "exam" in df.columns and "exam" in full_df.columns:
        if full_df["exam"].astype(str).is_unique:
            exam_to_idx = pd.Series(np.arange(len(full_df)), index=full_df["exam"].astype(str)).to_dict()
            df["original_index"] = df["exam"].astype(str).map(exam_to_idx)
        else:
            raise ValueError("full_csv exam column is not unique; cannot map split rows safely.")
    elif "id" in df.columns and "id" in full_df.columns and full_df["id"].astype(str).is_unique:
        id_to_idx = pd.Series(np.arange(len(full_df)), index=full_df["id"].astype(str)).to_dict()
        df["original_index"] = df["id"].astype(str).map(id_to_idx)
    else:
        raise ValueError(
            f"{split_name}: split needs original_index, or unique exam/id mapping via full_csv."
        )

    missing = int(df["original_index"].isna().sum())
    if missing:
        raise ValueError(f"{split_name}: could not map {missing} rows to original_index.")
    df["original_index"] = df["original_index"].astype(int)
    return df.reset_index(drop=True)


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


class MGMammoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bin_path: str | Path,
        full_num_rows: int,
        image_shape: tuple[int, int],
        dtype: str,
        image_size: int,
        train: bool,
        normalize_mode: str = "uint16",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        hflip_p: float = 0.5,
    ):
        self.df = df.reset_index(drop=True)
        self.bin_path = Path(bin_path)
        self.full_num_rows = int(full_num_rows)
        self.image_shape = image_shape
        self.dtype = np.dtype(dtype)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.normalize_mode = normalize_mode
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.hflip_p = float(hflip_p)
        self._imgs: Optional[np.memmap] = None

    def __len__(self) -> int:
        return len(self.df)

    def _open(self) -> np.memmap:
        if self._imgs is None:
            self._imgs = np.memmap(
                self.bin_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.full_num_rows, *self.image_shape),
            )
        return self._imgs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        original_index = int(row["original_index"])
        arr = self._open()[original_index].astype(np.float32)

        if self.normalize_mode == "uint16":
            if self.dtype == np.dtype("uint16"):
                arr = arr / 65535.0
            else:
                arr = arr / max(float(arr.max()), 1.0)
        elif self.normalize_mode == "per_image_percentile":
            lo, hi = np.percentile(arr, [self.percentile_low, self.percentile_high])
            arr = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)

        x = torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)
        if self.train and self.hflip_p > 0 and random.random() < self.hflip_p:
            x = torch.flip(x, dims=[2])
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        y = torch.tensor(int(row["target"]), dtype=torch.long)
        return x, y


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_model(args: argparse.Namespace) -> nn.Module:
    return timm.create_model(
        args.backbone,
        pretrained=bool(args.pretrained),
        num_classes=len(CLASS_NAMES),
        img_size=args.image_size,
        in_chans=1,
        drop_rate=float(args.dropout),
        drop_path_rate=float(args.drop_path_rate),
    )


def lr_for_epoch(args: argparse.Namespace, epoch_zero_based: int) -> float:
    if args.warmup_epochs > 0 and epoch_zero_based < args.warmup_epochs:
        return args.learning_rate * float(epoch_zero_based + 1) / float(args.warmup_epochs)
    progress = (epoch_zero_based - args.warmup_epochs + 1) / max(1, args.epochs - args.warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    all_logits = []
    all_y = []
    total_loss = 0.0
    total_n = 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    use_amp = amp and device.type == "cuda"
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = ce(logits.float(), y)
        total_loss += float(loss.item())
        total_n += int(y.numel())
        all_logits.append(logits.float().cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    return {
        "loss": total_loss / max(1, total_n),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


def plot_training_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 4))
    for i, key in enumerate(["train_loss", "val_loss", "val_balanced_accuracy", "val_macro_f1"]):
        ax = plt.subplot(1, 4, i + 1)
        ax.plot(epochs, history[key])
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], output_dir: Path, filename: str, title: str) -> None:
    arr = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(arr)
    plt.colorbar(im)
    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(int(arr[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def plot_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    names = ["train", "val", "test"]
    dfs = [train_df, val_df, test_df]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    plt.figure(figsize=(7, 4))
    for i, (name, df) in enumerate(zip(names, dfs)):
        counts = df["target"].value_counts().reindex([0, 1, 2], fill_value=0).values
        plt.bar(x + (i - 1) * width, counts, width=width, label=name)
    plt.xticks(x, CLASS_NAMES)
    plt.ylabel("rows")
    plt.title("Class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.amp = not bool(args.no_amp)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_json = ArgsForJson(
        full_csv=args.full_csv,
        bin=args.bin,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        image_height=args.image_height,
        image_width=args.image_width,
        memmap_dtype=args.memmap_dtype,
        normalize_mode=args.normalize_mode,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        train_max_samples=args.train_max_samples,
        minority_inclusive=bool(args.minority_inclusive),
        hflip_p=args.hflip_p,
        grad_clip_norm=args.grad_clip_norm,
        amp=bool(args.amp),
        pretrained=bool(args.pretrained),
    )
    save_json(asdict(args_json), output_dir / "config.json")

    print("Loading CSVs...", flush=True)
    full_df = read_csv(args.full_csv)
    train_df = prepare_split(read_csv(args.train_csv), full_df, "train")
    val_df = prepare_split(read_csv(args.val_csv), full_df, "val")
    test_df = prepare_split(read_csv(args.test_csv), full_df, "test")
    train_df = maybe_subsample_train(train_df, args.train_max_samples, args.minority_inclusive, args.seed)

    split_summary = {
        name: {
            "rows": int(len(df)),
            "collapsed_birads_counts": {str(k): int(v) for k, v in df["collapsed_birads"].value_counts().to_dict().items()},
        }
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    print(json.dumps(split_summary, indent=2), flush=True)
    save_json(split_summary, output_dir / "split_summary.json")
    train_df.to_csv(output_dir / "train_used.csv", index=False)
    plot_class_distribution(train_df, val_df, test_df, output_dir)

    expected_bytes = len(full_df) * args.image_height * args.image_width * np.dtype(args.memmap_dtype).itemsize
    actual_bytes = Path(args.bin).stat().st_size
    print(f"BIN check: expected={expected_bytes/1024**3:.3f} GiB actual={actual_bytes/1024**3:.3f} GiB match={expected_bytes == actual_bytes}")
    if expected_bytes != actual_bytes:
        raise RuntimeError("full_csv row count and bin size do not match.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    shape = (args.image_height, args.image_width)
    n_full = len(full_df)
    train_ds = MGMammoDataset(
        train_df, args.bin, n_full, shape, args.memmap_dtype, args.image_size, True,
        args.normalize_mode, args.percentile_low, args.percentile_high, args.hflip_p,
    )
    val_ds = MGMammoDataset(
        val_df, args.bin, n_full, shape, args.memmap_dtype, args.image_size, False,
        args.normalize_mode, args.percentile_low, args.percentile_high, 0.0,
    )
    test_ds = MGMammoDataset(
        test_df, args.bin, n_full, shape, args.memmap_dtype, args.image_size, False,
        args.normalize_mode, args.percentile_low, args.percentile_high, 0.0,
    )
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, train=True)
    val_loader = make_loader(val_ds, args.eval_batch_size, args.num_workers, train=False)
    test_loader = make_loader(test_ds, args.eval_batch_size, args.num_workers, train=False)

    class_weights = compute_class_weights(train_df["target"].to_numpy()).to(device)
    print("Class weights:", {CLASS_NAMES[i]: float(class_weights[i].cpu()) for i in range(3)}, flush=True)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = args.amp and device.type == "cuda"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "lr": [],
        "epoch_time_sec": [],
    }
    best_val_bal_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("Starting training...", flush=True)
    for epoch in range(args.epochs):
        start = time.perf_counter()
        lr = lr_for_epoch(args, epoch)
        set_optimizer_lr(optimizer, lr)
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            running_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_metrics = evaluate(model, val_loader, device, args.amp)

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["lr"].append(float(lr))
        history["epoch_time_sec"].append(float(time.perf_counter() - start))

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | lr={lr:.2e}",
            flush=True,
        )

        save_json(history, output_dir / "training_history.json")
        plot_training_curves(history, output_dir)

        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "class_weights": class_weights.detach().cpu(),
                    "args": asdict(args_json),
                    "val_metrics": val_metrics,
                },
                output_dir / "best_by_val_balanced_accuracy.pt",
            )
            plot_confusion_matrix(val_metrics["confusion_matrix"], output_dir, "confusion_matrix_val_best.png", "Validation confusion matrix")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch+1} epochs. Best epoch: {best_epoch}", flush=True)
            break

    print("Loading best checkpoint and evaluating test set...", flush=True)
    best = torch.load(output_dir / "best_by_val_balanced_accuracy.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, args.amp)
    plot_confusion_matrix(test_metrics["confusion_matrix"], output_dir, "confusion_matrix_test.png", "Test confusion matrix")

    summary = {
        "best_epoch_val_balanced_accuracy": int(best_epoch),
        "best_val_balanced_accuracy": float(best_val_bal_acc),
        "test": test_metrics,
        "final_epoch": len(history["train_loss"]),
        "class_names": CLASS_NAMES,
        "split_summary": split_summary,
        "config": asdict(args_json),
    }
    save_json(summary, output_dir / "metrics.json")
    print("Test summary:", flush=True)
    print(json.dumps({k: test_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2), flush=True)
    print(f"Wrote outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
