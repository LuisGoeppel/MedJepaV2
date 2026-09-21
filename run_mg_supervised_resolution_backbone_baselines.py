#!/usr/bin/env python3
"""
Supervised MG BI-RADS resolution/backbone baseline.

Runs:
  image sizes: 224, 384, 512
  backbones: vit_small_patch16_224 and resnet18 by default
  train subset: 50k examples from the train pool, including all follow_up and biopsy
                examples from the train pool, filled with routine examples
  loss: weighted cross entropy, weights computed from the sampled train subset

Outputs one folder per experiment plus summary.csv/json/report.pdf.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import traceback
from contextlib import redirect_stdout, redirect_stderr
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import timm
except ImportError as exc:
    raise SystemExit("ERROR: timm is not installed. Run: python -m pip install timm") from exc

try:
    import cv2
except ImportError as exc:
    raise SystemExit("ERROR: opencv-python-headless is not installed. Run: python -m pip install opencv-python-headless") from exc


CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class BinSpec:
    dtype: str
    height: int
    width: int
    channels: int
    exact_match: bool


@dataclass
class ExperimentSpec:
    name: str
    backbone: str
    image_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str) -> str:
    value = str(value).strip().lower().replace("/", "_")
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def parse_birads_number(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "missing", "unknown"}:
        return None
    m = re.search(r"\(([0-6])\)", text)
    if m:
        return int(m.group(1))
    try:
        f = float(text)
        if np.isfinite(f):
            return int(f)
    except Exception:
        pass
    m = re.search(r"(^|[^0-9])([0-6])([^0-9]|$)", text)
    if m:
        return int(m.group(2))
    return None


def collapse_birads_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    raw = str(value).strip().lower()
    if raw in {"", "nan", "none", "missing", "unknown"}:
        return None
    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if norm in CLASS_TO_INDEX:
        return norm

    # Native MG actionability strings.
    # Order matters: "probably benign" should be follow_up, not routine.
    if "biopsy" in raw or "suspicious" in raw or "malignan" in raw:
        return "biopsy"
    if "follow" in raw or "probably benign" in raw or "probably_benign" in norm:
        return "follow_up"
    if "routine" in raw or "healthy" in raw or "negative" in raw or raw == "benign" or norm == "benign":
        return "routine"

    n = parse_birads_number(value)
    if n is None:
        return None
    if n in {1, 2}:
        return "routine"
    if n in {0, 3}:
        return "follow_up"
    if n in {4, 5, 6}:
        return "biopsy"
    return None


def ensure_collapsed_birads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "collapsed_birads" in out.columns:
        out["collapsed_birads"] = out["collapsed_birads"].map(collapse_birads_value)
        return out
    candidate_cols = [c for c in ["birads", "original_birads", "birads_numeric"] if c in out.columns]
    if not candidate_cols:
        raise ValueError("Could not derive collapsed_birads: no birads/original_birads/birads_numeric column found.")
    source_col = candidate_cols[0]
    print(f"collapsed_birads not found; deriving it from '{source_col}'")
    out["collapsed_birads"] = out[source_col].map(collapse_birads_value)
    return out


def infer_machine_family(machine: object) -> str:
    if pd.isna(machine):
        return "unknown"
    text = str(machine).lower()
    if any(x in text for x in ["hologic", "lorad", "selenia", "dimensions", "3dimensions"]):
        return "Hologic/Lorad"
    if any(x in text for x in ["howtek", "lumisys", "lumysis"]):
        return "Howtek/Lumysis"
    if any(x in text for x in ["senographe", "ge healthcare", "general electric"]):
        return "GE/Senographe"
    if re.search(r"(^|[^a-z])ge([^a-z]|$)", text):
        return "GE/Senographe"
    return "unknown"


def normalize_view(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    cleaned = re.sub(r"[^A-Z0-9_ -]+", " ", text)
    tokens = set(re.split(r"[\s_/-]+", cleaned))
    for candidate in ["MLO", "CC", "LMO", "LM", "ML", "XCCL", "XCCM", "FB"]:
        if candidate in tokens or cleaned == candidate:
            return candidate
    m = re.search(r"(^|[^A-Z])(MLO|CC|XCCL|XCCM|LMO|LM|ML)([^A-Z]|$)", cleaned)
    if m:
        return m.group(2)
    return None


def infer_view_from_row(row: pd.Series) -> str:
    cols = [
        "view", "ViewPosition", "view_position", "viewposition", "projection", "position",
        "id", "context", "findings", "image_path", "path", "filename", "file",
        "dicom_path", "png_path", "jpg_path", "original_path", "exam",
    ]
    for col in cols:
        if col in row.index:
            v = normalize_view(row[col])
            if v is not None:
                return v
    return "unknown"


def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_collapsed_birads(df)
    out["source_index"] = np.arange(len(out), dtype=np.int64)
    out = out[out["collapsed_birads"].isin(CLASS_NAMES)].copy()
    out["target"] = out["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
    if "machine_family" not in out.columns:
        out["machine_family"] = out["machine"].map(infer_machine_family) if "machine" in out.columns else "unknown"
    if "view" not in out.columns:
        out["view"] = out.apply(infer_view_from_row, axis=1)
    return out


def infer_bin_spec(bin_path: Path, n_rows: int) -> BinSpec:
    actual = bin_path.stat().st_size
    candidates = [
        ("uint16", np.dtype("uint16"), 512, 512, 1),
        ("uint8", np.dtype("uint8"), 512, 512, 1),
        ("float32", np.dtype("float32"), 512, 512, 1),
        ("uint16", np.dtype("uint16"), 224, 224, 1),
        ("uint8", np.dtype("uint8"), 224, 224, 1),
    ]
    for dtype_name, dtype, h, w, c in candidates:
        if n_rows * h * w * c * dtype.itemsize == actual:
            return BinSpec(dtype=dtype_name, height=h, width=w, channels=c, exact_match=True)
    return BinSpec(dtype="uint16", height=512, width=512, channels=1, exact_match=False)


def open_memmap(bin_path: Path, spec: BinSpec, n_rows: int) -> np.memmap:
    dtype = np.dtype(spec.dtype)
    shape = (n_rows, spec.height, spec.width) if spec.channels == 1 else (n_rows, spec.height, spec.width, spec.channels)
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)


def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"rows": int(len(df))}
    if "patient" in df.columns:
        summary["patients"] = int(df["patient"].nunique(dropna=True))
    for col in ["collapsed_birads", "machine_family", "view", "dataset"]:
        if col in df.columns:
            summary[f"{col}_counts"] = {
                str(k): int(v)
                for k, v in df[col].fillna("missing").astype(str).value_counts().to_dict().items()
            }
    return summary


def mode_or_first(s: pd.Series) -> str:
    mode = s.mode()
    return str(mode.iloc[0]) if len(mode) > 0 else str(s.iloc[0])


def make_patient_disjoint_pools(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if "patient" not in df.columns:
        train_df, tmp = train_test_split(df, train_size=train_ratio, random_state=seed, shuffle=True, stratify=df["collapsed_birads"])
        val_fraction = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(tmp, train_size=val_fraction, random_state=seed + 1, shuffle=True, stratify=tmp["collapsed_birads"])
        return train_df.copy(), val_df.copy(), test_df.copy(), {"split_type": "row_level_fallback"}

    groups = (
        df.groupby("patient", dropna=False)
        .agg(n_rows=("patient", "size"), label=("collapsed_birads", mode_or_first))
        .reset_index()
    )
    try:
        train_groups, tmp_groups = train_test_split(groups, train_size=train_ratio, random_state=seed, shuffle=True, stratify=groups["label"])
    except ValueError:
        train_groups, tmp_groups = train_test_split(groups, train_size=train_ratio, random_state=seed, shuffle=True, stratify=None)
    val_fraction = val_ratio / (val_ratio + test_ratio)
    try:
        val_groups, test_groups = train_test_split(tmp_groups, train_size=val_fraction, random_state=seed + 1, shuffle=True, stratify=tmp_groups["label"])
    except ValueError:
        val_groups, test_groups = train_test_split(tmp_groups, train_size=val_fraction, random_state=seed + 1, shuffle=True, stratify=None)

    train_ids = set(train_groups["patient"])
    val_ids = set(val_groups["patient"])
    test_ids = set(test_groups["patient"])
    train_df = df[df["patient"].isin(train_ids)].copy()
    val_df = df[df["patient"].isin(val_ids)].copy()
    test_df = df[df["patient"].isin(test_ids)].copy()
    train_pat = set(train_df["patient"].dropna().astype(str))
    val_pat = set(val_df["patient"].dropna().astype(str))
    test_pat = set(test_df["patient"].dropna().astype(str))
    info = {
        "split_type": "patient_disjoint",
        "patient_overlap": {
            "train_vs_val": len(train_pat & val_pat),
            "train_vs_test": len(train_pat & test_pat),
            "val_vs_test": len(val_pat & test_pat),
        },
    }
    return train_df, val_df, test_df, info


def make_minority_inclusive_train_subset(
    train_pool: pd.DataFrame,
    train_size: int,
    seed: int,
    include_classes: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    parts = []
    used_indices = set()
    included_counts = {}
    for cls in include_classes:
        cls_df = train_pool[train_pool["collapsed_birads"] == cls]
        parts.append(cls_df)
        used_indices.update(cls_df.index.tolist())
        included_counts[cls] = int(len(cls_df))
    n_included = sum(len(p) for p in parts)
    remaining = train_size - n_included
    if remaining < 0:
        raise ValueError(f"train_size={train_size} is smaller than included minority count={n_included}.")
    routine_df = train_pool[(train_pool["collapsed_birads"] == "routine") & (~train_pool.index.isin(used_indices))]
    if len(routine_df) < remaining:
        raise ValueError(f"Need {remaining} routine examples, but only {len(routine_df)} are available.")
    routine_sample = routine_df.sample(n=remaining, random_state=seed)
    parts.append(routine_sample)
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed + 100).copy()
    info = {
        "requested_train_size": int(train_size),
        "included_all_classes": list(include_classes),
        "included_counts": included_counts,
        "routine_fill_count": int(remaining),
        "final_counts": {str(k): int(v) for k, v in sampled["collapsed_birads"].value_counts().to_dict().items()},
    }
    return sampled, info


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    counts = train_df["target"].value_counts().to_dict()
    total = len(train_df)
    weights = []
    weights_dict = {}
    for i, cls in enumerate(CLASS_NAMES):
        count = int(counts.get(i, 0))
        if count <= 0:
            raise ValueError(f"Class '{cls}' has zero training samples.")
        w = total / (len(CLASS_NAMES) * count)
        weights.append(w)
        weights_dict[cls] = float(w)
    return torch.tensor(weights, dtype=torch.float32, device=device), weights_dict


class MGMammographyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mmap: np.memmap, image_size: int, train: bool, augment: bool) -> None:
        self.df = df.reset_index(drop=True)
        self.mmap = mmap
        self.image_size = int(image_size)
        self.train = train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(arr, dtype=np.float32)
        if self.augment and self.train and random.random() < 0.5:
            arr = np.flip(arr, axis=1).copy()
        if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
            arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = self.mmap[int(row["source_index"])]
        x = self._preprocess(img)
        y = int(row["target"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_model(backbone: str, image_size: int, pretrained: bool, num_classes: int) -> nn.Module:
    kwargs = {"pretrained": bool(pretrained), "num_classes": int(num_classes), "in_chans": 1}
    try:
        return timm.create_model(backbone, img_size=int(image_size), **kwargs)
    except TypeError:
        return timm.create_model(backbone, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, use_amp: bool) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n = 0
    ys: List[int] = []
    preds: List[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        pred = torch.argmax(logits, dim=1)
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(pred.detach().cpu().numpy().tolist())
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(ys, preds, labels=labels)
    rec = recall_score(ys, preds, labels=labels, average=None, zero_division=0)
    return {
        "loss": total_loss / max(1, n),
        "accuracy": float(accuracy_score(ys, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {CLASS_NAMES[i]: float(rec[i]) for i in labels},
        "confusion_matrix": cm.astype(int).tolist(),
    }


def plot_training_curves(history: List[Dict[str, Any]], out_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(epochs, [h["val_balanced_accuracy"] for h in history])
    axes[0, 1].set_title("Validation balanced accuracy"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(epochs, [h["val_macro_f1"] for h in history])
    axes[1, 0].set_title("Validation macro F1"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(epochs, [h["lr"] for h in history])
    axes[1, 1].set_title("Learning rate"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], out_path: Path, title: str) -> None:
    arr = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_checkpoint(out_path: Path, model: nn.Module, spec: ExperimentSpec, args: argparse.Namespace, epoch: int, metrics: Dict[str, Any], class_weights: Dict[str, float]) -> None:
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "class_weights": class_weights,
            "model_config": {
                "backbone": spec.backbone,
                "image_size": int(spec.image_size),
                "num_classes": len(CLASS_NAMES),
                "in_chans": 1,
                "pretrained": bool(args.pretrained),
            },
            "experiment": spec.__dict__,
        },
        out_path,
    )


def make_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, mmap: np.memmap, image_size: int, args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = MGMammographyDataset(train_df, mmap, image_size=image_size, train=True, augment=not args.no_augment)
    val_ds = MGMammographyDataset(val_df, mmap, image_size=image_size, train=False, augment=False)
    test_ds = MGMammographyDataset(test_df, mmap, image_size=image_size, train=False, augment=False)
    eval_batch_size = args.eval_batch_size or args.batch_size
    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common),
        DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common),
    )


def train_one_experiment(spec: ExperimentSpec, args: argparse.Namespace, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, mmap: np.memmap, device: torch.device, output_dir: Path) -> Dict[str, Any]:
    set_seed(args.seed + spec.image_size + len(spec.backbone))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {"train": summarize_df(train_df), "val": summarize_df(val_df), "test": summarize_df(test_df)}
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2, sort_keys=True), encoding="utf-8")
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "experiment": spec.__dict__,
        "split_summary": split_summary,
        "class_names": CLASS_NAMES,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, mmap, spec.image_size, args)
    model = make_model(spec.backbone, spec.image_size, args.pretrained, len(CLASS_NAMES)).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} CUDA devices")
        model = nn.DataParallel(model)

    weights_tensor, weights_dict = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    history: List[Dict[str, Any]] = []
    best_bal_acc, best_macro_f1, best_loss = -1.0, -1.0, float("inf")
    best_epochs = {"val_balanced_accuracy": 0, "val_macro_f1": 0, "val_loss": 0}
    no_improve = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        pbar = tqdm(train_loader, desc=f"{spec.name} epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * x.size(0)
            n_train += x.size(0)
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
        scheduler.step()
        train_loss = train_loss_sum / max(1, n_train)
        val = evaluate(model, val_loader, device, criterion, args.amp and torch.cuda.is_available())
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val["loss"]),
            "val_accuracy": float(val["accuracy"]),
            "val_balanced_accuracy": float(val["balanced_accuracy"]),
            "val_macro_f1": float(val["macro_f1"]),
            "lr": lr,
            "elapsed_seconds": float(time.time() - start),
        }
        history.append(row)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        improved = False
        if val["balanced_accuracy"] > best_bal_acc + args.min_delta:
            best_bal_acc = float(val["balanced_accuracy"]); best_epochs["val_balanced_accuracy"] = epoch
            save_checkpoint(output_dir / "best_by_val_balanced_accuracy.pt", model, spec, args, epoch, val, weights_dict)
            improved = True
        if val["macro_f1"] > best_macro_f1 + args.min_delta:
            best_macro_f1 = float(val["macro_f1"]); best_epochs["val_macro_f1"] = epoch
            save_checkpoint(output_dir / "best_by_val_macro_f1.pt", model, spec, args, epoch, val, weights_dict)
            improved = True
        if val["loss"] < best_loss - args.min_delta:
            best_loss = float(val["loss"]); best_epochs["val_loss"] = epoch
            save_checkpoint(output_dir / "best_by_val_loss.pt", model, spec, args, epoch, val, weights_dict)
            improved = True
        no_improve = 0 if improved else no_improve + 1

        if epoch == 1 or epoch % args.log_every_epochs == 0 or epoch == args.epochs:
            print(f"[{spec.name}] epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} val_bal_acc={val['balanced_accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f} lr={lr:.6g}")

        if args.patience > 0 and no_improve >= args.patience:
            print(f"[{spec.name}] early stopping after {epoch} epochs; no improvement for {args.patience} epochs.")
            break

    save_checkpoint(output_dir / "final_model.pt", model, spec, args, history[-1]["epoch"] if history else 0, history[-1] if history else {}, weights_dict)

    def eval_ckpt(filename: str) -> Optional[Dict[str, Any]]:
        path = output_dir / filename
        if not path.exists():
            return None
        ckpt = torch.load(path, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=True)
        test = evaluate(model, test_loader, device, criterion, args.amp and torch.cuda.is_available())
        return {"checkpoint": filename, "epoch": int(ckpt.get("epoch", 0)), "val_metrics_at_checkpoint": ckpt.get("metrics", {}), "test_metrics": test}

    test_by_ckpt = {
        "best_by_val_balanced_accuracy": eval_ckpt("best_by_val_balanced_accuracy.pt"),
        "best_by_val_macro_f1": eval_ckpt("best_by_val_macro_f1.pt"),
        "best_by_val_loss": eval_ckpt("best_by_val_loss.pt"),
    }
    primary = test_by_ckpt["best_by_val_balanced_accuracy"]
    primary_test = primary["test_metrics"] if primary is not None else {}

    result = {
        "status": "completed",
        "experiment": spec.__dict__,
        "class_weights": weights_dict,
        "best_epochs": best_epochs,
        "test_by_checkpoint": test_by_ckpt,
        "test_metrics_primary_best_bal_acc": primary_test,
        "runtime_seconds": float(time.time() - start),
        "runtime_human": human_seconds(time.time() - start),
        "split_summary": split_summary,
        "epochs_run": len(history),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    plot_training_curves(history, output_dir / "training_curves.png")
    if primary_test and "confusion_matrix" in primary_test:
        plot_confusion_matrix(primary_test["confusion_matrix"], output_dir / "confusion_matrix_best_bal_acc.png", f"{spec.name}: test confusion matrix")

    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def make_specs(args: argparse.Namespace) -> List[ExperimentSpec]:
    specs = []
    for image_size in args.image_sizes:
        for backbone in args.backbones:
            name = f"{slugify(backbone)}_img{image_size}_minority_inclusive_50k_weighted_ce"
            specs.append(ExperimentSpec(name=name, backbone=backbone, image_size=int(image_size)))
    return specs




def _namespace_to_jsonable(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _namespace_from_jsonable(d: Dict[str, Any]) -> argparse.Namespace:
    path_keys = {"mg_dir", "output_dir"}
    out = {}
    for k, v in d.items():
        out[k] = Path(v) if k in path_keys and v is not None else v
    return argparse.Namespace(**out)


def _worker_run_one_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one experiment on one assigned GPU.

    The parent process creates the shared split/subset CSV files and owns summary aggregation.
    Each worker reopens the BIN memmap independently and writes only its own experiment directory.
    """
    spec = ExperimentSpec(**payload["spec"])
    args = _namespace_from_jsonable(payload["args"])
    gpu_id = payload.get("gpu_id")
    bin_spec = BinSpec(**payload["bin_spec"])
    n_raw_rows = int(payload["n_raw_rows"])
    bin_path = Path(payload["bin_path"])
    train_csv = Path(payload["train_csv"])
    val_csv = Path(payload["val_csv"])
    test_csv = Path(payload["test_csv"])

    exp_dir = args.output_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "experiment.log"

    try:
        with log_path.open("a", buffering=1, encoding="utf-8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
            print("=" * 90)
            print(f"Experiment: {spec.name}")
            print(f"Assigned GPU: {gpu_id}")
            print(f"Started at UTC: {datetime.now(timezone.utc).isoformat()}")
            print("=" * 90)

            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)
            mmap = open_memmap(bin_path, bin_spec, n_raw_rows)

            if torch.cuda.is_available():
                if gpu_id is None:
                    device = torch.device("cuda")
                else:
                    torch.cuda.set_device(0)
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

            print("device:", device)
            result = train_one_experiment(
                spec=spec,
                args=args,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                mmap=mmap,
                device=device,
                output_dir=exp_dir,
            )
            result["assigned_gpu"] = gpu_id
            result["log_path"] = str(log_path)
            print(f"Finished at UTC: {datetime.now(timezone.utc).isoformat()}")
            return result
    except Exception as exc:
        tb = traceback.format_exc()
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write("\n[FAILED]\n")
            log_f.write(tb)
            log_f.write("\n")
        result = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "assigned_gpu": gpu_id,
            "log_path": str(log_path),
            "experiment": spec.__dict__,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result


def _gpu_worker_loop(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Process experiments sequentially on one GPU.

    This avoids the common scheduling bug where two experiments are accidentally assigned to the
    same GPU while another GPU is idle.
    """
    while True:
        payload = task_queue.get()
        if payload is None:
            break
        payload["gpu_id"] = gpu_id
        result = _worker_run_one_experiment(payload)
        result_queue.put(result)



def run_experiments_parallel(
    specs: List[ExperimentSpec],
    args: argparse.Namespace,
    bin_path: Path,
    bin_spec: BinSpec,
    n_raw_rows: int,
    sampled_train_path: Path,
    val_pool_path: Path,
    test_pool_path: Path,
) -> List[Dict[str, Any]]:
    gpu_ids = [int(x) for x in (args.parallel_gpus or [])]
    if len(gpu_ids) == 0:
        raise ValueError("parallel_gpus must contain at least one GPU id")

    args_for_workers = argparse.Namespace(**vars(args))
    args_for_workers.data_parallel = False

    base_payloads: List[Dict[str, Any]] = []
    for spec in specs:
        base_payloads.append(
            {
                "spec": spec.__dict__,
                "args": _namespace_to_jsonable(args_for_workers),
                "gpu_id": None,  # filled by the per-GPU worker
                "bin_path": str(bin_path),
                "bin_spec": bin_spec.__dict__,
                "n_raw_rows": int(n_raw_rows),
                "train_csv": str(sampled_train_path),
                "val_csv": str(val_pool_path),
                "test_csv": str(test_pool_path),
            }
        )

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    for payload in base_payloads:
        task_queue.put(payload)
    for _ in gpu_ids:
        task_queue.put(None)

    print(f"Launching {len(base_payloads)} experiments across GPUs {gpu_ids}.")
    print("Each GPU runs at most one experiment at a time; when it finishes, it takes the next queued experiment.")

    workers: List[mp.Process] = []
    for gpu_id in gpu_ids:
        proc = ctx.Process(target=_gpu_worker_loop, args=(gpu_id, task_queue, result_queue), daemon=False)
        proc.start()
        workers.append(proc)

    results: List[Dict[str, Any]] = []
    try:
        for _ in range(len(base_payloads)):
            result = result_queue.get()
            exp = result.get("experiment", {})
            print(
                f"[DONE] {exp.get('name')} status={result.get('status')} "
                f"gpu={result.get('assigned_gpu')} log={result.get('log_path')}"
            )
            results.append(result)
            create_summary_outputs(args.output_dir, results)
    finally:
        for proc in workers:
            proc.join(timeout=5)
        for proc in workers:
            if proc.is_alive():
                proc.terminate()

    order = {spec.name: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order.get((r.get("experiment", {}) or {}).get("name", ""), 10**9))
    return results


def create_summary_outputs(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    rows = []
    for result in results:
        exp = result.get("experiment", {})
        primary = result.get("test_metrics_primary_best_bal_acc", {}) or {}
        split = result.get("split_summary", {}) or {}
        best = result.get("best_epochs", {}) or {}
        row = {
            "status": result.get("status"),
            "experiment": exp.get("name"),
            "backbone": exp.get("backbone"),
            "image_size": exp.get("image_size"),
            "epochs_run": result.get("epochs_run"),
            "best_epoch_val_bal_acc": best.get("val_balanced_accuracy"),
            "best_epoch_val_macro_f1": best.get("val_macro_f1"),
            "best_epoch_val_loss": best.get("val_loss"),
            "test_accuracy": primary.get("accuracy"),
            "test_balanced_accuracy": primary.get("balanced_accuracy"),
            "test_macro_f1": primary.get("macro_f1"),
            "test_loss": primary.get("loss"),
            "runtime_human": result.get("runtime_human"),
            "reason": result.get("reason"),
        }
        per_class = primary.get("per_class_recall", {}) or {}
        for cls in CLASS_NAMES:
            row[f"test_recall_{cls}"] = per_class.get(cls)
        for split_name in ["train", "val", "test"]:
            if split_name in split:
                row[f"{split_name}_rows"] = split[split_name].get("rows")
                counts = split[split_name].get("collapsed_birads_counts", {})
                for cls in CLASS_NAMES:
                    row[f"{split_name}_{cls}"] = counts.get(cls, 0)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    pdf_path = output_dir / "summary_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * max(1, len(summary_df)) + 2)))
        ax.axis("off")
        ax.set_title("MG supervised resolution/backbone baseline", fontsize=14, pad=16)
        display_cols = ["status", "backbone", "image_size", "epochs_run", "test_balanced_accuracy", "test_macro_f1", "test_accuracy", "best_epoch_val_bal_acc"]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
        for col in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        table = ax.table(cellText=display_df.fillna("").values, colLabels=display_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.25)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        completed = summary_df[summary_df["status"] == "completed"].copy()
        if not completed.empty:
            labels = [f"{b}\n{int(s)}" for b, s in zip(completed["backbone"].astype(str), completed["image_size"].astype(int))]
            for metric, title in [("test_balanced_accuracy", "Test balanced accuracy"), ("test_macro_f1", "Test macro F1")]:
                fig, ax = plt.subplots(figsize=(11, 5))
                x = np.arange(len(completed))
                ax.bar(x, completed[metric].astype(float).values)
                ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel(title); ax.set_title(f"{title} by backbone and image size"); ax.grid(axis="y", alpha=0.25)
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for result in results:
            if result.get("status") != "completed":
                continue
            exp_name = result.get("experiment", {}).get("name", "unknown")
            cm = (result.get("test_metrics_primary_best_bal_acc", {}) or {}).get("confusion_matrix")
            if cm is None:
                continue
            arr = np.array(cm, dtype=int)
            fig, ax = plt.subplots(figsize=(5.6, 5.0))
            im = ax.imshow(arr)
            ax.set_title(f"{exp_name}\nTest confusion matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, str(arr[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote summary CSV:  {output_dir / 'summary.csv'}")
    print(f"Wrote summary JSON: {output_dir / 'summary.json'}")
    print(f"Wrote summary PDF:  {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised MG resolution/backbone baselines.")
    parser.add_argument("--mg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-name", type=str, default="mg-only-all.csv")
    parser.add_argument("--bin-name", type=str, default="mg-only-all.bin")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[224, 384, 512])
    parser.add_argument("--backbones", type=str, nargs="+", default=["vit_small_patch16_224", "resnet18"])
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--include-all-classes", type=str, nargs="+", default=["follow_up", "biopsy"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--parallel-gpus", type=int, nargs="+", default=None, help="Run independent experiments in parallel on these GPU ids, e.g. --parallel-gpus 0 1 2 3. Do not combine with --data-parallel.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.image_sizes = [224]
        args.backbones = [args.backbones[0]]
        args.train_size = 600
        args.epochs = 2
        args.patience = 0
        args.batch_size = min(args.batch_size, 64)
        args.num_workers = min(args.num_workers, 2)
        args.output_dir = args.output_dir / "smoke_test"

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, bin_path = args.mg_dir / args.csv_name, args.mg_dir / args.bin_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"BIN not found: {bin_path}")

    print("Loading CSV...")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded rows: {len(df_raw):,}")
    df = enrich_metadata(df_raw)
    print(f"Rows after keeping valid target classes: {len(df):,}")
    print("collapsed_birads counts:", df["collapsed_birads"].value_counts().to_dict())
    print("machine_family counts:", df["machine_family"].value_counts().to_dict())
    print("view counts:", df["view"].value_counts().to_dict())

    bin_spec = infer_bin_spec(bin_path, len(df_raw))
    print("BIN spec:", bin_spec)
    if not bin_spec.exact_match:
        print("[WARN] BIN size did not exactly match known candidates; using fallback 512x512 uint16.")
    mmap = open_memmap(bin_path, bin_spec, len(df_raw))

    print("Creating fixed patient-disjoint train/val/test pools...")
    train_pool, val_pool, test_pool, split_info = make_patient_disjoint_pools(df, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    print("split info:", split_info)
    print("train pool:", summarize_df(train_pool))
    print("val pool:", summarize_df(val_pool))
    print("test pool:", summarize_df(test_pool))

    sampled_train, sampling_info = make_minority_inclusive_train_subset(train_pool, args.train_size, args.seed, args.include_all_classes)
    print("sampled train:", summarize_df(sampled_train))
    print("sampling info:", sampling_info)

    sampled_train_path = args.output_dir / f"sampled_train_{args.train_size}.csv"
    val_pool_path = args.output_dir / "val_pool.csv"
    test_pool_path = args.output_dir / "test_pool.csv"
    sampled_train.to_csv(sampled_train_path, index=False)
    val_pool.to_csv(val_pool_path, index=False)
    test_pool.to_csv(test_pool_path, index=False)
    global_split_summary = {
        "split_info": split_info,
        "sampling_info": sampling_info,
        "train_pool": summarize_df(train_pool),
        "sampled_train": summarize_df(sampled_train),
        "val_pool": summarize_df(val_pool),
        "test_pool": summarize_df(test_pool),
        "paths": {"sampled_train": str(sampled_train_path), "val_pool": str(val_pool_path), "test_pool": str(test_pool_path)},
    }
    (args.output_dir / "global_split_summary.json").write_text(json.dumps(global_split_summary, indent=2, sort_keys=True), encoding="utf-8")

    specs = make_specs(args)

    if args.parallel_gpus is not None and len(args.parallel_gpus) > 0:
        if args.data_parallel:
            raise ValueError("Do not combine --parallel-gpus with --data-parallel. Use one process per GPU instead.")
        results = run_experiments_parallel(
            specs=specs,
            args=args,
            bin_path=bin_path,
            bin_spec=bin_spec,
            n_raw_rows=len(df_raw),
            sampled_train_path=sampled_train_path,
            val_pool_path=val_pool_path,
            test_pool_path=test_pool_path,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        if torch.cuda.is_available():
            print("visible CUDA devices:", torch.cuda.device_count())

        results: List[Dict[str, Any]] = []
        for spec in specs:
            print("\n" + "=" * 90)
            print(f"Experiment: {spec.name}")
            print("=" * 90)
            exp_dir = args.output_dir / spec.name
            try:
                result = train_one_experiment(spec, args, sampled_train, val_pool, test_pool, mmap, device, exp_dir)
                results.append(result)
            except Exception as exc:
                print(f"[FAILED] {spec.name}: {type(exc).__name__}: {exc}")
                exp_dir.mkdir(parents=True, exist_ok=True)
                result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "experiment": spec.__dict__}
                (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                results.append(result)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            create_summary_outputs(args.output_dir, results)

    create_summary_outputs(args.output_dir, results)
    print("\nAll experiments finished or failed/skipped.")


if __name__ == "__main__":
    main()
