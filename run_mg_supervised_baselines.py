#!/usr/bin/env python3
"""
Run supervised MG BI-RADS baseline experiments in one script.

Creates supervised ViT baseline runs for:
  train sizes: 2k, 10k, 50k
  dataset settings:
    1) full MG
    2) machine_family == Hologic/Lorad
    3) machine_family == Hologic/Lorad and view == CC by default

For each dataset setting, the script first creates one fixed patient-disjoint
train/val/test pool. Then it samples exactly N balanced collapsed_birads
training examples from the train pool. Validation/test pools stay fixed across
2k/10k/50k for the same dataset setting.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    raise SystemExit("ERROR: timm is not installed. Install with: python -m pip install timm") from exc

try:
    import cv2
except ImportError as exc:
    raise SystemExit("ERROR: opencv-python-headless is not installed. Install with: python -m pip install opencv-python-headless") from exc


CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class ExperimentSpec:
    name: str
    setting_name: str
    train_size: int
    machine_family: Optional[str]
    view: Optional[str]


@dataclass
class BinSpec:
    dtype: str
    height: int
    width: int
    channels: int
    exact_match: bool


def set_global_seed(seed: int) -> None:
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
    explicit_cols = ["view", "ViewPosition", "view_position", "viewposition", "projection", "position"]
    text_cols = ["id", "context", "findings", "image_path", "path", "filename", "file", "dicom_path", "png_path", "jpg_path", "original_path"]
    for col in explicit_cols + text_cols:
        if col in row.index:
            value = normalize_view(row[col])
            if value is not None:
                return value
    return "unknown"


def clean_column_name(name: object) -> str:
    text = str(name).strip().lstrip("\ufeff")
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {c: clean_column_name(c) for c in out.columns}
    out = out.rename(columns=rename)

    # Common aliases. Only rename when this does not overwrite an existing column.
    aliases = {
        "birads_num": "birads_numeric",
        "birads_number": "birads_numeric",
        "birads_score": "birads_numeric",
        "original_birads_numeric": "birads_numeric",
        "original_birads_score": "original_birads",
        "actionability": "collapsed_birads",
        "action": "collapsed_birads",
        "label": "collapsed_birads",
        "view_position": "view",
        "viewposition": "view",
        "manufacturer_model_name": "machine",
    }
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    return out


def parse_birads_number(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "missing", "unknown"}:
        return None

    # Already numeric-like.
    try:
        f = float(text)
        if math.isfinite(f):
            return int(f)
    except Exception:
        pass

    # Strings such as "BI-RADS 4A", "BIRADS_3", "category 2".
    m = re.search(r"([0-6])", text)
    if m:
        return int(m.group(1))

    return None


def collapse_birads_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    raw = str(value).strip().lower()
    if raw in {"", "nan", "none", "missing", "unknown"}:
        return None

    # Normalize for exact label variants.
    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    if norm in CLASS_TO_INDEX:
        return norm

    # Native MG actionability strings:
    #   "healthy/routine"
    #   "probably benign (follow up)"
    #   "suspicious/malignancy-likely (biopsy)"
    #
    # Order matters: "probably benign (follow up)" contains "benign",
    # but should be follow_up, not routine.
    if "biopsy" in raw or "suspicious" in raw or "malignan" in raw:
        return "biopsy"

    if "follow" in raw or "probably benign" in raw or "probably_benign" in norm:
        return "follow_up"

    if "routine" in raw or "healthy" in raw or "negative" in raw or raw == "benign" or norm == "benign":
        return "routine"

    # Fallback for numeric BI-RADS or strings containing BI-RADS numbers.
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

def ensure_collapsed_birads(out: pd.DataFrame) -> pd.DataFrame:
    if "collapsed_birads" in out.columns:
        out["collapsed_birads"] = out["collapsed_birads"].map(collapse_birads_value)
        return out

    candidate_cols = [c for c in ["birads_numeric", "birads", "original_birads"] if c in out.columns]
    if not candidate_cols:
        raise ValueError(
            "Could not create collapsed_birads: no collapsed_birads, birads_numeric, birads, "
            f"or original_birads column found. Available columns: {list(out.columns)}"
        )

    print(f"collapsed_birads not found; deriving it from '{candidate_cols[0]}'")
    out["collapsed_birads"] = out[candidate_cols[0]].map(collapse_birads_value)
    return out


def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(df)
    out["source_index"] = np.arange(len(out), dtype=np.int64)

    if "machine_family" not in out.columns:
        out["machine_family"] = out["machine"].map(infer_machine_family) if "machine" in out.columns else "unknown"
    if "view" not in out.columns:
        out["view"] = out.apply(infer_view_from_row, axis=1)

    out = ensure_collapsed_birads(out)

    before = len(out)
    out = out[out["collapsed_birads"].isin(CLASS_NAMES)].copy()
    after = len(out)
    if after == 0:
        available = {}
        for col in ["collapsed_birads", "birads_numeric", "birads", "original_birads"]:
            if col in out.columns:
                available[col] = out[col].astype(str).value_counts().head(20).to_dict()
        raise ValueError(
            "After creating collapsed_birads, no rows matched the classes "
            f"{CLASS_NAMES}. Candidate value counts: {available}"
        )
    if after < before:
        print(f"Dropped {before - after:,} rows without a valid collapsed_birads class.")

    out["target"] = out["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
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
        expected = n_rows * h * w * c * dtype.itemsize
        if expected == actual:
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
            summary[f"{col}_counts"] = {str(k): int(v) for k, v in df[col].fillna("missing").astype(str).value_counts().to_dict().items()}
    return summary


def patient_level_label(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    def mode_or_first(s: pd.Series) -> str:
        mode = s.mode()
        return str(mode.iloc[0]) if len(mode) > 0 else str(s.iloc[0])
    return df.groupby(group_col, dropna=False).agg(n_rows=(group_col, "size"), label=("collapsed_birads", mode_or_first)).reset_index()


def make_patient_disjoint_pools(df: pd.DataFrame, seed: int, group_col: str = "patient") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if group_col not in df.columns:
        train_df, tmp = train_test_split(df, train_size=0.8, random_state=seed, shuffle=True, stratify=df["collapsed_birads"])
        val_df, test_df = train_test_split(tmp, train_size=0.5, random_state=seed + 1, shuffle=True, stratify=tmp["collapsed_birads"])
        return train_df.copy(), val_df.copy(), test_df.copy(), {"split_type": "row_level_fallback", "patient_overlap": {}}

    groups = patient_level_label(df, group_col)
    if len(groups) < 3:
        raise ValueError(f"Too few patients/groups for train/val/test split: {len(groups)}")
    try:
        train_groups, tmp_groups = train_test_split(groups, train_size=0.8, random_state=seed, shuffle=True, stratify=groups["label"])
    except ValueError:
        train_groups, tmp_groups = train_test_split(groups, train_size=0.8, random_state=seed, shuffle=True, stratify=None)
    try:
        val_groups, test_groups = train_test_split(tmp_groups, train_size=0.5, random_state=seed + 1, shuffle=True, stratify=tmp_groups["label"])
    except ValueError:
        val_groups, test_groups = train_test_split(tmp_groups, train_size=0.5, random_state=seed + 1, shuffle=True, stratify=None)

    train_ids, val_ids, test_ids = set(train_groups[group_col]), set(val_groups[group_col]), set(test_groups[group_col])
    train_df = df[df[group_col].isin(train_ids)].copy()
    val_df = df[df[group_col].isin(val_ids)].copy()
    test_df = df[df[group_col].isin(test_ids)].copy()

    train_pat = set(train_df[group_col].dropna().astype(str)); val_pat = set(val_df[group_col].dropna().astype(str)); test_pat = set(test_df[group_col].dropna().astype(str))
    info = {"split_type": "patient_disjoint", "patient_overlap": {"train_vs_val": len(train_pat & val_pat), "train_vs_test": len(train_pat & test_pat), "val_vs_test": len(val_pat & test_pat)}}
    return train_df, val_df, test_df, info


def balanced_train_sample(train_pool: pd.DataFrame, train_size: int, seed: int) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, int]]:
    counts = train_pool["collapsed_birads"].value_counts().to_dict()
    base = train_size // len(CLASS_NAMES)
    remainder = train_size % len(CLASS_NAMES)
    quotas = {cls: base + (1 if i < remainder else 0) for i, cls in enumerate(CLASS_NAMES)}
    missing = {cls: quotas[cls] - int(counts.get(cls, 0)) for cls in CLASS_NAMES if int(counts.get(cls, 0)) < quotas[cls]}
    if missing:
        reason = f"not enough training samples for balanced size {train_size}; missing quotas: {missing}; available counts: {counts}"
        return None, reason, quotas
    parts = []
    for i, cls in enumerate(CLASS_NAMES):
        subset = train_pool[train_pool["collapsed_birads"] == cls]
        parts.append(subset.sample(n=quotas[cls], random_state=seed + i))
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed + 100).copy()
    return sampled, None, quotas


class MGMammographyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mmap: np.memmap, image_size: int, train: bool, augment: bool) -> None:
        self.df = df.reset_index(drop=True)
        self.mmap = mmap
        self.image_size = image_size
        self.train = train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        lo = np.percentile(arr, 1.0); hi = np.percentile(arr, 99.0)
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
        source_index = int(row["source_index"])
        x = self._preprocess(self.mmap[source_index])
        y = int(row["target"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    kwargs = {"pretrained": bool(args.pretrained), "num_classes": num_classes, "in_chans": 1}
    try:
        return timm.create_model(args.backbone, img_size=args.image_size, **kwargs)
    except TypeError:
        return timm.create_model(args.backbone, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, use_amp: bool) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0; n = 0; ys: List[int] = []; preds: List[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x); loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0); n += x.size(0)
        pred = torch.argmax(logits, dim=1)
        ys.extend(y.detach().cpu().numpy().tolist()); preds.extend(pred.detach().cpu().numpy().tolist())
    labels = list(range(len(CLASS_NAMES)))
    if n == 0:
        return {"loss": None, "accuracy": None, "balanced_accuracy": None, "macro_f1": None, "per_class_recall": {}, "confusion_matrix": [[0 for _ in labels] for _ in labels]}
    cm = confusion_matrix(ys, preds, labels=labels)
    per_class_recall_values = recall_score(ys, preds, labels=labels, average=None, zero_division=0)
    return {
        "loss": total_loss / n,
        "accuracy": float(accuracy_score(ys, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {CLASS_NAMES[i]: float(per_class_recall_values[i]) for i in labels},
        "confusion_matrix": cm.astype(int).tolist(),
    }


def plot_training_curves(history: List[Dict[str, Any]], out_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(epochs, [h["val_balanced_accuracy"] for h in history]); axes[0, 1].set_title("Validation balanced accuracy"); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(epochs, [h["val_macro_f1"] for h in history]); axes[1, 0].set_title("Validation macro F1"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(epochs, [h["lr"] for h in history]); axes[1, 1].set_title("Learning rate"); axes[1, 1].grid(alpha=0.3)
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


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
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def save_checkpoint(out_path: Path, model: nn.Module, args: argparse.Namespace, experiment: ExperimentSpec, epoch: int, metrics: Dict[str, Any]) -> None:
    ckpt = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "class_names": CLASS_NAMES,
        "model_config": {"backbone": args.backbone, "image_size": args.image_size, "num_classes": len(CLASS_NAMES), "in_chans": 1, "pretrained": bool(args.pretrained)},
        "experiment": experiment.__dict__,
    }
    torch.save(ckpt, out_path)


def train_one_experiment(args: argparse.Namespace, experiment: ExperimentSpec, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, mmap: np.memmap, device: torch.device, output_dir: Path) -> Dict[str, Any]:
    set_global_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "sampled_train.csv", index=False); val_df.to_csv(output_dir / "val_pool.csv", index=False); test_df.to_csv(output_dir / "test_pool.csv", index=False)
    split_summary = {"train": summarize_df(train_df), "val": summarize_df(val_df), "test": summarize_df(test_df)}
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2, sort_keys=True), encoding="utf-8")
    config = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "experiment": experiment.__dict__, "split_summary": split_summary, "class_names": CLASS_NAMES}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    train_loader = DataLoader(MGMammographyDataset(train_df, mmap, args.image_size, True, not args.no_augment), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(MGMammographyDataset(val_df, mmap, args.image_size, False, False), batch_size=args.eval_batch_size or args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(MGMammographyDataset(test_df, mmap, args.image_size, False, False), batch_size=args.eval_batch_size or args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=args.num_workers > 0)

    model = make_model(args, num_classes=len(CLASS_NAMES)).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using torch.nn.DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    history: List[Dict[str, Any]] = []
    best_val_bal_acc = -1.0; best_epoch = 0; best_val_metrics: Dict[str, Any] = {}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss_sum = 0.0; n_train = 0
        pbar = tqdm(train_loader, desc=f"{experiment.name} epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss_sum += float(loss.item()) * x.size(0); n_train += x.size(0)
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
        scheduler.step()
        train_loss = train_loss_sum / max(1, n_train)
        val_metrics = evaluate(model, val_loader, device=device, criterion=criterion, use_amp=args.amp and torch.cuda.is_available())
        row = {"epoch": epoch, "train_loss": float(train_loss), "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"], "val_macro_f1": val_metrics["macro_f1"], "lr": float(optimizer.param_groups[0]["lr"]), "elapsed_seconds": float(time.time() - start_time)}
        history.append(row)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        val_bal_acc = val_metrics["balanced_accuracy"] if val_metrics["balanced_accuracy"] is not None else -1.0
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = float(val_bal_acc); best_epoch = epoch; best_val_metrics = val_metrics
            save_checkpoint(output_dir / "best_model.pt", model, args, experiment, epoch, val_metrics)
        if epoch == 1 or epoch % args.log_every_epochs == 0 or epoch == args.epochs:
            print(f"[{experiment.name}] epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

    save_checkpoint(output_dir / "final_model.pt", model, args, experiment, args.epochs, history[-1] if history else {})
    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    unwrap_model(model).load_state_dict(best_ckpt["model_state_dict"], strict=True)
    test_metrics = evaluate(model, test_loader, device=device, criterion=criterion, use_amp=args.amp and torch.cuda.is_available())
    metrics = {"status": "completed", "experiment": experiment.__dict__, "best_epoch": int(best_epoch), "best_val_metrics": best_val_metrics, "test_metrics_best_model": test_metrics, "runtime_seconds": float(time.time() - start_time), "runtime_human": human_seconds(time.time() - start_time), "split_summary": split_summary}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], output_dir / "confusion_matrix_best.png", f"{experiment.name}: test confusion matrix")
    return metrics


def make_experiment_specs(args: argparse.Namespace) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    settings = [("full", None, None), ("hologic_lorad", args.machine_family, None), (f"hologic_lorad_{slugify(args.single_view)}", args.machine_family, args.single_view)]
    for size in args.train_sizes:
        for setting_name, machine_family, view in settings:
            specs.append(ExperimentSpec(name=f"{setting_name}_balanced_{size}", setting_name=setting_name, train_size=int(size), machine_family=machine_family, view=view))
    return specs


def filter_setting(df: pd.DataFrame, machine_family: Optional[str], view: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if machine_family is not None:
        out = out[out["machine_family"].astype(str).str.strip().str.casefold() == machine_family.strip().casefold()].copy()
    if view is not None:
        out = out[out["view"].astype(str).str.strip().str.casefold() == view.strip().casefold()].copy()
    return out


def create_summary_outputs(output_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in all_results:
        exp = result.get("experiment", {}); test = result.get("test_metrics_best_model", {}); split = result.get("split_summary", {})
        row = {"status": result.get("status"), "experiment": exp.get("name"), "setting": exp.get("setting_name"), "train_size_requested": exp.get("train_size"), "machine_family": exp.get("machine_family"), "view": exp.get("view"), "reason": result.get("reason"), "best_epoch": result.get("best_epoch"), "test_accuracy": test.get("accuracy"), "test_balanced_accuracy": test.get("balanced_accuracy"), "test_macro_f1": test.get("macro_f1"), "runtime_human": result.get("runtime_human")}
        for split_name in ["train", "val", "test"]:
            if split_name in split:
                row[f"{split_name}_rows"] = split[split_name].get("rows")
                row[f"{split_name}_patients"] = split[split_name].get("patients")
                counts = split[split_name].get("collapsed_birads_counts", {})
                for cls in CLASS_NAMES:
                    row[f"{split_name}_{cls}"] = counts.get(cls, 0)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")

    pdf_path = output_dir / "summary_report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * len(summary_df) + 2))); ax.axis("off")
        ax.set_title("MG supervised BI-RADS baselines: summary", fontsize=14, pad=16)
        display_cols = ["status", "experiment", "train_size_requested", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "best_epoch", "reason"]
        display_df = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
        for col in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        table = ax.table(cellText=display_df.fillna("").values, colLabels=display_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.25)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        completed = summary_df[summary_df["status"] == "completed"].copy()
        if not completed.empty:
            fig, ax = plt.subplots(figsize=(13, 5)); x = np.arange(len(completed))
            ax.bar(x, completed["test_balanced_accuracy"].astype(float).values)
            ax.set_xticks(x); ax.set_xticklabels(completed["experiment"].astype(str).tolist(), rotation=45, ha="right")
            ax.set_ylabel("Test balanced accuracy"); ax.set_ylim(0.0, 1.0); ax.set_title("Test balanced accuracy by experiment"); ax.grid(axis="y", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        for result in all_results:
            if result.get("status") != "completed":
                continue
            exp_name = result.get("experiment", {}).get("name", "unknown"); cm = result.get("test_metrics_best_model", {}).get("confusion_matrix")
            if cm is None:
                continue
            arr = np.array(cm, dtype=int); fig, ax = plt.subplots(figsize=(5.6, 5.0)); im = ax.imshow(arr)
            ax.set_title(f"{exp_name}\nTest confusion matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, str(arr[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote aggregate summary: {output_dir / 'summary.csv'}")
    print(f"Wrote aggregate JSON:    {output_dir / 'summary.json'}")
    print(f"Wrote summary PDF:       {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised ViT BI-RADS baseline experiments on MG data.")
    parser.add_argument("--mg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-name", type=str, default="mg-only-all.csv")
    parser.add_argument("--bin-name", type=str, default="mg-only-all.bin")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[2000, 10000, 50000])
    parser.add_argument("--machine-family", type=str, default="Hologic/Lorad")
    parser.add_argument("--single-view", type=str, default="CC")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-parallel", action="store_true", help="Use torch.nn.DataParallel if multiple GPUs are visible.")
    parser.add_argument("--no-amp", dest="amp", action="store_false"); parser.set_defaults(amp=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true", help="Override train sizes and epochs for a fast end-to-end check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.train_sizes = [300]; args.epochs = 2; args.batch_size = min(args.batch_size, 64); args.output_dir = args.output_dir / "smoke_test"
    set_global_seed(args.seed)
    csv_path = args.mg_dir / args.csv_name; bin_path = args.mg_dir / args.bin_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"BIN not found: {bin_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV...")
    df_raw = pd.read_csv(csv_path); print(f"Loaded rows: {len(df_raw):,}")
    df = enrich_metadata(df_raw)
    print(f"Rows after keeping valid collapsed_birads classes: {len(df):,}")
    print("collapsed_birads counts:", df["collapsed_birads"].value_counts().to_dict())
    print("machine_family counts:", df["machine_family"].value_counts().to_dict())
    print("view counts:", df["view"].value_counts().to_dict())
    bin_spec = infer_bin_spec(bin_path, len(df_raw)); print("BIN spec:", bin_spec)
    if not bin_spec.exact_match:
        print("[WARN] BIN size did not exactly match known candidates; using fallback 512x512 uint16.")
    mmap = open_memmap(bin_path, bin_spec, len(df_raw))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", device)
    if torch.cuda.is_available():
        print("Visible CUDA devices:", torch.cuda.device_count())

    specs = make_experiment_specs(args)
    setting_specs = [("full", None, None), ("hologic_lorad", args.machine_family, None), (f"hologic_lorad_{slugify(args.single_view)}", args.machine_family, args.single_view)]
    setting_pools: Dict[str, Dict[str, Any]] = {}
    for setting_name, machine_family, view in setting_specs:
        setting_df = filter_setting(df, machine_family, view)
        print(f"\nSetting {setting_name}: {len(setting_df):,} rows")
        print("  collapsed_birads:", setting_df["collapsed_birads"].value_counts().to_dict())
        if len(setting_df) == 0:
            setting_pools[setting_name] = {"status": "empty", "reason": "no rows after filter"}; continue
        try:
            train_pool, val_pool, test_pool, split_info = make_patient_disjoint_pools(setting_df, seed=args.seed, group_col="patient")
            setting_pools[setting_name] = {"status": "ok", "train_pool": train_pool, "val_pool": val_pool, "test_pool": test_pool, "split_info": split_info}
            print("  train pool:", summarize_df(train_pool)); print("  val pool:  ", summarize_df(val_pool)); print("  test pool: ", summarize_df(test_pool)); print("  split info:", split_info)
        except Exception as exc:
            setting_pools[setting_name] = {"status": "failed", "reason": str(exc)}; print(f"  [SKIP SETTING] {exc}")

    all_results: List[Dict[str, Any]] = []
    for spec in specs:
        print("\n" + "=" * 80); print(f"Experiment: {spec.name}"); print("=" * 80)
        pools = setting_pools.get(spec.setting_name); exp_dir = args.output_dir / spec.name; exp_dir.mkdir(parents=True, exist_ok=True)
        if pools is None or pools.get("status") != "ok":
            reason = pools.get("reason", "setting pool not available") if pools else "setting pool not available"
            print(f"[SKIP] {reason}"); result = {"status": "skipped", "reason": reason, "experiment": spec.__dict__}
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"); all_results.append(result); create_summary_outputs(args.output_dir, all_results); continue
        train_pool, val_pool, test_pool = pools["train_pool"], pools["val_pool"], pools["test_pool"]
        sampled_train, reason, quotas = balanced_train_sample(train_pool, train_size=spec.train_size, seed=args.seed + spec.train_size)
        if sampled_train is None:
            print(f"[SKIP] {reason}")
            result = {"status": "skipped", "reason": reason, "experiment": spec.__dict__, "requested_quotas": quotas, "train_pool_summary": summarize_df(train_pool), "val_pool_summary": summarize_df(val_pool), "test_pool_summary": summarize_df(test_pool)}
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"); all_results.append(result); create_summary_outputs(args.output_dir, all_results); continue
        print("Sampled train:", summarize_df(sampled_train)); print("Val pool:", summarize_df(val_pool)); print("Test pool:", summarize_df(test_pool))
        try:
            metrics = train_one_experiment(args, spec, sampled_train, val_pool, test_pool, mmap, device, exp_dir); all_results.append(metrics)
        except Exception as exc:
            print(f"[FAILED] {spec.name}: {type(exc).__name__}: {exc}")
            result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "experiment": spec.__dict__}
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"); all_results.append(result)
        create_summary_outputs(args.output_dir, all_results)
    create_summary_outputs(args.output_dir, all_results)
    print("\nAll experiments finished or skipped.")


if __name__ == "__main__":
    main()
