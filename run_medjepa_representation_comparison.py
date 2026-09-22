#!/usr/bin/env python3
"""
Compare three frozen representations from a trained MedJEPA/LeJEPA checkpoint:

1. head512: the current 512-dimensional timm-head embedding used by the training script.
2. cls: the raw final ViT CLS token, before the timm classification/head layer.
3. patch_cross_attention: final ViT patch tokens collapsed with a small supervised
   cross-attention pooling head trained on collapsed BI-RADS.

For each representation, the script evaluates collapsed BI-RADS classification with:
- a linear probe (main metric: balanced accuracy)
- a small MLP classifier (main metric: balanced accuracy)

It also creates a single PDF report with:
- summary and metric comparison tables
- PCA views for head512, cls, and patch_cross_attention colored by
  collapsed_birads, view, machine_family, and dataset.

Important notes:
- The JEPA encoder is always frozen.
- The head512 and cls probes train only small classifiers on precomputed features.
- The patch_cross_attention experiment trains a small supervised pooling/classifier
  head on top of frozen patch tokens; its PCA uses the pooled representation from
  the best patch MLP model.
"""

from __future__ import annotations

from core.config import deep_get, save_json
from core.data import (
    MammographyDataset,
    prepare_labels,
    read_csv_clean,
    set_seed,
    _normalize_view,
    _normalize_laterality,
)
from core.models import ViTEncoder, model_hints, read_checkpoint, strip_module_prefix
from core.transforms import ConfigurableMGAugmentation

import argparse
import copy
import datetime as _dt
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Config / CLI
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    image_size: int = 384
    image_height: int = 512
    image_width: int = 512
    memmap_dtype: str = "uint16"
    normalize_mode: str = "uint16"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    backbone_name: str = "vit_small_patch8_224"
    backbone_num_classes: int = 512
    backbone_output_dim: int = 512
    projection_dim: int = 16
    projector_hidden_dim: int = 2048
    drop_path_rate: float = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen MedJEPA representation comparison report.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint_epoch_XXXX.pt or final_lejepa_checkpoint.pt")
    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--bin", required=True, type=str)
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--val-csv", required=True, type=str)
    p.add_argument("--test-csv", required=True, type=str)
    p.add_argument("--output-pdf", required=True, type=str)
    p.add_argument("--output-json", default=None, type=str)

    p.add_argument("--batch-size", type=int, default=256, help="Batch size for frozen feature extraction.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--probe-epochs", type=int, default=50, help="Epochs for head512/CLS linear and MLP probes.")
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-learning-rate", type=float, default=1e-3)
    p.add_argument("--probe-weight-decay", type=float, default=1e-7)
    p.add_argument("--probe-train-max-samples", type=int, default=60000, help="Balanced train subset size for head512/CLS probes. 0 = full train.")

    p.add_argument("--mlp-hidden-dim", type=int, default=256)
    p.add_argument("--mlp-dropout", type=float, default=0.1)

    p.add_argument("--patch-epochs", type=int, default=20, help="Epochs for patch cross-attention downstream heads.")
    p.add_argument("--patch-train-max-samples", type=int, default=60000, help="Balanced train subset size for patch heads. 0 = full train.")
    p.add_argument("--patch-batch-size", type=int, default=128)
    p.add_argument("--patch-learning-rate", type=float, default=1e-3)
    p.add_argument("--patch-weight-decay", type=float, default=1e-5)
    p.add_argument("--patch-num-queries", type=int, default=1)
    p.add_argument("--patch-attn-heads", type=int, default=4)
    p.add_argument("--skip-patch", action="store_true", help="Only run head512 and cls experiments.")

    p.add_argument("--pca-max-samples", type=int, default=1998)
    p.add_argument("--pca-sampling", choices=["balanced_collapsed", "random"], default="balanced_collapsed")
    p.add_argument("--pca-split", choices=["train", "val", "test"], default="test")

    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use BF16 autocast for frozen encoder inference on CUDA. Enabled by default on CUDA unless --no-amp.")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Checkpoint/model
# -----------------------------------------------------------------------------


class MedJEPAEncoder(ViTEncoder):
    """Compatible with train_medjepa_mg_v5/v6 checkpoints."""

    def forward_head512(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,1,H,W]
        return self.backbone(x)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw CLS token and patch tokens from the ViT before the head.

        For ViT patch16/img224 this should be:
          cls: [B, 384]
          patch_tokens: [B, 196, 384]
        """
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            # Some timm models can return dict-like features. Prefer common keys.
            for key in ("x", "tokens", "features"):
                if key in out:
                    out = out[key]
                    break
        if out.ndim != 3:
            raise RuntimeError(
                f"Expected backbone.forward_features(x) to return token sequence [B,N,D], got shape {tuple(out.shape)}. "
                "This script currently expects a ViT-like timm model."
            )
        cls = out[:, 0]
        patches = out[:, 1:]
        return cls, patches


def load_checkpoint_and_model(
    path: str | Path, device: torch.device
) -> tuple[MedJEPAEncoder, dict[str, Any], dict[str, Any], ModelConfig]:
    ckpt = read_checkpoint(path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict containing model_state_dict.")
    raw_cfg = ckpt.get("config", {}) or {}
    aug_cfg = ckpt.get("augmentation_config", {}) or {}
    image_size = int(raw_cfg.get("image_size", 0) or 0)
    if image_size <= 0:
        image_size = int(deep_get(aug_cfg, ["image", "output_size"], 384))
    cfg = ModelConfig(
        image_size=image_size,
        image_height=int(raw_cfg.get("image_height", 512)),
        image_width=int(raw_cfg.get("image_width", 512)),
        memmap_dtype=str(raw_cfg.get("memmap_dtype", "uint16")),
        normalize_mode=str(raw_cfg.get("normalize_mode", "uint16")),
        percentile_low=float(raw_cfg.get("percentile_low", 1.0)),
        percentile_high=float(raw_cfg.get("percentile_high", 99.0)),
        backbone_name=str(raw_cfg.get("backbone_name", "vit_small_patch8_224")),
        backbone_output_dim=int(raw_cfg.get("backbone_output_dim", 512)),
        projection_dim=int(raw_cfg.get("projection_dim", 16)),
        projector_hidden_dim=int(raw_cfg.get("projector_hidden_dim", 2048)),
        drop_path_rate=float(raw_cfg.get("drop_path_rate", 0.1)),
    )
    # Infer the backbone head dimensions from weights, including headless runs.
    hints = model_hints(ckpt)
    for name in ("backbone_num_classes", "backbone_output_dim", "image_size"):
        if name in hints:
            setattr(cfg, name, int(hints[name]))
    model = MedJEPAEncoder(cfg)
    state = {strip_module_prefix(key): value for key, value in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, raw_cfg, aug_cfg, cfg


# -----------------------------------------------------------------------------
# CSV / metadata
# -----------------------------------------------------------------------------


def _safe_json_loads(x: Any) -> dict[str, Any]:
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "missing"}:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _machine_family(x: Any) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    lo = s.lower()
    if not s or lo in {"nan", "none", "unknown", "missing"}:
        return "unknown"
    if "hologic" in lo or "lorad" in lo or "selenia" in lo:
        return "Hologic/Lorad"
    if "ge" in lo or "senograph" in lo or "senographe" in lo:
        return "GE/Senographe"
    if "howtek" in lo or "lumysis" in lo:
        return "Howtek/Lumysis"
    return "Other"


def add_derived_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "context" in df.columns:
        parsed = df["context"].apply(_safe_json_loads)
        if "view" not in df.columns:
            df["view"] = parsed.apply(lambda d: _normalize_view(deep_get(d, ["exam", "view"], "unknown")))
        else:
            df["view"] = df["view"].apply(_normalize_view)
        if "laterality" not in df.columns:
            df["laterality"] = parsed.apply(
                lambda d: _normalize_laterality(deep_get(d, ["exam", "laterality"], "unknown"))
            )
        else:
            df["laterality"] = df["laterality"].apply(_normalize_laterality)
    else:
        df["view"] = df.get("view", "unknown")
        df["view"] = df["view"].apply(_normalize_view) if isinstance(df["view"], pd.Series) else "unknown"
        df["laterality"] = df.get("laterality", "unknown")
        df["laterality"] = (
            df["laterality"].apply(_normalize_laterality) if isinstance(df["laterality"], pd.Series) else "unknown"
        )

    if "machine" in df.columns:
        df["machine_family"] = df["machine"].apply(_machine_family)
    else:
        df["machine_family"] = "unknown"
    if "dataset" not in df.columns:
        df["dataset"] = "unknown"
    return df


def ensure_original_index(split_df: pd.DataFrame, full_df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_df = split_df.copy()
    if "original_index" in split_df.columns:
        split_df["original_index"] = split_df["original_index"].astype(int)
        return split_df

    full = full_df_raw.copy()
    full["original_index"] = np.arange(len(full))

    # Prefer a rich composite key because full MG CSV may contain duplicated ids.
    preferred_cols = [
        "id",
        "patient",
        "dataset",
        "modality",
        "machine",
        "exam",
        "segmentation",
        "context",
        "findings",
        "original_birads",
        "birads",
    ]
    cols = [c for c in preferred_cols if c in split_df.columns and c in full.columns]
    if not cols:
        raise ValueError(f"{split_name} split needs original_index or shared metadata columns for memmap lookup.")

    # Try longest key first; if it is unique in full, use it.
    for k in range(len(cols), 0, -1):
        use_cols = cols[:k]
        full_key = full[use_cols + ["original_index"]].copy()
        if full_key.duplicated(use_cols).any():
            continue
        merged = split_df.merge(full_key, on=use_cols, how="left", validate="many_to_one")
        missing = int(merged["original_index"].isna().sum())
        if missing == 0:
            merged["original_index"] = merged["original_index"].astype(int)
            print(f"Mapped {split_name} split to memmap rows using composite key: {use_cols}", flush=True)
            return merged

    # Last fallback: id-only if it is unique enough.
    if "id" in split_df.columns and "id" in full.columns:
        if not full["id"].astype(str).duplicated().any():
            id_to_idx = pd.Series(full["original_index"].to_numpy(), index=full["id"].astype(str)).to_dict()
            split_df["original_index"] = split_df["id"].astype(str).map(id_to_idx)
            missing = int(split_df["original_index"].isna().sum())
            if missing == 0:
                split_df["original_index"] = split_df["original_index"].astype(int)
                return split_df

    raise ValueError(
        f"Could not reconstruct original_index for {split_name}. Full CSV has duplicate keys or split lacks enough metadata. "
        "Best fix: add original_index to split CSVs."
    )


def load_splits(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    full_raw = read_csv_clean(args.full_csv)
    full_num_rows = len(full_raw)
    full_df = add_derived_metadata(prepare_labels(full_raw))
    if "original_index" not in full_df.columns:
        full_df = full_df.reset_index(drop=False).rename(columns={"index": "original_index"})

    train_df = add_derived_metadata(prepare_labels(ensure_original_index(read_csv_clean(args.train_csv), full_raw, "train")))
    val_df = add_derived_metadata(prepare_labels(ensure_original_index(read_csv_clean(args.val_csv), full_raw, "val")))
    test_df = add_derived_metadata(prepare_labels(ensure_original_index(read_csv_clean(args.test_csv), full_raw, "test")))
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: rows={len(df):,}, collapsed={df['collapsed_birads'].value_counts().to_dict()}, view={df['view'].value_counts().to_dict()}", flush=True)
    return full_df, train_df, val_df, test_df, full_num_rows


def balanced_sample_df(df: pd.DataFrame, label_col: str, max_total: int, seed: int) -> pd.DataFrame:
    if max_total <= 0 or len(df) <= max_total:
        return df.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    labels = sorted(df[label_col].dropna().unique().tolist())
    per_class = max(1, max_total // max(1, len(labels)))
    parts = []
    for lab in labels:
        sub = df[df[label_col] == lab]
        n = min(len(sub), per_class)
        if n <= 0:
            continue
        parts.append(sub.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[:0].copy()
    if len(out) > max_total:
        out = out.sample(n=max_total, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Deterministic MG evaluation dataset/transform
# -----------------------------------------------------------------------------


class MGEvalDataset(MammographyDataset):
    """Shared image reading with the experiment's (image, label, row) batches."""

    def __init__(self, df, bin_path, full_num_rows, cfg, aug_cfg):
        super().__init__(
            df,
            bin_path,
            full_num_rows,
            (cfg.image_height, cfg.image_width),
            cfg.memmap_dtype,
            ConfigurableMGAugmentation(aug_cfg, cfg.image_size, train=False),
            cfg.normalize_mode,
            cfg.percentile_low,
            cfg.percentile_high,
        )

    def __getitem__(self, idx):
        return self.transform(self.image_at(idx)), int(self.df.iloc[idx]["target_collapsed"]), idx


def make_loader(df: pd.DataFrame, bin_path: str | Path, full_num_rows: int, cfg: ModelConfig, aug_cfg: dict[str, Any],
                batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    ds = MGEvalDataset(df, bin_path, full_num_rows, cfg, aug_cfg)
    kwargs = dict(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------


@torch.inference_mode()
def extract_features(model: MedJEPAEncoder, loader: DataLoader, device: torch.device, representation: str, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    for x, y, idx in tqdm(loader, desc=f"Extract {representation}"):
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_cuda and use_amp)):
            if representation == "head512":
                z = model.forward_head512(x)
            elif representation == "cls":
                z, _ = model.forward_tokens(x)
            else:
                raise ValueError(f"Unsupported feature representation: {representation}")
        feats.append(z.float().cpu())
        labels.append(y.cpu())
        indices.append(idx.cpu())
    return torch.cat(feats), torch.cat(labels), torch.cat(indices)


@torch.inference_mode()
def extract_patch_pooled(model: MedJEPAEncoder, downstream: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    downstream.eval()
    feats, labels, indices = [], [], []
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    for x, y, idx in tqdm(loader, desc="Extract patch-cross-attention pooled"):
        x = x.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_cuda and use_amp)):
            _, patches = model.forward_tokens(x)
        pooled = downstream.pool(patches.float())
        feats.append(pooled.float().cpu())
        labels.append(y.cpu())
        indices.append(idx.cpu())
    return torch.cat(feats), torch.cat(labels), torch.cat(indices)


# -----------------------------------------------------------------------------
# Probes on precomputed features
# -----------------------------------------------------------------------------


class TabularProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int = 3, kind: str = "linear", hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights_from_labels(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    w = counts.sum() / counts.clamp_min(1.0)
    return w / w.mean().clamp_min(1e-12)


def compute_metrics_from_pred(true: np.ndarray, pred: np.ndarray, prefix: str = "") -> dict[str, Any]:
    names = ["routine", "follow_up", "biopsy"]
    return {
        f"{prefix}accuracy": float(accuracy_score(true, pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        f"{prefix}macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        f"{prefix}weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        f"{prefix}confusion_matrix": confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
        f"{prefix}classification_report": classification_report(true, pred, labels=[0, 1, 2], target_names=names, zero_division=0, output_dict=True),
    }


def eval_tabular_probe(probe: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, Any]:
    probe.eval()
    preds = []
    bs = 4096
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            logits = probe(x[i:i+bs].to(device))
            preds.append(logits.argmax(dim=1).cpu())
    pred = torch.cat(preds).numpy()
    true = y.cpu().numpy()
    return compute_metrics_from_pred(true, pred)


def train_tabular_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    dim = int(x_train.shape[1])
    probe = TabularProbe(dim, num_classes=3, kind=kind, hidden_dim=args.mlp_hidden_dim, dropout=args.mlp_dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_learning_rate, weight_decay=args.probe_weight_decay)
    weights = None if args.no_class_weights else class_weights_from_labels(y_train, 3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    ds = TensorDataset(x_train.float(), y_train.long())
    loader = DataLoader(ds, batch_size=args.probe_batch_size, shuffle=True, num_workers=0, drop_last=False)
    best_state = copy.deepcopy(probe.state_dict())
    best_val = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        val_metrics = eval_tabular_probe(probe, x_val, y_val, device)
        val_bal = float(val_metrics["balanced_accuracy"])
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, n_seen), "val_balanced_accuracy": val_bal})
        if val_bal > best_val:
            best_val = val_bal
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    metrics = {
        "kind": kind,
        "best_epoch": best_epoch,
        "history": history,
        "train": eval_tabular_probe(probe, x_train, y_train, device),
        "val": eval_tabular_probe(probe, x_val, y_val, device),
        "test": eval_tabular_probe(probe, x_test, y_test, device),
    }
    return probe, metrics


# -----------------------------------------------------------------------------
# Patch-token cross-attention downstream heads
# -----------------------------------------------------------------------------


class CrossAttentionPool(nn.Module):
    def __init__(self, dim: int, num_queries: int = 1, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b = patch_tokens.size(0)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.attn(q, patch_tokens, patch_tokens, need_weights=False)
        pooled = pooled.mean(dim=1)
        return self.ln(pooled)


class PatchCrossAttentionClassifier(nn.Module):
    def __init__(self, token_dim: int, kind: str, num_classes: int = 3, num_queries: int = 1, num_heads: int = 4,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pooler = CrossAttentionPool(token_dim, num_queries=num_queries, num_heads=num_heads)
        if kind == "linear":
            self.classifier = nn.Linear(token_dim, num_classes)
        elif kind == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(kind)

    def pool(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.pooler(patch_tokens)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(patch_tokens))


def eval_patch_classifier(
    encoder: MedJEPAEncoder,
    clf: PatchCrossAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    encoder.eval()
    clf.eval()
    preds, trues = [], []
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    with torch.inference_mode():
        for x, y, _idx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_cuda and use_amp)):
                _, patches = encoder.forward_tokens(x)
            logits = clf(patches.float())
            preds.append(logits.argmax(dim=1).cpu())
            trues.append(y.cpu())
    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()
    return compute_metrics_from_pred(true, pred)


def train_patch_classifier(
    encoder: MedJEPAEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    token_dim: int,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    y_train_for_weights: torch.Tensor,
) -> tuple[PatchCrossAttentionClassifier, dict[str, Any]]:
    clf = PatchCrossAttentionClassifier(
        token_dim,
        kind=kind,
        num_queries=args.patch_num_queries,
        num_heads=args.patch_attn_heads,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.patch_learning_rate, weight_decay=args.patch_weight_decay)
    weights = None if args.no_class_weights else class_weights_from_labels(y_train_for_weights, 3).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_state = copy.deepcopy(clf.state_dict())
    best_val = -1.0
    best_epoch = 0
    history = []
    encoder.eval()
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    for epoch in range(1, args.patch_epochs + 1):
        clf.train()
        total_loss = 0.0
        n_seen = 0
        for x, y, _idx in tqdm(train_loader, desc=f"Patch {kind} epoch {epoch}/{args.patch_epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.inference_mode():
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_cuda and use_amp)):
                    _, patches = encoder.forward_tokens(x)
            opt.zero_grad(set_to_none=True)
            logits = clf(patches.float())
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(x)
            n_seen += len(x)
        val_metrics = eval_patch_classifier(encoder, clf, val_loader, device, use_amp)
        val_bal = float(val_metrics["balanced_accuracy"])
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, n_seen), "val_balanced_accuracy": val_bal})
        print(f"Patch {kind} epoch {epoch}: val balanced acc={val_bal:.4f}", flush=True)
        if val_bal > best_val:
            best_val = val_bal
            best_epoch = epoch
            best_state = copy.deepcopy(clf.state_dict())
    clf.load_state_dict(best_state)
    metrics = {
        "kind": kind,
        "best_epoch": best_epoch,
        "history": history,
        "train": eval_patch_classifier(encoder, clf, train_loader, device, use_amp),
        "val": eval_patch_classifier(encoder, clf, val_loader, device, use_amp),
        "test": eval_patch_classifier(encoder, clf, test_loader, device, use_amp),
    }
    return clf, metrics


# -----------------------------------------------------------------------------
# PCA / plotting
# -----------------------------------------------------------------------------

COLLAPSED_COLORS = {
    "routine": "#1f77b4",
    "follow_up": "#9467bd",
    "biopsy": "#d62728",
}


def categorical_color_map(values: Iterable[Any]) -> dict[Any, Any]:
    vals = [v for v in pd.Series(list(values)).dropna().unique().tolist()]
    vals = sorted(vals, key=lambda x: str(x))
    cmap = plt.get_cmap("tab20")
    return {v: cmap(i % 20) for i, v in enumerate(vals)}


def compute_pca(features: torch.Tensor, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x = features.float().numpy()
    n_components = min(n_components, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(x)
    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    evr = np.zeros(3, dtype=np.float64)
    evr[: len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_
    return coords, evr


def add_text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 10) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=17, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        fig.text(0.05, y, line, fontsize=fontsize, va="top", family="monospace" if line.startswith("  ") else None)
        y -= 0.030
        if y < 0.05:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(11, 8.5))
            y = 0.95
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_metrics_table_page(pdf: PdfPages, metrics: dict[str, Any]) -> None:
    reps = [r for r in ["head512", "cls", "patch_cross_attention"] if r in metrics]
    columns = ["Representation", "Linear bal. acc", "MLP bal. acc", "Linear macro F1", "MLP macro F1"]
    rows = []
    for rep in reps:
        lin = metrics[rep].get("linear", {}).get("test", {})
        mlp = metrics[rep].get("mlp", {}).get("test", {})
        rows.append([
            rep,
            f"{lin.get('balanced_accuracy', float('nan')):.4f}" if lin else "-",
            f"{mlp.get('balanced_accuracy', float('nan')):.4f}" if mlp else "-",
            f"{lin.get('macro_f1', float('nan')):.4f}" if lin else "-",
            f"{mlp.get('macro_f1', float('nan')):.4f}" if mlp else "-",
        ])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.set_title("Collapsed BI-RADS probe comparison on test split", fontsize=15, fontweight="bold", pad=18)
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_balacc_bar_page(pdf: PdfPages, metrics: dict[str, Any]) -> None:
    reps = [r for r in ["head512", "cls", "patch_cross_attention"] if r in metrics]
    linear_vals = [metrics[r].get("linear", {}).get("test", {}).get("balanced_accuracy", np.nan) for r in reps]
    mlp_vals = [metrics[r].get("mlp", {}).get("test", {}).get("balanced_accuracy", np.nan) for r in reps]
    x = np.arange(len(reps))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, linear_vals, width, label="Linear probe")
    ax.bar(x + width / 2, mlp_vals, width, label="MLP classifier")
    ax.axhline(1/3, linestyle="--", linewidth=1, label="3-class random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(reps)
    ax.set_ylim(0, max(1.0, np.nanmax(linear_vals + mlp_vals) + 0.05))
    ax.set_ylabel("Test balanced accuracy")
    ax.set_title("BI-RADS balanced accuracy by representation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def scatter_2d(ax, coords: np.ndarray, meta: pd.DataFrame, color_col: str, title: str, evr: np.ndarray) -> None:
    if color_col == "collapsed_birads":
        order = ["routine", "follow_up", "biopsy"]
        color_map = COLLAPSED_COLORS
    elif color_col == "view":
        order = ["MLO", "CC", "unknown"]
        vals = [v for v in order if v in set(meta[color_col].astype(str))]
        extra = sorted(set(meta[color_col].astype(str)) - set(vals))
        order = vals + extra
        color_map = categorical_color_map(order)
    else:
        order = sorted(meta[color_col].astype(str).dropna().unique().tolist())
        color_map = categorical_color_map(order)
    for lab in order:
        mask = meta[color_col].astype(str).values == str(lab)
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.65, linewidths=0, label=str(lab), color=color_map.get(lab, None), rasterized=True)
    ax.set_title(f"{title}\nPC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)


def add_pca_comparison_page(pdf: PdfPages, pca_results: dict[str, dict[str, Any]], meta: pd.DataFrame, color_col: str) -> None:
    reps = [r for r in ["head512", "cls", "patch_cross_attention"] if r in pca_results]
    fig, axes = plt.subplots(1, len(reps), figsize=(5.1 * len(reps), 4.7), squeeze=False)
    for ax, rep in zip(axes[0], reps):
        scatter_2d(ax, pca_results[rep]["coords"], meta, color_col, rep, pca_results[rep]["explained_variance_ratio"])
    # One consolidated legend on the right.
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", fontsize=8, markerscale=2)
        fig.subplots_adjust(right=0.83)
    fig.suptitle(f"Local PCA per representation colored by {color_col}", fontsize=15, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_pdf_report(
    output_pdf: str | Path,
    args: argparse.Namespace,
    cfg: ModelConfig,
    checkpoint_raw_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    metrics: dict[str, Any],
    pca_results: dict[str, dict[str, Any]],
    elapsed_sec: float,
) -> None:
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        lines = [
            f"Created at: {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z",
            f"Checkpoint: {args.checkpoint}",
            f"Backbone: {cfg.backbone_name}",
            f"Image size: {cfg.image_size}",
            f"Current head embedding dimension: {cfg.backbone_output_dim}",
            "Raw token dimension inferred from model during extraction (see JSON for exact feature dims).",
            "",
            "Representations:",
            "  head512: timm-head output used by previous PCA/linear probes.",
            "  cls: raw final ViT CLS token before timm head.",
            "  patch_cross_attention: frozen patch tokens collapsed by supervised cross-attention pooling.",
            "",
            "Probe target: collapsed BI-RADS (routine / follow_up / biopsy). Main metric: balanced accuracy.",
            "PCA colorings: collapsed_birads, view, machine_family, dataset.",
            "PCA note: PCA is fitted separately for each representation; axes are local and not directly comparable.",
            "Patch PCA note: patch_cross_attention PCA uses pooled vectors from the best patch MLP classifier.",
            "",
            f"Train rows: {len(train_df):,}; val rows: {len(val_df):,}; test rows: {len(test_df):,}; PCA rows: {len(pca_df):,}",
            f"Train collapsed counts: {train_df['collapsed_birads'].value_counts().to_dict()}",
            f"PCA collapsed counts: {pca_df['collapsed_birads'].value_counts().to_dict()}",
            f"PCA view counts: {pca_df['view'].value_counts().to_dict()}",
            f"PCA machine family counts: {pca_df['machine_family'].value_counts().to_dict()}",
            f"Elapsed wall time: {elapsed_sec / 60:.1f} min",
        ]
        add_text_page(pdf, "MedJEPA representation comparison report", lines)
        add_metrics_table_page(pdf, metrics)
        add_balacc_bar_page(pdf, metrics)
        for col in ["collapsed_birads", "view", "machine_family", "dataset"]:
            add_pca_comparison_page(pdf, pca_results, pca_df, col)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Device: {device}; use_amp={use_amp}", flush=True)

    model, raw_cfg, aug_cfg, cfg = load_checkpoint_and_model(args.checkpoint, device)
    print("Loaded checkpoint model config:", asdict(cfg), flush=True)

    full_df, train_df, val_df, test_df, full_num_rows = load_splits(args)

    probe_train_df = balanced_sample_df(train_df, "target_collapsed", args.probe_train_max_samples, args.seed)
    patch_train_df = balanced_sample_df(train_df, "target_collapsed", args.patch_train_max_samples, args.seed + 1)
    split_for_pca = {"train": train_df, "val": val_df, "test": test_df}[args.pca_split]
    if args.pca_sampling == "balanced_collapsed":
        pca_df = balanced_sample_df(split_for_pca, "target_collapsed", args.pca_max_samples, args.seed + 2)
    else:
        pca_df = split_for_pca.sample(n=min(len(split_for_pca), args.pca_max_samples), random_state=args.seed + 2).reset_index(drop=True)

    print(f"Probe train rows: {len(probe_train_df):,}; patch train rows: {len(patch_train_df):,}; PCA rows: {len(pca_df):,}", flush=True)

    train_loader = make_loader(probe_train_df, args.bin, full_num_rows, cfg, aug_cfg, args.batch_size, args.num_workers)
    val_loader = make_loader(val_df, args.bin, full_num_rows, cfg, aug_cfg, args.batch_size, args.num_workers)
    test_loader = make_loader(test_df, args.bin, full_num_rows, cfg, aug_cfg, args.batch_size, args.num_workers)
    pca_loader = make_loader(pca_df, args.bin, full_num_rows, cfg, aug_cfg, args.batch_size, args.num_workers)

    metrics: dict[str, Any] = {}
    pca_results: dict[str, dict[str, Any]] = {}
    feature_dims: dict[str, int] = {}

    # Head512 and raw CLS token experiments.
    for rep in ["head512", "cls"]:
        print(f"\n=== Representation: {rep} ===", flush=True)
        x_train, y_train, _ = extract_features(model, train_loader, device, rep, use_amp)
        x_val, y_val, _ = extract_features(model, val_loader, device, rep, use_amp)
        x_test, y_test, _ = extract_features(model, test_loader, device, rep, use_amp)
        x_pca, _y_pca, _ = extract_features(model, pca_loader, device, rep, use_amp)
        feature_dims[rep] = int(x_train.shape[1])

        _, linear_metrics = train_tabular_probe(x_train, y_train, x_val, y_val, x_test, y_test, "linear", args, device)
        _, mlp_metrics = train_tabular_probe(x_train, y_train, x_val, y_val, x_test, y_test, "mlp", args, device)
        metrics[rep] = {"linear": linear_metrics, "mlp": mlp_metrics}
        coords, evr = compute_pca(x_pca, n_components=3)
        pca_results[rep] = {"coords": coords, "explained_variance_ratio": evr, "feature_dim": int(x_pca.shape[1])}
        print(f"{rep}: linear test balanced acc={linear_metrics['test']['balanced_accuracy']:.4f}; mlp test balanced acc={mlp_metrics['test']['balanced_accuracy']:.4f}", flush=True)

    # Patch tokens with trainable supervised cross-attention pooling.
    if not args.skip_patch:
        print("\n=== Representation: patch_cross_attention ===", flush=True)
        patch_train_loader = make_loader(patch_train_df, args.bin, full_num_rows, cfg, aug_cfg, args.patch_batch_size, args.num_workers, shuffle=True)
        patch_val_loader = make_loader(val_df, args.bin, full_num_rows, cfg, aug_cfg, args.patch_batch_size, args.num_workers)
        patch_test_loader = make_loader(test_df, args.bin, full_num_rows, cfg, aug_cfg, args.patch_batch_size, args.num_workers)
        patch_pca_loader = make_loader(pca_df, args.bin, full_num_rows, cfg, aug_cfg, args.patch_batch_size, args.num_workers)

        # Infer token dimension.
        with torch.inference_mode():
            xb, _, _ = next(iter(patch_train_loader))
            xb = xb.to(device)
            with autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=(device.type == "cuda" and use_amp)):
                _cls, patches = model.forward_tokens(xb)
            token_dim = int(patches.shape[-1])
            num_patch_tokens = int(patches.shape[1])
        feature_dims["patch_tokens"] = token_dim
        feature_dims["num_patch_tokens"] = num_patch_tokens
        print(f"Patch tokens: N={num_patch_tokens}, dim={token_dim}", flush=True)

        y_patch_train = torch.tensor(patch_train_df["target_collapsed"].to_numpy(), dtype=torch.long)
        patch_linear, patch_linear_metrics = train_patch_classifier(model, patch_train_loader, patch_val_loader, patch_test_loader, token_dim, "linear", args, device, use_amp, y_patch_train)
        patch_mlp, patch_mlp_metrics = train_patch_classifier(model, patch_train_loader, patch_val_loader, patch_test_loader, token_dim, "mlp", args, device, use_amp, y_patch_train)
        metrics["patch_cross_attention"] = {"linear": patch_linear_metrics, "mlp": patch_mlp_metrics}
        # Use the best MLP pooler for PCA because it is the more expressive BI-RADS classifier.
        x_pca_patch, _y_pca_patch, _ = extract_patch_pooled(model, patch_mlp, patch_pca_loader, device, use_amp)
        coords, evr = compute_pca(x_pca_patch, n_components=3)
        pca_results["patch_cross_attention"] = {"coords": coords, "explained_variance_ratio": evr, "feature_dim": int(x_pca_patch.shape[1]), "pca_pooler": "best_patch_mlp"}
        print(f"patch_cross_attention: linear test balanced acc={patch_linear_metrics['test']['balanced_accuracy']:.4f}; mlp test balanced acc={patch_mlp_metrics['test']['balanced_accuracy']:.4f}", flush=True)

    elapsed = time.time() - start_time
    output_json = Path(args.output_json) if args.output_json else Path(args.output_pdf).with_suffix(".json")
    result = {
        "script": Path(__file__).name,
        "created_at_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "checkpoint": str(args.checkpoint),
        "model_config": asdict(cfg),
        "checkpoint_config_subset": {k: raw_cfg.get(k) for k in ["run_name", "backbone_name", "image_size", "backbone_output_dim", "projection_dim", "projector_hidden_dim", "drop_path_rate", "lambda_sigreg", "projection_normalization"]},
        "args": vars(args),
        "data": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "probe_train_rows": len(probe_train_df),
            "patch_train_rows": len(patch_train_df),
            "pca_rows": len(pca_df),
            "pca_counts": {
                "collapsed_birads": pca_df["collapsed_birads"].value_counts().to_dict(),
                "view": pca_df["view"].value_counts().to_dict(),
                "machine_family": pca_df["machine_family"].value_counts().to_dict(),
                "dataset": pca_df["dataset"].value_counts().to_dict(),
            },
        },
        "feature_dims": feature_dims,
        "metrics": metrics,
        "pca": {rep: {"explained_variance_ratio": v["explained_variance_ratio"].tolist(), "feature_dim": v.get("feature_dim"), "pca_pooler": v.get("pca_pooler")} for rep, v in pca_results.items()},
        "elapsed_sec": elapsed,
    }
    save_json(result, output_json)
    create_pdf_report(args.output_pdf, args, cfg, raw_cfg, train_df, val_df, test_df, pca_df, metrics, pca_results, elapsed)
    print(f"Wrote PDF: {args.output_pdf}", flush=True)
    print(f"Wrote JSON: {output_json}", flush=True)


if __name__ == "__main__":
    main()
