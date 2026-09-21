#!/usr/bin/env python3
"""
Run multiple frozen linear probes on MedJEPA / LeJEPA backbone embeddings.

Probes included by default:
- dataset
- machine
- view
- laterality
- collapsed_birads

Output: one JSON file containing metrics, label mappings, confusion matrices,
classification reports, class counts, and timing information for every probe.

This script is intentionally plot-free. It is meant to quantify whether the
representation is more predictive of clinical labels or domain/acquisition labels.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("Missing dependency: timm. Install with: pip install timm") from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )
except ImportError as exc:
    raise ImportError("Missing dependency: scikit-learn. Install with: pip install scikit-learn") from exc


# -----------------------------
# Config / CLI
# -----------------------------

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
    backbone_output_dim: int = 512
    projection_dim: int = 16
    projector_hidden_dim: int = 2048
    drop_path_rate: float = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run frozen linear probes on saved MedJEPA embeddings.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to final_lejepa_checkpoint.pt")
    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--bin", required=True, type=str)
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--val-csv", required=True, type=str)
    p.add_argument("--test-csv", required=True, type=str)
    p.add_argument("--output-json", required=True, type=str)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--probe-epochs", type=int, default=50)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--probe-learning-rate", type=float, default=1e-3)
    p.add_argument("--probe-weight-decay", type=float, default=1e-7)
    p.add_argument("--probe-train-max-samples", type=int, default=60000,
                   help="0 means use all train features. Otherwise balanced subsample per probe target.")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--select-best-by", type=str, default="val_macro_f1",
                   choices=["val_macro_f1", "val_balanced_accuracy", "last"],
                   help="Checkpoint the best linear probe by validation metric, or use the last epoch.")

    p.add_argument("--machine-min-train-count", type=int, default=20,
                   help="Machine classes with fewer train samples are grouped into Other. Use 1 to keep all.")
    p.add_argument("--machine-top-k", type=int, default=0,
                   help="If >0, keep only top-K train machine labels and group the rest into Other.")
    p.add_argument("--include-machine-family", action="store_true",
                   help="Also run an extra probe for machine_family in addition to raw/grouped machine.")
    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def deep_get(dct: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict containing model_state_dict.")
    cfg = ckpt.get("config", {}) or {}
    aug_cfg = ckpt.get("augmentation_config", {}) or {}
    return ckpt["model_state_dict"], cfg, aug_cfg


def config_from_checkpoint(raw_cfg: dict[str, Any], aug_cfg: dict[str, Any]) -> ModelConfig:
    # The training script saved config as a dict via asdict(cfg). Use safe fallbacks.
    image_size = int(raw_cfg.get("image_size", 0) or 0)
    if image_size <= 0:
        image_size = int(deep_get(aug_cfg, ["image", "output_size"], 384))
    return ModelConfig(
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


# -----------------------------
# Label and metadata handling
# -----------------------------

def normalize_birads_value(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    for k in ["1", "2", "3", "4", "5"]:
        if s == k or s.startswith(k + ".") or s.startswith(k + " ") or f"({k})" in s:
            return int(k)
    try:
        return int(float(s))
    except Exception:
        return np.nan


def collapse_birads_numeric(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    x = int(x)
    if x in (1, 2):
        return "routine"
    if x == 3:
        return "follow_up"
    if x in (4, 5):
        return "biopsy"
    return "unknown"


def read_csv_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label_col = "original_birads" if "original_birads" in df.columns else "birads"
    if label_col not in df.columns:
        raise ValueError("CSV must contain original_birads or birads.")
    df["birads_numeric"] = df[label_col].apply(normalize_birads_value)
    df = df[df["birads_numeric"].isin([1, 2, 3, 4, 5])].copy()
    df["birads_numeric"] = df["birads_numeric"].astype(int)
    df["collapsed_birads"] = df["birads_numeric"].apply(collapse_birads_numeric)
    return df


def ensure_original_index(split_df: pd.DataFrame, full_df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_df = split_df.copy()
    if "original_index" in split_df.columns:
        split_df["original_index"] = split_df["original_index"].astype(int)
        return split_df
    if "id" not in split_df.columns or "id" not in full_df_raw.columns:
        raise ValueError(f"{split_name} split needs original_index or id for memmap lookup.")
    id_to_idx = pd.Series(np.arange(len(full_df_raw)), index=full_df_raw["id"].astype(str)).to_dict()
    split_df["original_index"] = split_df["id"].astype(str).map(id_to_idx)
    missing = int(split_df["original_index"].isna().sum())
    if missing:
        raise ValueError(f"Could not map {missing} rows in {split_name} split by id.")
    split_df["original_index"] = split_df["original_index"].astype(int)
    return split_df


def load_splits(full_csv: str, train_csv: str, val_csv: str, test_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_raw = read_csv_clean(full_csv)
    full_df = prepare_labels(full_raw)
    if "original_index" not in full_df.columns:
        full_df = full_df.reset_index(drop=False).rename(columns={"index": "original_index"})

    train_df = prepare_labels(ensure_original_index(read_csv_clean(train_csv), full_raw, "train"))
    val_df = prepare_labels(ensure_original_index(read_csv_clean(val_csv), full_raw, "val"))
    test_df = prepare_labels(ensure_original_index(read_csv_clean(test_csv), full_raw, "test"))
    return full_df, train_df, val_df, test_df


def parse_context_cell(x: Any) -> dict[str, Any]:
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def extract_context_field(ctx: Any, field: str) -> str:
    obj = parse_context_cell(ctx)
    exam = obj.get("exam", {}) if isinstance(obj, dict) else {}
    if not isinstance(exam, dict):
        return "unknown"
    val = exam.get(field, "unknown")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "unknown"
    s = str(val).strip().lower()
    return s if s else "unknown"


def clean_machine_label(x: Any) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    return s if s else "unknown"


def machine_family(x: Any) -> str:
    s = clean_machine_label(x).lower()
    if s == "unknown":
        return "unknown"
    if "hologic" in s or "lorad" in s or "selenia" in s:
        return "hologic"
    if "ge" in s or "senograph" in s or "senographe" in s:
        return "ge"
    if "howtek" in s or "lumysis" in s or "dba" in s:
        return "howtek_lumysis"
    if "fuji" in s or "fujifilm" in s:
        return "fujifilm"
    if "siemens" in s:
        return "siemens"
    if "philips" in s:
        return "philips"
    return "other"


def add_probe_metadata(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split
    if "dataset" not in df.columns:
        df["dataset"] = "unknown"
    df["dataset"] = df["dataset"].fillna("unknown").astype(str)

    if "machine" not in df.columns:
        df["machine"] = "unknown"
    df["machine"] = df["machine"].apply(clean_machine_label)
    df["machine_family"] = df["machine"].apply(machine_family)

    if "context" not in df.columns:
        df["context"] = ""
    df["view"] = df["context"].apply(lambda x: extract_context_field(x, "view"))
    df["laterality"] = df["context"].apply(lambda x: extract_context_field(x, "laterality"))
    return df


def apply_machine_grouping(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                           min_train_count: int, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    counts = train_df["machine"].value_counts()
    keep = set(counts.index.tolist())
    if min_train_count > 1:
        keep &= set(counts[counts >= min_train_count].index.tolist())
    if top_k and top_k > 0:
        keep &= set(counts.head(top_k).index.tolist())
    if not keep:
        keep = set(counts.head(max(1, top_k or 10)).index.tolist())

    def group(s: pd.Series) -> pd.Series:
        return s.apply(lambda x: x if x in keep else "Other")

    out_train, out_val, out_test = train_df.copy(), val_df.copy(), test_df.copy()
    out_train["machine"] = group(out_train["machine"])
    out_val["machine"] = group(out_val["machine"])
    out_test["machine"] = group(out_test["machine"])
    info = {
        "machine_grouping": {
            "min_train_count": int(min_train_count),
            "top_k": int(top_k),
            "kept_labels": sorted([str(x) for x in keep]),
            "num_kept_labels": int(len(keep)),
            "train_raw_num_labels": int(len(counts)),
        }
    }
    return out_train, out_val, out_test, info


# -----------------------------
# Dataset / model
# -----------------------------

class EvalMGAugmentation(nn.Module):
    def __init__(self, aug_cfg: dict[str, Any], image_size: int):
        super().__init__()
        self.cfg = aug_cfg
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._foreground_crop(x)
        x = TF.resize(x, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return x.clamp(0, 1)

    def _foreground_crop(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["preprocessing", "foreground_crop"], {})
        if not c.get("enabled", False):
            return x
        threshold = float(c.get("threshold_abs", 1e-6))
        margin_frac = float(c.get("margin_frac", 0.05))
        min_area_frac = float(c.get("min_foreground_area_frac", 0.01))
        mask = x[0] > threshold
        ys, xs = torch.where(mask)
        h, w = x.shape[-2:]
        if len(xs) < int(h * w * min_area_frac):
            return x
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mh, mw = int((y1 - y0) * margin_frac), int((x1 - x0) * margin_frac)
        return x[:, max(0, y0 - mh):min(h, y1 + mh), max(0, x0 - mw):min(w, x1 + mw)]


class MedJEPAEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, bin_path: str | Path, full_num_rows: int,
                 model_cfg: ModelConfig, transform: nn.Module):
        self.df = df.reset_index(drop=True)
        self.bin_path = Path(bin_path)
        self.full_num_rows = full_num_rows
        self.image_shape = (model_cfg.image_height, model_cfg.image_width)
        self.dtype = np.dtype(model_cfg.memmap_dtype)
        self.normalize_mode = model_cfg.normalize_mode
        self.percentile_low = model_cfg.percentile_low
        self.percentile_high = model_cfg.percentile_high
        self.transform = transform
        self._imgs: Optional[np.memmap] = None
        if "original_index" not in self.df.columns:
            raise ValueError("Split dataframe requires original_index.")

    def __len__(self) -> int:
        return len(self.df)

    def _open(self) -> np.memmap:
        if self._imgs is None:
            self._imgs = np.memmap(self.bin_path, dtype=self.dtype, mode="r", shape=(self.full_num_rows, *self.image_shape))
        return self._imgs

    def _load_tensor(self, original_index: int) -> torch.Tensor:
        arr = self._open()[original_index].astype(np.float32)
        if self.normalize_mode == "uint16":
            arr = arr / 65535.0 if self.dtype == np.dtype("uint16") else arr / max(float(arr.max()), 1.0)
        elif self.normalize_mode == "per_image_percentile":
            lo, hi = np.percentile(arr, [self.percentile_low, self.percentile_high])
            arr = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_tensor(int(row["original_index"]))
        x = self.transform(x).unsqueeze(0)  # [V=1,C,H,W]
        return x, idx


def collate_eval(batch):
    views = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return views, indices


class ViTEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=False,
            num_classes=cfg.backbone_output_dim,
            drop_path_rate=cfg.drop_path_rate,
            img_size=cfg.image_size,
            in_chans=1,
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.backbone_output_dim, cfg.projector_hidden_dim),
            nn.BatchNorm1d(cfg.projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.projector_hidden_dim, cfg.projector_hidden_dim),
            nn.BatchNorm1d(cfg.projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.projector_hidden_dim, cfg.projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, v = x.shape[:2]
        flat = x.flatten(0, 1)
        emb = self.backbone(flat)
        proj = self.proj(emb).reshape(b, v, -1)  # [B,V,D], v3-compatible
        return emb, proj


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Handles checkpoints that may accidentally contain DataParallel prefixes.
    if all(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


@torch.inference_mode()
def extract_features(df: pd.DataFrame, bin_path: str, n_full: int, model_cfg: ModelConfig, aug_cfg: dict[str, Any],
                     net: nn.Module, device: torch.device, batch_size: int, num_workers: int,
                     split_name: str) -> tuple[torch.Tensor, pd.DataFrame, float]:
    t0 = time.perf_counter()
    ds = MedJEPAEvalDataset(df, bin_path, n_full, model_cfg, EvalMGAugmentation(aug_cfg, model_cfg.image_size))
    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_eval,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    feats, order = [], []
    net.eval()
    for views, idx in tqdm(loader, desc=f"Extract features [{split_name}]"):
        views = views.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            emb, _ = net(views)
        feats.append(emb.float().cpu())
        order.append(idx.cpu())
    feat = torch.cat(feats, dim=0)
    idx_all = torch.cat(order, dim=0).numpy()
    out_df = df.iloc[idx_all].reset_index(drop=True)
    return feat, out_df, time.perf_counter() - t0


# -----------------------------
# Linear probe training
# -----------------------------

class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_counts(labels: list[str]) -> dict[str, int]:
    vc = pd.Series(labels).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def make_label_map(train_labels: list[str], val_labels: list[str], test_labels: list[str], ordered: Optional[list[str]] = None) -> dict[str, int]:
    present = set(train_labels) | set(val_labels) | set(test_labels)
    if ordered:
        names = [x for x in ordered if x in present]
        names += sorted([x for x in present if x not in set(names)])
    else:
        # Put Other/unknown at the end for readability.
        names = sorted([x for x in present if x not in {"Other", "unknown"}])
        if "Other" in present:
            names.append("Other")
        if "unknown" in present:
            names.append("unknown")
    return {name: i for i, name in enumerate(names)}


def labels_to_tensor(labels: list[str], label_to_idx: dict[str, int]) -> torch.Tensor:
    return torch.tensor([label_to_idx[str(x)] for x in labels], dtype=torch.long)


def balanced_subset_indices(y: torch.Tensor, max_total: int, seed: int) -> torch.Tensor:
    if max_total <= 0 or len(y) <= max_total:
        return torch.arange(len(y))
    g = torch.Generator().manual_seed(seed)
    classes = sorted(y.unique().tolist())
    per_class = max(1, max_total // max(1, len(classes)))
    chunks = []
    for c in classes:
        idx = torch.where(y == int(c))[0]
        perm = torch.randperm(len(idx), generator=g)
        chunks.append(idx[perm[: min(per_class, len(idx))]])
    out = torch.cat(chunks)
    return out[torch.randperm(len(out), generator=g)]


def compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y.cpu(), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-12)


def evaluate_probe(probe: nn.Module, x: torch.Tensor, y: torch.Tensor, class_names: list[str], device: torch.device) -> dict[str, Any]:
    probe.eval()
    with torch.inference_mode():
        logits = probe(x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()

    true = y.cpu().numpy()

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
    }

def run_one_probe(target_name: str,
                  train_x: torch.Tensor, val_x: torch.Tensor, test_x: torch.Tensor,
                  train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    t0 = time.perf_counter()
    if target_name not in train_df.columns:
        return {"target": target_name, "status": "skipped", "reason": f"Column {target_name!r} not found."}

    train_labels = train_df[target_name].fillna("unknown").astype(str).tolist()
    val_labels = val_df[target_name].fillna("unknown").astype(str).tolist()
    test_labels = test_df[target_name].fillna("unknown").astype(str).tolist()

    ordered = None
    if target_name == "collapsed_birads":
        ordered = ["routine", "follow_up", "biopsy"]
    elif target_name == "laterality":
        ordered = ["left", "right", "bilateral", "unknown"]
    elif target_name == "view":
        ordered = ["cranial caudal", "mediolateral oblique", "mediolateral", "lateromedial", "unknown"]

    label_to_idx = make_label_map(train_labels, val_labels, test_labels, ordered=ordered)
    idx_to_label = [None] * len(label_to_idx)
    for k, v in label_to_idx.items():
        idx_to_label[v] = k
    class_names = [str(x) for x in idx_to_label]
    num_classes = len(class_names)

    if num_classes < 2:
        return {"target": target_name, "status": "skipped", "reason": "Fewer than 2 classes available."}

    train_y = labels_to_tensor(train_labels, label_to_idx)
    val_y = labels_to_tensor(val_labels, label_to_idx)
    test_y = labels_to_tensor(test_labels, label_to_idx)

    # Balanced subsample only for probe training. Evaluation always uses full val/test.
    train_idx = balanced_subset_indices(train_y, args.probe_train_max_samples, args.seed)
    train_x_sub = train_x[train_idx]
    train_y_sub = train_y[train_idx]

    probe = LinearProbe(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_learning_rate, weight_decay=args.probe_weight_decay)
    weights = None if args.no_class_weights else compute_class_weights(train_y_sub, num_classes).to(device)
    loader = DataLoader(TensorDataset(train_x_sub, train_y_sub), batch_size=args.probe_batch_size, shuffle=True, drop_last=False)

    best_score = -float("inf")
    best_state = None
    best_epoch = -1

    for epoch in tqdm(range(args.probe_epochs), desc=f"Train probe [{target_name}]", leave=False):
        probe.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(xb.size(0))

        val_metrics = evaluate_probe(probe, val_x, val_y, class_names, device)

        if args.select_best_by == "last":
            score = float(epoch)
        elif args.select_best_by == "val_balanced_accuracy":
            score = val_metrics["balanced_accuracy"]
        else:
            score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)

    train_metrics = evaluate_probe(probe, train_x_sub, train_y_sub, class_names, device)
    val_metrics = evaluate_probe(probe, val_x, val_y, class_names, device)
    test_metrics = evaluate_probe(probe, test_x, test_y, class_names, device)

    return {
        "target": target_name,
        "status": "ok",
        "num_classes": int(num_classes),
        "class_names": class_names,
        "label_to_index": label_to_idx,
        "counts": {
            "train_full": class_counts(train_labels),
            "train_probe_subset": class_counts([class_names[int(i)] for i in train_y_sub.tolist()]),
            "val": class_counts(val_labels),
            "test": class_counts(test_labels),
        },
        "probe_training": {
            "epochs": int(args.probe_epochs),
            "best_epoch": int(best_epoch),
            "selected_by": args.select_best_by,
            "used_class_weights": bool(not args.no_class_weights),
            "train_samples_used": int(len(train_y_sub)),
        },
        "metrics": {
            "train_probe_subset": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "timing_sec": float(time.perf_counter() - t0),
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    set_seed(args.seed)
    script_t0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Visible CUDA devices:", torch.cuda.device_count())

    # Load checkpoint/model.
    t0 = time.perf_counter()
    state_dict, raw_cfg, aug_cfg = load_checkpoint(args.checkpoint, device)
    model_cfg = config_from_checkpoint(raw_cfg, aug_cfg)
    net = ViTEncoder(model_cfg).to(device)
    missing, unexpected = net.load_state_dict(clean_state_dict(state_dict), strict=False)
    if missing or unexpected:
        print("WARNING: non-strict checkpoint load:")
        print("  missing:", missing)
        print("  unexpected:", unexpected)
    net.eval()
    model_load_sec = time.perf_counter() - t0

    # Load splits / metadata.
    t0 = time.perf_counter()
    full_df, train_df, val_df, test_df = load_splits(args.full_csv, args.train_csv, args.val_csv, args.test_csv)
    train_df = add_probe_metadata(train_df, "train")
    val_df = add_probe_metadata(val_df, "val")
    test_df = add_probe_metadata(test_df, "test")
    train_df, val_df, test_df, machine_group_info = apply_machine_grouping(
        train_df, val_df, test_df, args.machine_min_train_count, args.machine_top_k
    )
    metadata_sec = time.perf_counter() - t0

    # Validate memmap size.
    dtype = np.dtype(model_cfg.memmap_dtype)
    expected_bytes = len(full_df) * model_cfg.image_height * model_cfg.image_width * dtype.itemsize
    actual_bytes = Path(args.bin).stat().st_size
    if expected_bytes != actual_bytes:
        raise RuntimeError(
            f"Full CSV and BIN do not match: expected {expected_bytes} bytes, got {actual_bytes} bytes."
        )

    # Extract embeddings once.
    n_full = len(full_df)
    train_x, train_df_ordered, train_extract_sec = extract_features(
        train_df, args.bin, n_full, model_cfg, aug_cfg, net, device, args.batch_size, args.num_workers, "train"
    )
    val_x, val_df_ordered, val_extract_sec = extract_features(
        val_df, args.bin, n_full, model_cfg, aug_cfg, net, device, args.batch_size, args.num_workers, "val"
    )
    test_x, test_df_ordered, test_extract_sec = extract_features(
        test_df, args.bin, n_full, model_cfg, aug_cfg, net, device, args.batch_size, args.num_workers, "test"
    )

    # Probe targets.
    targets = ["dataset", "machine", "view", "laterality", "collapsed_birads"]
    if args.include_machine_family:
        targets.insert(2, "machine_family")

    probes: dict[str, Any] = {}
    for target in targets:
        probes[target] = run_one_probe(
            target,
            train_x,
            val_x,
            test_x,
            train_df_ordered,
            val_df_ordered,
            test_df_ordered,
            args,
            device,
        )
        if probes[target].get("status") == "ok":
            m = probes[target]["metrics"]["test"]
            print(
                f"{target:16s} | test acc={m['accuracy']:.4f} | "
                f"bal_acc={m['balanced_accuracy']:.4f} | macro_f1={m['macro_f1']:.4f}",
                flush=True,
            )
        else:
            print(f"{target:16s} | skipped: {probes[target].get('reason')}", flush=True)

    summary_rows = []
    for name, result in probes.items():
        if result.get("status") != "ok":
            continue
        val_m = result["metrics"]["val"]
        test_m = result["metrics"]["test"]
        summary_rows.append({
            "target": name,
            "num_classes": result["num_classes"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "test_accuracy": test_m["accuracy"],
            "test_balanced_accuracy": test_m["balanced_accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_weighted_f1": test_m["weighted_f1"],
        })

    output = {
        "script": "run_medjepa_linear_probe_report.py",
        "checkpoint": str(args.checkpoint),
        "output_json": str(args.output_json),
        "model_config": model_cfg.__dict__,
        "probe_settings": {
            "probe_epochs": int(args.probe_epochs),
            "probe_batch_size": int(args.probe_batch_size),
            "probe_learning_rate": float(args.probe_learning_rate),
            "probe_weight_decay": float(args.probe_weight_decay),
            "probe_train_max_samples": int(args.probe_train_max_samples),
            "used_class_weights": bool(not args.no_class_weights),
            "select_best_by": args.select_best_by,
        },
        "data": {
            "num_train_rows": int(len(train_df_ordered)),
            "num_val_rows": int(len(val_df_ordered)),
            "num_test_rows": int(len(test_df_ordered)),
            **machine_group_info,
        },
        "timing_sec": {
            "model_load": float(model_load_sec),
            "metadata_loading_and_preparation": float(metadata_sec),
            "feature_extraction_train": float(train_extract_sec),
            "feature_extraction_val": float(val_extract_sec),
            "feature_extraction_test": float(test_extract_sec),
            "feature_extraction_total": float(train_extract_sec + val_extract_sec + test_extract_sec),
            "total_script_wall_time": float(time.perf_counter() - script_t0),
        },
        "summary": summary_rows,
        "probes": probes,
    }
    save_json(output, args.output_json)
    print("Saved:", args.output_json)


if __name__ == "__main__":
    main()
