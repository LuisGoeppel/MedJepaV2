#!/usr/bin/env python3
"""
PCA / representation outlier analysis for MedJEPA LeJEPA checkpoints.

Goals
-----
1) Load a v6/v7-style final LeJEPA checkpoint and reconstruct the mammography model.
2) Extract deterministic backbone embeddings and raw projector outputs from a chosen split.
3) Detect compact remote PCA islands and isolated high-dimensional outliers with a hybrid iterative robust-PCA + local-density detector, while explicitly reporting which PCs and which iteration caused each detection.
4) Produce before/after PCA comparisons where ALL detected outliers are removed together.
5) Compare representation statistics before/after outlier removal.
6) Visualize at most N detailed outliers (default 10), each against its nearest normal example.
7) Optionally compare the same samples to a successful reference checkpoint.
8) For detailed outliers, generate stochastic SSL views to test whether the anomaly is
   image-driven or augmentation-driven.
9) Write a self-contained HTML report plus aggregate JSON and a top-N CSV.

The script intentionally does NOT list every detected outlier individually. Outliers beyond
--max-details only contribute to aggregate statistics, as requested.

Typical use
-----------
python pca_outlier_analysis.py \
  --checkpoint /path/to/author_sigreg/models/final_lejepa_checkpoint.pt \
  --reference-checkpoint /path/to/pooled_sigreg/models/final_lejepa_checkpoint.pt \
  --split test \
  --max-details 10 \
  --output-dir /path/to/pca_outlier_analysis

Dependencies
------------
numpy, pandas, torch, torchvision, timm, scikit-learn, matplotlib, Pillow
Optional: opencv-python (only if CLAHE/top-corner component logic is enabled by the aug config)
"""

from __future__ import annotations

import argparse
import base64
import gc
import html
import io
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from tqdm.auto import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("Missing dependency: timm. Install with: pip install timm") from exc


CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_csv_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def robust_location_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(np.std(finite))
    if scale < 1e-12:
        scale = 1.0
    return med, scale


def robust_z(x: np.ndarray, med: float, scale: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - med) / max(scale, 1e-12)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Simple empirical percentile ranks in [0,100]."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.full(len(values), 100.0)
    return 100.0 * ranks / float(len(values) - 1)


def safe_scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        kk = str(k)
        while kk.startswith("module."):
            kk = kk[len("module."):]
        out[kk] = v
    return out


def html_escape(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def fmt_num(x: Any, digits: int = 3) -> str:
    try:
        xf = float(x)
    except Exception:
        return html_escape(x)
    if not np.isfinite(xf):
        return "n/a"
    if abs(xf) >= 10000 or (0 < abs(xf) < 0.001):
        return f"{xf:.3e}"
    return f"{xf:.{digits}f}"


# -----------------------------------------------------------------------------
# Labels / metadata / split loading
# -----------------------------------------------------------------------------

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


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if (
        "collapsed_birads" in df.columns
        and len(df) > 0
        and set(df["collapsed_birads"].dropna().astype(str).unique()).issubset(set(CLASS_NAMES))
    ):
        df = df[df["collapsed_birads"].isin(CLASS_NAMES)].copy()
        df["target_collapsed"] = df["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
        if "birads_numeric" not in df.columns:
            label_col = "original_birads" if "original_birads" in df.columns else "birads"
            if label_col in df.columns:
                df["birads_numeric"] = df[label_col].apply(normalize_birads_value)
        return df

    label_col = "original_birads" if "original_birads" in df.columns else "birads"
    if label_col not in df.columns:
        raise ValueError("CSV must contain collapsed_birads, original_birads, or birads.")

    df["birads_numeric"] = df[label_col].apply(normalize_birads_value)
    df = df[df["birads_numeric"].isin([1, 2, 3, 4, 5])].copy()
    df["birads_numeric"] = df["birads_numeric"].astype(int)
    df["collapsed_birads"] = df["birads_numeric"].apply(collapse_birads_numeric)
    df["target_collapsed"] = df["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
    return df


def infer_machine_family(machine: Any) -> str:
    if pd.isna(machine):
        return "unknown"
    text = str(machine).lower()
    if any(t in text for t in ["hologic", "lorad", "selenia", "dimensions"]):
        return "Hologic/Lorad"
    if any(t in text for t in ["howtek", "lumisys", "lumysis"]):
        return "Howtek/Lumysis"
    if any(t in text for t in ["senographe", "senograph", "ge healthcare", "general electric"]):
        return "GE/Senographe"
    if re.search(r"(^|[^a-z])ge([^a-z]|$)", text):
        return "GE/Senographe"
    return "unknown"


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "machine_family" not in df.columns:
        if "machine" in df.columns:
            df["machine_family"] = df["machine"].map(infer_machine_family)
        else:
            df["machine_family"] = "unknown"
    for col in ["dataset", "machine", "view", "laterality", "patient", "id"]:
        if col not in df.columns:
            df[col] = "unknown"
    return df


def ensure_original_index(
    split_df: pd.DataFrame,
    full_df_raw: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Recover the row index in mg-only-all.csv for every split row.

    The MG CSV is known to contain a small number of duplicate `id` values, so `id`
    alone is not always a safe lookup key. Prefer a genuinely unique column such as
    `exam`, then fall back to unique composite keys.
    """
    split_df = split_df.copy()

    if "original_index" in split_df.columns:
        split_df["original_index"] = split_df["original_index"].astype(int)
        return split_df

    # Try single columns first. `exam` is unique in the current MG dataset, whereas
    # `id` is not guaranteed to be unique.
    single_candidates = ["exam", "id"]
    for key in single_candidates:
        if key not in split_df.columns or key not in full_df_raw.columns:
            continue

        full_key = full_df_raw[key].astype(str)
        if full_key.duplicated().any():
            continue

        mapping = pd.Series(
            np.arange(len(full_df_raw), dtype=np.int64),
            index=full_key,
        )
        mapped = split_df[key].astype(str).map(mapping)

        if mapped.notna().all():
            split_df["original_index"] = mapped.astype(int)
            print(
                f"[index mapping] {split_name}: matched {len(split_df):,} rows "
                f"using unique key '{key}'.",
                flush=True,
            )
            return split_df

    # Fall back to composite keys. Keep the candidates conservative so the mapping
    # remains interpretable and deterministic.
    composite_candidates = [
        ["id", "exam"],
        ["patient", "exam"],
        ["id", "patient", "exam"],
        ["id", "patient", "dataset", "exam"],
        ["id", "patient", "dataset", "machine", "exam"],
    ]

    def make_composite(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        return (
            df[cols]
            .fillna("<NA>")
            .astype(str)
            .agg("||".join, axis=1)
        )

    for cols in composite_candidates:
        if not all(c in split_df.columns and c in full_df_raw.columns for c in cols):
            continue

        full_key = make_composite(full_df_raw, cols)
        if full_key.duplicated().any():
            continue

        mapping = pd.Series(
            np.arange(len(full_df_raw), dtype=np.int64),
            index=full_key,
        )
        split_key = make_composite(split_df, cols)
        mapped = split_key.map(mapping)

        if mapped.notna().all():
            split_df["original_index"] = mapped.astype(int)
            print(
                f"[index mapping] {split_name}: matched {len(split_df):,} rows "
                f"using composite key {cols}.",
                flush=True,
            )
            return split_df

    # Give a useful diagnostic instead of silently making an ambiguous assignment.
    duplicate_id_count = None
    if "id" in full_df_raw.columns:
        duplicate_id_count = int(full_df_raw["id"].astype(str).duplicated(keep=False).sum())

    raise ValueError(
        f"Could not safely reconstruct original_index for split '{split_name}'. "
        f"Tried unique keys {single_candidates} and composites {composite_candidates}. "
        f"Rows in split={len(split_df):,}, rows in full CSV={len(full_df_raw):,}, "
        f"rows participating in duplicate full-CSV ids={duplicate_id_count}. "
        "If this happens with a different dataset, add a stable unique row identifier "
        "to the split CSV or extend the candidate keys."
    )


# -----------------------------------------------------------------------------
# Checkpoint/model reconstruction
# -----------------------------------------------------------------------------

def cfg_get(cfg: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in cfg and cfg[name] not in (None, ""):
            return cfg[name]
    return default


def checkpoint_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        payload = dict(obj)
    elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        payload = {"model_state_dict": obj, "config": {}, "augmentation_config": {}}
    else:
        raise ValueError(f"Unsupported checkpoint structure: {path}")
    payload["model_state_dict"] = strip_module_prefix(payload["model_state_dict"])
    payload.setdefault("config", {})
    payload.setdefault("augmentation_config", {})
    return payload


def infer_architecture(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(payload.get("config", {}) or {})
    state = payload["model_state_dict"]

    patch_w = state.get("backbone.patch_embed.proj.weight")
    patch_size = int(patch_w.shape[-1]) if isinstance(patch_w, torch.Tensor) and patch_w.ndim == 4 else 16

    backbone_output_dim = cfg_get(cfg, "backbone_output_dim", "embedding_dim", default=None)
    head_w = state.get("backbone.head.weight")
    if backbone_output_dim is None and isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
        backbone_output_dim = int(head_w.shape[0])
    if backbone_output_dim is None:
        backbone_output_dim = 512

    proj0 = state.get("proj.0.weight")
    proj_last = state.get("proj.6.weight")
    hidden_dim = cfg_get(cfg, "projector_hidden_dim", default=None)
    projection_dim = cfg_get(cfg, "projection_dim", default=None)
    if hidden_dim is None and isinstance(proj0, torch.Tensor):
        hidden_dim = int(proj0.shape[0])
    if projection_dim is None and isinstance(proj_last, torch.Tensor):
        projection_dim = int(proj_last.shape[0])
    hidden_dim = int(hidden_dim or 2048)
    projection_dim = int(projection_dim or 16)

    image_size = int(cfg_get(cfg, "image_size", default=224))
    if image_size <= 0:
        pos = state.get("backbone.pos_embed")
        if isinstance(pos, torch.Tensor) and pos.ndim == 3:
            n_patches = int(pos.shape[1]) - 1
            grid = int(round(math.sqrt(max(1, n_patches))))
            image_size = grid * patch_size
        else:
            image_size = 224

    backbone_name = cfg_get(cfg, "backbone_name", "backbone", default=None)
    if not backbone_name:
        backbone_name = f"vit_small_patch{patch_size}_224"

    return {
        "backbone_name": str(backbone_name),
        "image_size": image_size,
        "backbone_output_dim": int(backbone_output_dim),
        "projection_dim": projection_dim,
        "projector_hidden_dim": hidden_dim,
        "drop_path_rate": float(cfg_get(cfg, "drop_path_rate", default=0.0)),
        "image_height": int(cfg_get(cfg, "image_height", default=512)),
        "image_width": int(cfg_get(cfg, "image_width", default=512)),
        "memmap_dtype": str(cfg_get(cfg, "memmap_dtype", default="uint16")),
        "normalize_mode": str(cfg_get(cfg, "normalize_mode", default="uint16")),
        "percentile_low": float(cfg_get(cfg, "percentile_low", default=1.0)),
        "percentile_high": float(cfg_get(cfg, "percentile_high", default=99.0)),
        "sigreg_mode": str(cfg_get(cfg, "sigreg_mode", default="unknown")),
        "projection_normalization": str(cfg_get(cfg, "projection_normalization", default="unknown")),
        "script_version": payload.get("script_version", "unknown"),
        "checkpoint_epoch": payload.get("epoch", None),
    }


class ViTEncoder(nn.Module):
    def __init__(self, arch: dict[str, Any]):
        super().__init__()
        self.backbone = timm.create_model(
            arch["backbone_name"],
            pretrained=False,
            num_classes=int(arch["backbone_output_dim"]),
            drop_path_rate=float(arch.get("drop_path_rate", 0.0)),
            img_size=int(arch["image_size"]),
            in_chans=1,
        )
        self.proj = nn.Sequential(
            nn.Linear(int(arch["backbone_output_dim"]), int(arch["projector_hidden_dim"])),
            nn.BatchNorm1d(int(arch["projector_hidden_dim"])),
            nn.ReLU(inplace=True),
            nn.Linear(int(arch["projector_hidden_dim"]), int(arch["projector_hidden_dim"])),
            nn.BatchNorm1d(int(arch["projector_hidden_dim"])),
            nn.ReLU(inplace=True),
            nn.Linear(int(arch["projector_hidden_dim"]), int(arch["projection_dim"])),
        )

    def encode_one(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)
        proj = self.proj(emb)
        return emb, proj


def load_weights_into_existing_model(
    model: ViTEncoder,
    payload: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Load a compatible checkpoint into the already-created model.

    Reusing the model avoids constructing a second timm ViT in the same process.
    On this Determined/UCX environment, the second timm model construction can
    segfault inside timm's weight initialization after DataLoader activity.
    """
    current = model.state_dict()
    incoming = payload["model_state_dict"]

    incompatible_shapes = []
    for k, v in incoming.items():
        if k in current and tuple(current[k].shape) != tuple(v.shape):
            incompatible_shapes.append((k, tuple(current[k].shape), tuple(v.shape)))

    if incompatible_shapes:
        raise RuntimeError(
            f"{label} checkpoint is not architecture-compatible with the existing model. "
            f"Shape mismatches (first 10): {incompatible_shapes[:10]}"
        )

    info = model.load_state_dict(incoming, strict=False)

    meaningful_missing = [k for k in info.missing_keys if not k.endswith("num_batches_tracked")]
    meaningful_unexpected = [k for k in info.unexpected_keys if not k.endswith("num_batches_tracked")]
    if meaningful_missing or meaningful_unexpected:
        raise RuntimeError(
            f"{label} checkpoint/model mismatch.\n"
            f"Missing keys: {meaningful_missing[:20]}\n"
            f"Unexpected keys: {meaningful_unexpected[:20]}"
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return {
        "missing_keys": list(info.missing_keys),
        "unexpected_keys": list(info.unexpected_keys),
    }


def rebuild_projector_in_place(
    model: ViTEncoder,
    arch: dict[str, Any],
    device: torch.device,
) -> None:
    """Rebuild only the projector MLP without constructing a new timm ViT."""
    model.proj = nn.Sequential(
        nn.Linear(int(arch["backbone_output_dim"]), int(arch["projector_hidden_dim"])),
        nn.BatchNorm1d(int(arch["projector_hidden_dim"])),
        nn.ReLU(inplace=True),
        nn.Linear(int(arch["projector_hidden_dim"]), int(arch["projector_hidden_dim"])),
        nn.BatchNorm1d(int(arch["projector_hidden_dim"])),
        nn.ReLU(inplace=True),
        nn.Linear(int(arch["projector_hidden_dim"]), int(arch["projection_dim"])),
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def build_model(payload: dict[str, Any], device: torch.device) -> tuple[ViTEncoder, dict[str, Any], dict[str, Any]]:
    arch = infer_architecture(payload)
    model = ViTEncoder(arch).to(device)
    load_info = load_weights_into_existing_model(model, payload, "Primary")
    return model, arch, load_info


# -----------------------------------------------------------------------------
# Mammography preprocessing / augmentation
# -----------------------------------------------------------------------------

def deep_get(dct: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class ConfigurableMGAugmentation(nn.Module):
    _warned_no_cv2 = False

    def __init__(self, aug_cfg: dict[str, Any], image_size: int, train: bool):
        super().__init__()
        self.cfg = aug_cfg or {}
        self.image_size = int(image_size)
        self.train = bool(train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._foreground_crop(x)
        x = self._mask_top_corner(x)

        if self.train:
            pre_resize_cfg = deep_get(self.cfg, ["preprocessing", "resize_after_foreground_crop"], {})
            if pre_resize_cfg.get("enabled", True):
                x = self._resize(x, int(pre_resize_cfg.get("size", max(self.image_size, 256))))
            else:
                x = self._resize(x, self.image_size)
            x = self._random_resized_crop(x)
            x = self._horizontal_flip(x)
            x = self._vertical_flip(x)
            x = self._large_rotation(x)
            x = self._random_affine(x)
            x = self._gamma(x)
            x = self._brightness_contrast(x)
            x = self._noise(x)
            x = self._blur(x)
            x = self._sharpen(x)
            x = self._histogram_equalization(x)
            x = self._clahe(x)
            x = self._intensity_inversion(x)
            x = self._posterization(x)
            x = self._random_erasing(x)
            x = self._cutout(x)
            return x.float().clamp(0, 1)

        return self._resize(x, self.image_size).float().clamp(0, 1)

    def _foreground_crop(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["preprocessing", "foreground_crop"], {})
        if not c.get("enabled", False):
            return x
        threshold = float(c.get("threshold_abs", 1e-6))
        margin_frac = float(c.get("margin_frac", 0.05))
        min_area_frac = float(c.get("min_foreground_area_frac", 0.01))
        fallback = bool(c.get("fallback_to_original", True))
        mask = x[0] > threshold
        ys, xs = torch.where(mask)
        h, w = x.shape[-2:]
        if len(xs) < int(h * w * min_area_frac):
            return x if fallback else x[:, :h, :w]
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mh = int((y1 - y0) * margin_frac)
        mw = int((x1 - x0) * margin_frac)
        return x[:, max(0, y0 - mh):min(h, y1 + mh), max(0, x0 - mw):min(w, x1 + mw)]

    def _mask_top_corner(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["preprocessing", "top_right_corner_mask"], {})
        if not c.get("enabled", False):
            return x
        frac_x = float(c.get("frac_x", 0.30))
        frac_y = float(c.get("frac_y", 0.12))
        value = float(c.get("value", 0.0))
        foreground_threshold = float(c.get("foreground_threshold", 1e-4))
        min_component_area_frac = float(c.get("min_component_area_frac", 0.0002))
        skip_if_single_component = bool(c.get("skip_if_single_component", True))

        _, h, w = x.shape
        mh = max(1, int(round(h * frac_y)))
        mw = max(1, int(round(w * frac_x)))
        foreground = x[0] > foreground_threshold

        if skip_if_single_component:
            try:
                import cv2  # type: ignore
                mask_np = foreground.detach().cpu().numpy().astype("uint8")
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
                min_area = max(1, int(round(h * w * min_component_area_frac)))
                relevant = 0
                for label_idx in range(1, num_labels):
                    if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_area:
                        relevant += 1
                if relevant <= 1:
                    return x
            except Exception:
                pass

        left = foreground[:, : w // 2].float().sum().item()
        right = foreground[:, w // 2:].float().sum().item()
        x = x.clone()
        if left <= right:
            x[:, :mh, :mw] = value
        else:
            x[:, :mh, w - mw:] = value
        return x

    @staticmethod
    def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
        return TF.resize(x, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _random_resized_crop(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "random_resized_crop"], {})
        if not c.get("enabled", False):
            return self._resize(x, self.image_size)
        scale = tuple(c.get("scale", [0.85, 1.0]))
        ratio = tuple(c.get("ratio", [0.9, 1.1]))
        size = int(c.get("size", self.image_size))
        i, j, h, w = RandomResizedCrop.get_params(x, scale=scale, ratio=ratio)
        return TF.resized_crop(
            x, i, j, h, w, [size, size],
            interpolation=InterpolationMode.BILINEAR, antialias=True
        )

    def _horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "horizontal_flip"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.5)):
            return TF.hflip(x)
        return x

    def _vertical_flip(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "vertical_flip"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.0)):
            return TF.vflip(x)
        return x

    def _large_rotation(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "large_rotation_90_180"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.0)):
            angle = random.choice(c.get("angles", [90, 180, 270]))
            return TF.rotate(x, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        return x

    def _random_affine(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["spatial", "random_affine"], {})
        if not c.get("enabled", False) or random.random() > float(c.get("p", 0.5)):
            return x
        degrees = float(c.get("degrees", 3.0))
        tr = c.get("translate", [0.02, 0.02])
        sc = c.get("scale", [0.97, 1.03])
        sh = c.get("shear", [0.0, 0.0])
        angle = random.uniform(-degrees, degrees)
        h, w = x.shape[-2:]
        tx = int(random.uniform(-float(tr[0]), float(tr[0])) * w)
        ty = int(random.uniform(-float(tr[1]), float(tr[1])) * h)
        scale = random.uniform(float(sc[0]), float(sc[1]))
        shear = [random.uniform(float(sh[0]), float(sh[1])), 0.0]
        return TF.affine(
            x, angle=angle, translate=[tx, ty], scale=scale, shear=shear,
            interpolation=InterpolationMode.BILINEAR, fill=[float(c.get("fill", 0.0))]
        )

    def _gamma(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "random_gamma"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.5)):
            g = c.get("gamma", [0.9, 1.1])
            return x.clamp(0, 1).pow(random.uniform(float(g[0]), float(g[1])))
        return x

    def _brightness_contrast(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "brightness_contrast"], {})
        if not c.get("enabled", False) or random.random() > float(c.get("p", 0.5)):
            return x
        b = c.get("brightness", [0.95, 1.05])
        co = c.get("contrast", [0.9, 1.1])
        brightness = random.uniform(float(b[0]), float(b[1]))
        contrast = random.uniform(float(co[0]), float(co[1]))
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return ((x - mean) * contrast + mean).mul(brightness).clamp(0, 1)

    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "gaussian_noise"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.2)):
            sr = c.get("std", [0.0, 0.01])
            std = random.uniform(float(sr[0]), float(sr[1]))
            out = x + torch.randn_like(x) * std
            return out.clamp(0, 1) if c.get("clip", True) else out
        return x

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "gaussian_blur"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.1)):
            k = int(c.get("kernel_size", 3))
            if k % 2 == 0:
                k += 1
            return TF.gaussian_blur(x, kernel_size=[k, k], sigma=tuple(c.get("sigma", [0.1, 0.6])))
        return x

    def _sharpen(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "sharpen"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 0.0)):
            factors = c.get("sharpness_factor", [1.0, 1.2])
            factor = random.uniform(float(factors[0]), float(factors[1]))
            return TF.adjust_sharpness(x, sharpness_factor=factor).clamp(0, 1)
        return x

    @staticmethod
    def _to_uint8(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(0, 1) * 255.0).round().to(torch.uint8)

    @staticmethod
    def _from_uint8(x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0

    def _histogram_equalization(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "histogram_equalization"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 1.0)):
            return self._from_uint8(TF.equalize(self._to_uint8(x))).clamp(0, 1)
        return x

    def _clahe(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "clahe"], {})
        if not (c.get("enabled", False) and random.random() < float(c.get("p", 1.0))):
            return x
        try:
            import cv2  # type: ignore
            arr = (x.squeeze(0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=float(c.get("clip_limit", 2.0)),
                tileGridSize=tuple(c.get("tile_grid_size", [8, 8])),
            )
            out = clahe.apply(arr).astype(np.float32) / 255.0
            return torch.from_numpy(out).unsqueeze(0).to(dtype=x.dtype)
        except Exception:
            if not ConfigurableMGAugmentation._warned_no_cv2:
                print("WARNING: CLAHE requested but OpenCV/cv2 is unavailable or failed. Skipping CLAHE.")
                ConfigurableMGAugmentation._warned_no_cv2 = True
            return x

    def _intensity_inversion(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "intensity_inversion"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 1.0)):
            return 1.0 - x
        return x

    def _posterization(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["intensity", "posterization"], {})
        if c.get("enabled", False) and random.random() < float(c.get("p", 1.0)):
            bits = max(1, min(8, int(c.get("bits", 6))))
            return self._from_uint8(TF.posterize(self._to_uint8(x), bits=bits)).clamp(0, 1)
        return x

    def _random_erasing(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["occlusion", "random_erasing"], {})
        if not (c.get("enabled", False) and random.random() < float(c.get("p", 0.0))):
            return x
        scale = c.get("scale", [0.01, 0.03])
        ratio = c.get("ratio", [0.3, 3.3])
        value = float(c.get("value", 0.0))
        _, h, w = x.shape
        area = h * w
        for _ in range(10):
            target = random.uniform(float(scale[0]), float(scale[1])) * area
            aspect = math.exp(random.uniform(math.log(float(ratio[0])), math.log(float(ratio[1]))))
            erase_h = int(round(math.sqrt(target * aspect)))
            erase_w = int(round(math.sqrt(target / aspect)))
            if erase_h < h and erase_w < w:
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)
                x = x.clone()
                x[:, i:i + erase_h, j:j + erase_w] = value
                return x
        return x

    def _cutout(self, x: torch.Tensor) -> torch.Tensor:
        c = deep_get(self.cfg, ["occlusion", "cutout"], {})
        if not (c.get("enabled", False) and random.random() < float(c.get("p", 0.0))):
            return x
        size_frac = float(c.get("size_frac", 0.05))
        value = float(c.get("value", 0.0))
        _, h, w = x.shape
        ch = max(1, int(h * size_frac))
        cw = max(1, int(w * size_frac))
        i = random.randint(0, max(0, h - ch))
        j = random.randint(0, max(0, w - cw))
        x = x.clone()
        x[:, i:i + ch, j:j + cw] = value
        return x


class MGAnalysisDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bin_path: str | Path,
        full_num_rows: int,
        image_shape: tuple[int, int],
        dtype: str,
        transform: nn.Module,
        normalize_mode: str,
        percentile_low: float,
        percentile_high: float,
    ):
        self.df = df.reset_index(drop=True)
        self.bin_path = Path(bin_path)
        self.full_num_rows = int(full_num_rows)
        self.image_shape = tuple(image_shape)
        self.dtype = np.dtype(dtype)
        self.transform = transform
        self.normalize_mode = normalize_mode
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
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

    def load_raw_tensor(self, dataset_idx: int) -> torch.Tensor:
        row = self.df.iloc[int(dataset_idx)]
        arr = self._open()[int(row["original_index"])].astype(np.float32)
        if self.normalize_mode == "uint16":
            if self.dtype == np.dtype("uint16"):
                arr = arr / 65535.0
            else:
                arr = arr / max(float(arr.max()), 1.0)
        elif self.normalize_mode == "per_image_percentile":
            lo, hi = np.percentile(arr, [self.percentile_low, self.percentile_high])
            arr = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.load_raw_tensor(int(idx))
        x = self.transform(x)
        return x, torch.tensor(int(idx), dtype=torch.long)


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_representations(
    model: ViTEncoder,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    use_cuda = device.type == "cuda"

    for x, _ in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        with autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            enabled=bool(use_amp and use_cuda),
        ):
            emb, proj = model.encode_one(x)
        embeddings.append(emb.detach().float().cpu().numpy())
        projections.append(proj.detach().float().cpu().numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(projections, axis=0)


# -----------------------------------------------------------------------------
# Representation statistics and outlier screening
# -----------------------------------------------------------------------------

def covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros(x.shape[1], dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    return np.maximum(eig, 0.0)[::-1]


def effective_rank_from_eig(eig: np.ndarray) -> float:
    eig = np.asarray(eig, dtype=np.float64)
    total = float(eig.sum())
    if total <= 1e-20:
        return 0.0
    p = eig / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(np.maximum(p, 1e-20))).sum()))


def representation_statistics(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    norms = np.linalg.norm(x, axis=1)
    eig = covariance_eigenvalues(x)
    total = float(eig.sum())
    ratios = eig / total if total > 0 else np.zeros_like(eig)

    return {
        "rows": int(n),
        "dimension": int(d),
        "mean_feature_norm": float(norms.mean()) if len(norms) else 0.0,
        "median_feature_norm": float(np.median(norms)) if len(norms) else 0.0,
        "std_feature_norm": float(norms.std()) if len(norms) else 0.0,
        "mean_dimension_std": float(x.std(axis=0, ddof=1).mean()) if n > 1 else 0.0,
        "effective_rank": effective_rank_from_eig(eig),
        "normalized_effective_rank": effective_rank_from_eig(eig) / max(1, d),
        "pc1_variance_pct": float(100.0 * ratios[0]) if len(ratios) > 0 else 0.0,
        "pc2_variance_pct": float(100.0 * ratios[1]) if len(ratios) > 1 else 0.0,
        "pc1_pc2_variance_pct": float(100.0 * ratios[:2].sum()),
        "top5_variance_pct": float(100.0 * ratios[:5].sum()),
        "largest_cov_eigenvalue": float(eig[0]) if len(eig) else 0.0,
        "covariance_trace": total,
    }


@dataclass
class OutlierModel:
    # Local-density / radius model fitted on the dominant population.
    pca: PCA
    whiten_scale: np.ndarray
    radius_med: float
    radius_scale: float
    norm_med: float
    norm_scale: float
    knn_med: float
    knn_scale: float
    median_knn_distance: float
    nn_normal: Optional[NearestNeighbors]
    normal_zw: np.ndarray
    normal_indices: np.ndarray
    robust_z_threshold: float
    hard_knn_ratio: float
    knn_k: int

    # Global robust-PCA model. This is specifically intended to detect small,
    # compact islands that are far away from the dominant population.
    global_pca: PCA
    global_pc_median: np.ndarray
    global_pc_scale: np.ndarray
    global_z_threshold: float
    global_pca_components: int


def robust_pc_location_scale(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-PC robust center/scale using median and MAD.

    A standard-deviation fallback is used only for numerically degenerate PCs.
    """
    coords = np.asarray(coords, dtype=np.float64)
    med = np.median(coords, axis=0)
    mad = np.median(np.abs(coords - med[None, :]), axis=0)
    scale = 1.4826 * mad

    std = coords.std(axis=0, ddof=1) if len(coords) > 1 else np.ones(coords.shape[1])
    bad = (~np.isfinite(scale)) | (scale < 1e-10)
    scale[bad] = std[bad]
    bad = (~np.isfinite(scale)) | (scale < 1e-10)
    scale[bad] = 1.0
    return med.astype(np.float64), scale.astype(np.float64)


def fit_iterative_global_pca_outliers(
    x: np.ndarray,
    z_threshold: float,
    max_pca_components: int,
    max_iterations: int,
    max_outlier_fraction: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PCA,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    list[dict[str, Any]],
]:
    """Detect globally displaced samples with iterative robust PCA.

    Why this exists:
    ----------------
    A compact remote island can have perfectly normal local kNN distances because
    its members are near each other. The previous detector therefore missed exactly
    the right-hand PCA island we care about.

    Here PCA supplies directions, but NOT the scale. Along each PC we use the median
    and MAD of the current dominant population. A small remote cluster can therefore
    define/align with PC1 yet still receive an enormous robust z-score instead of
    being normalized away by the large standard deviation that it created.

    The detector is iterative:
      fit on current dominant population -> flag global anomalies -> refit -> repeat.

    The returned mask is the UNION over iterations. The final PCA/median/scale are
    refitted once on the final dominant population and are used for scoring new
    stochastic views.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 10:
        raise ValueError(f"Need more samples for global PCA outlier analysis; got {n}.")
    if not (0.0 < max_outlier_fraction < 1.0):
        raise ValueError("--max-outlier-fraction must be between 0 and 1.")

    global_mask = np.zeros(n, dtype=bool)
    iteration_found = np.full(n, -1, dtype=np.int32)
    capped = False
    max_allowed = max(1, int(math.floor(max_outlier_fraction * n)))
    iteration_history: list[dict[str, Any]] = []

    last_score = np.zeros(n, dtype=np.float64)
    last_dom_pc = np.zeros(n, dtype=np.int32)
    last_signed_z = np.zeros(n, dtype=np.float64)

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        ref_idx = np.where(~global_mask)[0]
        if len(ref_idx) < 3:
            break

        n_components = max(
            2,
            min(int(max_pca_components), d, len(ref_idx) - 1),
        )
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=seed + iteration - 1,
        )
        ref_coords = pca.fit_transform(x[ref_idx])
        pc_med, pc_scale = robust_pc_location_scale(ref_coords)

        all_coords = pca.transform(x)
        signed_z = (all_coords - pc_med[None, :]) / pc_scale[None, :]
        abs_z = np.abs(signed_z)
        score = abs_z.max(axis=1)
        dom_pc = abs_z.argmax(axis=1).astype(np.int32) + 1
        dom_signed_z = signed_z[np.arange(n), dom_pc - 1]

        candidate = score >= float(z_threshold)
        new_mask = candidate & ~global_mask
        new_idx = np.where(new_mask)[0]

        last_score = score
        last_dom_pc = dom_pc
        last_signed_z = dom_signed_z

        raw_new_count = int(len(new_idx))
        cumulative_before = int(global_mask.sum())
        remaining_capacity = max_allowed - cumulative_before

        if raw_new_count == 0:
            iteration_history.append({
                "iteration": int(iteration),
                "reference_rows": int(len(ref_idx)),
                "candidates_above_threshold_total": int(candidate.sum()),
                "new_candidates_before_cap": 0,
                "accepted_new_outliers": 0,
                "cumulative_outliers": cumulative_before,
                "cap_hit_this_iteration": False,
                "max_abs_robust_pc_z": float(np.max(score)),
            })
            break

        if remaining_capacity <= 0:
            capped = True
            iteration_history.append({
                "iteration": int(iteration),
                "reference_rows": int(len(ref_idx)),
                "candidates_above_threshold_total": int(candidate.sum()),
                "new_candidates_before_cap": raw_new_count,
                "accepted_new_outliers": 0,
                "cumulative_outliers": cumulative_before,
                "cap_hit_this_iteration": True,
                "max_abs_robust_pc_z": float(np.max(score)),
            })
            break

        accepted_before_cap = raw_new_count
        cap_hit_this_iteration = False
        if raw_new_count > remaining_capacity:
            # Safety valve: keep only the strongest newly found points.
            order = new_idx[np.argsort(score[new_idx])[::-1]]
            new_idx = order[:remaining_capacity]
            new_mask = np.zeros(n, dtype=bool)
            new_mask[new_idx] = True
            capped = True
            cap_hit_this_iteration = True

        global_mask[new_idx] = True
        iteration_found[new_idx] = iteration

        iteration_history.append({
            "iteration": int(iteration),
            "reference_rows": int(len(ref_idx)),
            "candidates_above_threshold_total": int(candidate.sum()),
            "new_candidates_before_cap": int(accepted_before_cap),
            "accepted_new_outliers": int(len(new_idx)),
            "cumulative_outliers": int(global_mask.sum()),
            "cap_hit_this_iteration": bool(cap_hit_this_iteration),
            "max_abs_robust_pc_z": float(np.max(score)),
        })

        if capped:
            break

    # Final global model on the final dominant population.
    ref_idx = np.where(~global_mask)[0]
    if len(ref_idx) < 3:
        ref_idx = np.arange(n)

    n_components = max(2, min(int(max_pca_components), d, len(ref_idx) - 1))
    final_pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed + 1009,
    )
    ref_coords = final_pca.fit_transform(x[ref_idx])
    final_med, final_scale = robust_pc_location_scale(ref_coords)
    all_coords = final_pca.transform(x)
    final_signed_z_all = (all_coords - final_med[None, :]) / final_scale[None, :]
    final_abs_z = np.abs(final_signed_z_all)
    final_score = final_abs_z.max(axis=1)
    final_dom_pc = final_abs_z.argmax(axis=1).astype(np.int32) + 1
    final_dom_signed_z = final_signed_z_all[np.arange(n), final_dom_pc - 1]

    # Keep the iterative UNION as the actual decision mask. Final scores are for
    # interpretability/ranking and may differ slightly after the final refit.
    return (
        global_mask,
        final_score,
        final_dom_pc,
        final_dom_signed_z,
        final_pca,
        final_med,
        final_scale,
        iteration_found,
        capped,
        iteration_history,
    )


def _mean_knn_distance_to_reference(
    query_zw: np.ndarray,
    reference_zw: np.ndarray,
    query_original_indices: Optional[np.ndarray],
    reference_original_indices: np.ndarray,
    knn_k: int,
) -> tuple[np.ndarray, NearestNeighbors]:
    """Mean distance to the dominant/reference population.

    For reference samples themselves, remove their zero-distance self-neighbour.
    For already-global-outlier samples, use the k nearest dominant samples directly.
    """
    query_zw = np.asarray(query_zw, dtype=np.float64)
    reference_zw = np.asarray(reference_zw, dtype=np.float64)

    if len(reference_zw) == 0:
        raise ValueError("No reference samples remain for kNN outlier scoring.")

    n_neighbors = min(max(1, int(knn_k)) + 1, len(reference_zw))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(reference_zw)
    distances, indices = nn.kneighbors(query_zw)

    out = np.empty(len(query_zw), dtype=np.float64)
    ref_pos = {
        int(orig_idx): pos
        for pos, orig_idx in enumerate(reference_original_indices.tolist())
    }

    for i in range(len(query_zw)):
        d = distances[i]
        if query_original_indices is not None:
            orig = int(query_original_indices[i])
            own_ref_pos = ref_pos.get(orig)
        else:
            own_ref_pos = None

        if own_ref_pos is not None:
            # Remove the exact self-neighbour if present. Do not assume it is always
            # column zero; numerical ties can occasionally reorder neighbours.
            neigh_ref_positions = indices[i]
            keep = neigh_ref_positions != own_ref_pos
            d_use = d[keep][: int(knn_k)]
        else:
            d_use = d[: int(knn_k)]

        if len(d_use) == 0:
            out[i] = 0.0
        else:
            out[i] = float(np.mean(d_use))

    return out, nn


def fit_outlier_screen(
    x: np.ndarray,
    robust_z_threshold: float,
    hard_knn_ratio: float,
    knn_k: int,
    max_pca_components: int,
    seed: int,
    global_z_threshold: float,
    global_pca_components: int,
    global_max_iterations: int,
    max_outlier_fraction: float,
) -> tuple[pd.DataFrame, OutlierModel, np.ndarray, list[dict[str, Any]]]:
    """Hybrid global + local detector.

    GLOBAL branch:
      iterative robust PCA with per-PC median/MAD scaling.
      Detects compact but remote islands.

    LOCAL branch:
      radius / norm / kNN diagnostics fitted on the global-dominant population.
      Detects isolated bridge points and other sparse anomalies.

    Final outlier = global OR local.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < max(knn_k + 3, 10):
        raise ValueError(f"Need more samples for outlier analysis; got {n}.")

    (
        global_mask,
        global_score,
        global_dom_pc,
        global_dom_signed_z,
        global_pca,
        global_pc_med,
        global_pc_scale,
        iteration_found,
        global_capped,
        global_iteration_history,
    ) = fit_iterative_global_pca_outliers(
        x=x,
        z_threshold=global_z_threshold,
        max_pca_components=global_pca_components,
        max_iterations=global_max_iterations,
        max_outlier_fraction=max_outlier_fraction,
        seed=seed,
    )

    # Fit local geometry only on the dominant population found by the global branch.
    dominant_indices = np.where(~global_mask)[0]
    if len(dominant_indices) < max(knn_k + 3, 10):
        dominant_indices = np.arange(n)

    n_components = max(
        2,
        min(int(max_pca_components), d, len(dominant_indices) - 1),
    )
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
    )
    z_ref = pca.fit_transform(x[dominant_indices])
    z_all = pca.transform(x)
    whiten_scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
    zw_ref = z_ref / whiten_scale[None, :]
    zw_all = z_all / whiten_scale[None, :]

    radius_all = np.linalg.norm(zw_all, axis=1)
    norms_all = np.linalg.norm(x, axis=1)

    mean_knn_all, _ = _mean_knn_distance_to_reference(
        query_zw=zw_all,
        reference_zw=zw_ref,
        query_original_indices=np.arange(n, dtype=np.int64),
        reference_original_indices=dominant_indices.astype(np.int64),
        knn_k=knn_k,
    )

    # Robust baselines use only the dominant/reference population.
    radius_med, radius_scale = robust_location_scale(radius_all[dominant_indices])
    norm_med, norm_scale = robust_location_scale(norms_all[dominant_indices])
    knn_med, knn_scale = robust_location_scale(mean_knn_all[dominant_indices])

    radius_z = robust_z(radius_all, radius_med, radius_scale)
    norm_z = robust_z(norms_all, norm_med, norm_scale)
    knn_z = robust_z(mean_knn_all, knn_med, knn_scale)
    knn_ratio = mean_knn_all / max(knn_med, 1e-12)

    votes = (
        (radius_z >= robust_z_threshold).astype(np.int32)
        + (norm_z >= robust_z_threshold).astype(np.int32)
        + (knn_z >= robust_z_threshold).astype(np.int32)
    )
    local_strong = (votes >= 2) | (knn_ratio >= hard_knn_ratio)
    strong = global_mask | local_strong

    local_score = np.maximum.reduce([
        np.maximum(radius_z, 0.0),
        np.maximum(norm_z, 0.0),
        np.maximum(knn_z, 0.0),
        np.log2(np.maximum(knn_ratio, 1.0)) * 2.0,
    ])
    score = np.maximum(local_score, global_score)
    pct = percentile_ranks(score)

    # Final nearest-normal model uses the FINAL hybrid mask, not only the global mask.
    normal_indices = np.where(~strong)[0]
    nearest_normal_idx = np.full(n, -1, dtype=np.int64)
    nearest_normal_distance = np.full(n, np.nan, dtype=np.float64)
    relative_normal_separation = np.full(n, np.nan, dtype=np.float64)
    nn_normal = None
    normal_zw = np.empty((0, zw_all.shape[1]), dtype=np.float64)

    if len(normal_indices) > 0:
        normal_zw = zw_all[normal_indices]
        nn_normal = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        nn_normal.fit(normal_zw)
        q_dist, q_ind = nn_normal.kneighbors(zw_all)
        nearest_normal_idx = normal_indices[q_ind[:, 0]]
        nearest_normal_distance = q_dist[:, 0]
        relative_normal_separation = nearest_normal_distance / max(knn_med, 1e-12)

    # Record which global iteration found each point by replaying only for the final
    # mask is unnecessary for decisions; -1 means "not global". For the report we use
    # a binary global flag and final robust-PC score/PC index.
    metrics = pd.DataFrame({
        "global_is_outlier": global_mask,
        "global_iteration_found": iteration_found,
        "global_max_abs_pc_z": global_score,
        "global_dominant_pc": global_dom_pc,
        "global_dominant_signed_z": global_dom_signed_z,
        "global_detection_capped": np.full(n, bool(global_capped)),
        "local_is_outlier": local_strong,
        "radius": radius_all,
        "radius_robust_z": radius_z,
        "feature_norm": norms_all,
        "norm_robust_z": norm_z,
        "mean_knn_distance": mean_knn_all,
        "knn_robust_z": knn_z,
        "knn_ratio_to_median": knn_ratio,
        "vote_count": votes,
        "outlier_score": score,
        "outlier_score_percentile": pct,
        "is_outlier": strong,
        "nearest_normal_index": nearest_normal_idx,
        "nearest_normal_distance": nearest_normal_distance,
        "relative_normal_separation": relative_normal_separation,
    })

    model = OutlierModel(
        pca=pca,
        whiten_scale=whiten_scale,
        radius_med=radius_med,
        radius_scale=radius_scale,
        norm_med=norm_med,
        norm_scale=norm_scale,
        knn_med=knn_med,
        knn_scale=knn_scale,
        median_knn_distance=float(knn_med),
        nn_normal=nn_normal,
        normal_zw=normal_zw,
        normal_indices=normal_indices,
        robust_z_threshold=float(robust_z_threshold),
        hard_knn_ratio=float(hard_knn_ratio),
        knn_k=int(knn_k),
        global_pca=global_pca,
        global_pc_median=global_pc_med,
        global_pc_scale=global_pc_scale,
        global_z_threshold=float(global_z_threshold),
        global_pca_components=int(global_pca.n_components_),
    )
    return metrics, model, zw_all, global_iteration_history


def score_new_points(x: np.ndarray, model: OutlierModel) -> pd.DataFrame:
    """Score stochastic/new views against the already fitted hybrid detector."""
    x = np.asarray(x, dtype=np.float64)

    # Global robust-PCA score.
    global_coords = model.global_pca.transform(x)
    global_signed_z = (
        global_coords - model.global_pc_median[None, :]
    ) / model.global_pc_scale[None, :]
    global_abs_z = np.abs(global_signed_z)
    global_score = global_abs_z.max(axis=1)
    global_dom_pc = global_abs_z.argmax(axis=1).astype(np.int32) + 1
    global_dom_signed_z = global_signed_z[
        np.arange(len(x)),
        global_dom_pc - 1,
    ]
    global_strong = global_score >= model.global_z_threshold

    # Local score relative to final normals.
    z = model.pca.transform(x)
    zw = z / model.whiten_scale[None, :]
    radius = np.linalg.norm(zw, axis=1)
    norms = np.linalg.norm(x, axis=1)

    if len(model.normal_zw) > 0:
        n_query = min(max(1, model.knn_k), len(model.normal_zw))
        nn = NearestNeighbors(n_neighbors=n_query, metric="euclidean", n_jobs=-1)
        nn.fit(model.normal_zw)
        d, _ = nn.kneighbors(zw)
        mean_knn = d.mean(axis=1)
    else:
        mean_knn = np.zeros(len(x), dtype=np.float64)

    radius_z = robust_z(radius, model.radius_med, model.radius_scale)
    norm_z = robust_z(norms, model.norm_med, model.norm_scale)
    knn_z = robust_z(mean_knn, model.knn_med, model.knn_scale)
    knn_ratio = mean_knn / max(model.knn_med, 1e-12)

    votes = (
        (radius_z >= model.robust_z_threshold).astype(np.int32)
        + (norm_z >= model.robust_z_threshold).astype(np.int32)
        + (knn_z >= model.robust_z_threshold).astype(np.int32)
    )
    local_strong = (votes >= 2) | (knn_ratio >= model.hard_knn_ratio)
    strong = global_strong | local_strong

    local_score = np.maximum.reduce([
        np.maximum(radius_z, 0.0),
        np.maximum(norm_z, 0.0),
        np.maximum(knn_z, 0.0),
        np.log2(np.maximum(knn_ratio, 1.0)) * 2.0,
    ])
    score = np.maximum(local_score, global_score)

    return pd.DataFrame({
        "global_is_outlier": global_strong,
        "global_max_abs_pc_z": global_score,
        "global_dominant_pc": global_dom_pc,
        "global_dominant_signed_z": global_dom_signed_z,
        "local_is_outlier": local_strong,
        "radius_robust_z": radius_z,
        "norm_robust_z": norm_z,
        "knn_robust_z": knn_z,
        "knn_ratio_to_median": knn_ratio,
        "vote_count": votes,
        "outlier_score": score,
        "is_outlier": strong,
    })


# -----------------------------------------------------------------------------
# Plotting / image serialization
# -----------------------------------------------------------------------------

def figure_to_data_uri(fig: plt.Figure, dpi: int = 140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def tensor_to_data_uri(x: torch.Tensor, title: Optional[str] = None) -> str:
    arr = x.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    return figure_to_data_uri(fig, dpi=120)


def make_pca_colored_plot(
    x: np.ndarray,
    labels: np.ndarray,
    title: str,
    seed: int,
) -> tuple[str, dict[str, float]]:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASS_NAMES:
        mask = labels.astype(str) == label
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=11, alpha=0.65, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{title}\nPC1={100*evr[0]:.2f}%, PC2={100*evr[1]:.2f}%")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return figure_to_data_uri(fig), {
        "pc1_variance_pct": float(100 * evr[0]),
        "pc2_variance_pct": float(100 * evr[1]),
    }


def make_pca_outlier_plot(
    x: np.ndarray,
    is_outlier: np.ndarray,
    title: str,
    seed: int,
    annotate_indices: Optional[list[int]] = None,
) -> str:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    normal = ~is_outlier
    ax.scatter(coords[normal, 0], coords[normal, 1], s=9, alpha=0.35, label="normal")
    if is_outlier.any():
        ax.scatter(coords[is_outlier, 0], coords[is_outlier, 1], s=38, alpha=0.95, label="detected outlier")
    if annotate_indices:
        for rank, idx in enumerate(annotate_indices, start=1):
            ax.annotate(
                f"#{rank}",
                (coords[idx, 0], coords[idx, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{title}\nPC1={100*evr[0]:.2f}%, PC2={100*evr[1]:.2f}%")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return figure_to_data_uri(fig)


def _pca_axis_limits(coords: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = float(coords[:, 0].min()), float(coords[:, 0].max())
    y0, y1 = float(coords[:, 1].min()), float(coords[:, 1].max())
    xpad = max(1e-6, 0.04 * (x1 - x0))
    ypad = max(1e-6, 0.04 * (y1 - y0))
    return (x0 - xpad, x1 + xpad), (y0 - ypad, y1 + ypad)


def make_pca_detection_reason_plot(
    x: np.ndarray,
    metrics: pd.DataFrame,
    title: str,
    seed: int,
    annotate_indices: Optional[list[int]] = None,
) -> str:
    """Show WHY samples were detected, while keeping ordinary PCA coordinates."""
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_
    xlim, ylim = _pca_axis_limits(coords)

    global_mask = metrics["global_is_outlier"].to_numpy(dtype=bool)
    local_mask = metrics["local_is_outlier"].to_numpy(dtype=bool)
    dom_pc = metrics["global_dominant_pc"].to_numpy(dtype=int)

    normal = ~(global_mask | local_mask)
    global_pc12 = global_mask & (dom_pc <= 2)
    global_higher = global_mask & (dom_pc >= 3)
    local_only = local_mask & ~global_mask

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[normal, 0], coords[normal, 1], s=8, alpha=0.24, label="normal")
    if global_pc12.any():
        ax.scatter(
            coords[global_pc12, 0], coords[global_pc12, 1],
            s=34, alpha=0.90, marker="o",
            label="global anomaly driven by PC1/PC2",
        )
    if global_higher.any():
        ax.scatter(
            coords[global_higher, 0], coords[global_higher, 1],
            s=36, alpha=0.90, marker="^",
            label="global anomaly driven by PC3+",
        )
    if local_only.any():
        ax.scatter(
            coords[local_only, 0], coords[local_only, 1],
            s=40, alpha=0.95, marker="x",
            label="local/kNN anomaly only",
        )

    if annotate_indices:
        for rank, idx in enumerate(annotate_indices, start=1):
            ax.annotate(
                f"#{rank}",
                (coords[idx, 0], coords[idx, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"{title}\\nPC1={100*evr[0]:.2f}%, PC2={100*evr[1]:.2f}%"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return figure_to_data_uri(fig)


def make_pca_same_basis_cleaned_plot(
    x: np.ndarray,
    labels: np.ndarray,
    remove_mask: np.ndarray,
    title: str,
    seed: int,
) -> str:
    """Fit PCA on ALL samples, then hide removed samples without refitting PCA.

    This gives a true visual A/B comparison: coordinates and axis limits are exactly
    those of the original PCA, so a far-right survivor cannot disappear simply because
    PCA was refitted.
    """
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_
    xlim, ylim = _pca_axis_limits(coords)
    keep = ~remove_mask

    fig, ax = plt.subplots(figsize=(8, 6))
    kept_labels = labels.astype(str)
    for label in CLASS_NAMES:
        mask = keep & (kept_labels == label)
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=11,
                alpha=0.65,
                label=label,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"{title}\\n"
        f"Original PCA basis retained; PC1={100*evr[0]:.2f}%, PC2={100*evr[1]:.2f}%"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return figure_to_data_uri(fig)


def make_score_histogram(metrics: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = metrics["knn_ratio_to_median"].to_numpy()
    ax.hist(vals[np.isfinite(vals)], bins=60)
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean kNN distance / median mean kNN distance")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def make_global_score_histogram(
    metrics: pd.DataFrame,
    threshold: float,
    title: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = metrics["global_max_abs_pc_z"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    # The extreme tail can make the central distribution unreadable. Plot through
    # the 99.9th percentile while reporting the true maxima numerically elsewhere.
    if len(vals):
        upper = max(float(threshold) * 1.25, float(np.percentile(vals, 99.9)))
        shown = vals[vals <= upper]
    else:
        shown = vals
    ax.hist(shown, bins=60)
    ax.axvline(float(threshold), linestyle="--", linewidth=1.2)
    ax.set_xlabel("Maximum absolute robust PC z-score")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return figure_to_data_uri(fig)


# -----------------------------------------------------------------------------
# Aggregation / report helpers
# -----------------------------------------------------------------------------

def aggregate_outliers(df: pd.DataFrame, is_outlier: np.ndarray, column: str) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    values = df[column].fillna("missing").astype(str)
    tmp = pd.DataFrame({"value": values, "is_outlier": is_outlier.astype(bool)})
    total = tmp.groupby("value").size().rename("total")
    outs = tmp[tmp["is_outlier"]].groupby("value").size().rename("outliers")
    merged = pd.concat([total, outs], axis=1).fillna(0)
    merged["outliers"] = merged["outliers"].astype(int)
    merged["outlier_rate_pct"] = 100.0 * merged["outliers"] / merged["total"].clip(lower=1)
    merged = merged.sort_values(["outliers", "total"], ascending=False)
    rows = []
    for idx, row in merged.iterrows():
        if int(row["outliers"]) <= 0:
            continue
        rows.append({
            column: str(idx),
            "outliers": int(row["outliers"]),
            "total": int(row["total"]),
            "outlier_rate_pct": float(row["outlier_rate_pct"]),
        })
    return rows


def dict_table_html(rows: list[dict[str, Any]], columns: Optional[list[str]] = None) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    if columns is None:
        columns = list(rows[0].keys())
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html_escape(c)}</th>" for c in columns)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                txt = fmt_num(v)
            else:
                txt = html_escape(v)
            parts.append(f"<td>{txt}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def stats_comparison_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "rows",
        "dimension",
        "pc1_variance_pct",
        "pc2_variance_pct",
        "pc1_pc2_variance_pct",
        "top5_variance_pct",
        "effective_rank",
        "normalized_effective_rank",
        "mean_feature_norm",
        "std_feature_norm",
        "mean_dimension_std",
        "largest_cov_eigenvalue",
        "covariance_trace",
    ]
    rows = []
    for k in keys:
        b = before.get(k, np.nan)
        a = after.get(k, np.nan)
        change = None
        try:
            change = float(a) - float(b)
        except Exception:
            pass
        rows.append({"statistic": k, "before": b, "after": a, "change_after_minus_before": change})
    return rows


def metadata_for_row(row: pd.Series) -> dict[str, Any]:
    cols = [
        "id", "patient", "dataset", "machine", "machine_family", "view", "laterality",
        "birads_numeric", "collapsed_birads", "original_birads", "birads", "original_index",
    ]
    out = {}
    for c in cols:
        if c in row.index:
            v = row[c]
            out[c] = "missing" if pd.isna(v) else safe_scalar(v)
    return out


# -----------------------------------------------------------------------------
# Main analysis of one representation
# -----------------------------------------------------------------------------

def analyze_space(
    name: str,
    x: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics, screen_model, _, global_iteration_history = fit_outlier_screen(
        x,
        robust_z_threshold=args.robust_z_threshold,
        hard_knn_ratio=args.hard_knn_ratio,
        knn_k=args.knn_k,
        max_pca_components=args.screen_pca_components,
        seed=args.seed,
        global_z_threshold=args.global_pca_z_threshold,
        global_pca_components=args.global_pca_components,
        global_max_iterations=args.global_pca_max_iterations,
        max_outlier_fraction=args.max_outlier_fraction,
    )

    out_mask = metrics["is_outlier"].to_numpy(dtype=bool)
    before = representation_statistics(x)
    after = representation_statistics(x[~out_mask]) if (~out_mask).sum() >= 3 else {}

    top_indices = (
        metrics.index[out_mask]
        .to_numpy()[np.argsort(metrics.loc[out_mask, "outlier_score"].to_numpy())[::-1]]
        .tolist()
    )

    before_pca_uri, before_pca_stats = make_pca_colored_plot(
        x, labels, f"{name}: PCA before outlier removal", args.seed
    )
    outlier_pca_uri = make_pca_outlier_plot(
        x, out_mask, f"{name}: same PCA, detected outliers highlighted", args.seed,
        annotate_indices=top_indices[: args.max_details],
    )
    reason_pca_uri = make_pca_detection_reason_plot(
        x,
        metrics,
        f"{name}: detection reasons in original PCA coordinates",
        args.seed,
        annotate_indices=top_indices[: args.max_details],
    )

    after_pca_uri = None
    after_pca_stats = None
    if (~out_mask).sum() >= 3:
        after_pca_uri, after_pca_stats = make_pca_colored_plot(
            x[~out_mask],
            labels[~out_mask],
            f"{name}: PCA after removing ALL {int(out_mask.sum())} detected outliers",
            args.seed,
        )

    return {
        "name": name,
        "x": x,
        "metrics": metrics,
        "screen_model": screen_model,
        "is_outlier": out_mask,
        "num_outliers": int(out_mask.sum()),
        "outlier_fraction": float(out_mask.mean()),
        "top_indices": top_indices,
        "stats_before": before,
        "stats_after": after,
        "before_pca_uri": before_pca_uri,
        "before_pca_stats": before_pca_stats,
        "outlier_pca_uri": outlier_pca_uri,
        "reason_pca_uri": reason_pca_uri,
        "after_pca_uri": after_pca_uri,
        "after_pca_stats": after_pca_stats,
        "score_hist_uri": make_score_histogram(metrics, f"{name}: kNN separation distribution"),
        "global_score_hist_uri": make_global_score_histogram(
            metrics,
            args.global_pca_z_threshold,
            f"{name}: robust global-PCA anomaly score",
        ),
        "global_outlier_count": int(metrics["global_is_outlier"].sum()),
        "local_outlier_count": int(metrics["local_is_outlier"].sum()),
        "global_local_overlap_count": int(
            (metrics["global_is_outlier"] & metrics["local_is_outlier"]).sum()
        ),
        "global_detection_capped": bool(metrics["global_detection_capped"].iloc[0]),
        "max_global_pc_z": float(metrics["global_max_abs_pc_z"].max()),
        "global_iteration_history": global_iteration_history,
        "global_pc12_outlier_count": int(
            (metrics["global_is_outlier"] & (metrics["global_dominant_pc"] <= 2)).sum()
        ),
        "global_higher_pc_outlier_count": int(
            (metrics["global_is_outlier"] & (metrics["global_dominant_pc"] >= 3)).sum()
        ),
        "local_only_outlier_count": int(
            (metrics["local_is_outlier"] & ~metrics["global_is_outlier"]).sum()
        ),
    }


# -----------------------------------------------------------------------------
# Reference-checkpoint comparison
# -----------------------------------------------------------------------------

def reference_comparison(
    primary_outlier_mask: np.ndarray,
    primary_top_indices: list[int],
    reference_space: dict[str, Any],
) -> dict[str, Any]:
    ref_metrics = reference_space["metrics"]
    ref_mask = reference_space["is_outlier"]

    primary_idx = np.where(primary_outlier_mask)[0]
    overlap = int(ref_mask[primary_idx].sum()) if len(primary_idx) else 0

    ref_percentiles = ref_metrics.loc[primary_idx, "outlier_score_percentile"].to_numpy() if len(primary_idx) else np.array([])
    ref_ratios = ref_metrics.loc[primary_idx, "knn_ratio_to_median"].to_numpy() if len(primary_idx) else np.array([])

    details = []
    for idx in primary_top_indices:
        details.append({
            "index": int(idx),
            "reference_detected_outlier": bool(ref_metrics.loc[idx, "is_outlier"]),
            "reference_score_percentile": float(ref_metrics.loc[idx, "outlier_score_percentile"]),
            "reference_knn_ratio": float(ref_metrics.loc[idx, "knn_ratio_to_median"]),
            "reference_relative_normal_separation": float(ref_metrics.loc[idx, "relative_normal_separation"]),
        })

    return {
        "primary_outlier_count": int(len(primary_idx)),
        "also_reference_outlier_count": overlap,
        "also_reference_outlier_fraction": float(overlap / max(1, len(primary_idx))),
        "median_reference_score_percentile_for_primary_outliers": (
            float(np.median(ref_percentiles)) if len(ref_percentiles) else None
        ),
        "median_reference_knn_ratio_for_primary_outliers": (
            float(np.median(ref_ratios)) if len(ref_ratios) else None
        ),
        "top_details": details,
    }


# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------

CSS = """
body { font-family: Arial, Helvetica, sans-serif; margin: 0; color: #1f2937; background: #f4f6f8; }
main { max-width: 1450px; margin: 0 auto; padding: 28px; }
h1, h2, h3, h4 { color: #111827; }
.card { background: white; border: 1px solid #d9dee5; border-radius: 10px; padding: 18px; margin: 18px 0; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; }
img.plot { width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }
img.mammo { width: 100%; max-height: 430px; object-fit: contain; background: #111; border-radius: 5px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #d1d5db; padding: 7px 9px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
.good { color: #166534; font-weight: 700; }
.warn { color: #9a3412; font-weight: 700; }
.bad { color: #991b1b; font-weight: 700; }
.muted { color: #6b7280; }
.metric { font-size: 24px; font-weight: 700; }
.small { font-size: 12px; }
.outlier-card { border-left: 6px solid #b91c1c; }
code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
@media (max-width: 900px) {
  .grid2, .grid3 { grid-template-columns: 1fr; }
}
"""


def report_space_section(space: dict[str, Any], title: str) -> str:
    n_out = space["num_outliers"]
    removed = int(space.get("removed_outliers", n_out))
    cls = "good" if n_out == 0 else "warn"
    highlight_uri = space.get("combined_outlier_pca_uri", space["outlier_pca_uri"])
    s = [
        f"<section class='card'><h2>{html_escape(title)}</h2>",
        f"<p><span class='metric {cls}'>{n_out}</span> strong outlier(s) detected in this representation space "
        f"({100*space['outlier_fraction']:.3f}% of analyzed samples).</p>",
        "<p class='muted small'>Hybrid detector: (1) iterative global PCA with per-component median/MAD scaling "
        "to catch compact remote islands, plus (2) local radius / feature-norm / kNN diagnostics fitted on the "
        "dominant population to catch isolated bridge points. Final outlier = global OR local.</p>",
        "<table>",
        f"<tr><th>Global robust-PCA outliers</th><td>{space['global_outlier_count']}</td></tr>",
        f"<tr><th>Global anomalies driven by PC1/PC2</th><td>{space['global_pc12_outlier_count']}</td></tr>",
        f"<tr><th>Global anomalies driven by PC3+</th><td>{space['global_higher_pc_outlier_count']}</td></tr>",
        f"<tr><th>Local/isolated outliers</th><td>{space['local_outlier_count']}</td></tr>",
        f"<tr><th>Local-only outliers</th><td>{space['local_only_outlier_count']}</td></tr>",
        f"<tr><th>Detected by both global and local criteria</th><td>{space['global_local_overlap_count']}</td></tr>",
        f"<tr><th>Largest global robust-PC z-score</th><td>{fmt_num(space['max_global_pc_z'])}</td></tr>",
        f"<tr><th>Global safety cap reached</th><td>{space['global_detection_capped']}</td></tr>",
        "</table>",
        "<div class='grid2'>",
        f"<div><img class='plot' src='{space['before_pca_uri']}'></div>",
        f"<div><img class='plot' src='{space['reason_pca_uri']}'></div>",
        "</div>",
        "<h3>Global robust-PCA iterations</h3>",
        "<p class='muted small'>This table makes detector runaway visible. "
        "'New candidates before cap' is the number the iteration wanted to add; "
        "'accepted' is the number retained after the safety limit.</p>",
        dict_table_html(space["global_iteration_history"]),
    ]
    if space["after_pca_uri"]:
        s += [
            "<h3>Before / after comparison</h3>",
            f"<p>The final combined detection rule found <b>{removed}</b> outlier(s). "
            f"All {removed} are removed <b>at once</b> for the after-version.</p>",
            "<p class='muted small'><b>Middle:</b> the original PCA basis and axis limits are retained, "
            "so any surviving far-right points remain visibly far right. "
            "<b>Right:</b> PCA is refitted after removal and therefore answers a different question: "
            "what geometry remains once the detected samples are excluded?</p>",
            "<div class='grid3'>",
            f"<div><img class='plot' src='{space['before_pca_uri']}'></div>",
            f"<div><img class='plot' src='{space['same_basis_cleaned_pca_uri']}'></div>",
            f"<div><img class='plot' src='{space['after_pca_uri']}'></div>",
            "</div>",
            dict_table_html(stats_comparison_rows(space["stats_before"], space["stats_after"])),
        ]
    else:
        if removed == 0:
            s.append("<p class='good'>No after-PCA was needed because the final combined rule found no strong outliers.</p>")
        else:
            s.append("<p class='muted'>No after-PCA was produced because fewer than three normal samples remained.</p>")
    s += [
        "<h3>Detector score distributions</h3>",
        "<div class='grid2'>",
        f"<div><img class='plot' src='{space['global_score_hist_uri']}'></div>",
        f"<div><img class='plot' src='{space['score_hist_uri']}'></div>",
        "</div>",
        "</section>",
    ]
    return "".join(s)


def build_report(
    args: argparse.Namespace,
    checkpoint: Path,
    reference_checkpoint: Optional[Path],
    arch: dict[str, Any],
    reference_arch: Optional[dict[str, Any]],
    split_df: pd.DataFrame,
    spaces: dict[str, dict[str, Any]],
    combined_mask: np.ndarray,
    top_combined_indices: list[int],
    detailed_cards: list[str],
    aggregate_tables: dict[str, list[dict[str, Any]]],
    reference_summary: Optional[dict[str, Any]],
    runtime_sec: float,
) -> str:
    n = len(split_df)
    n_out = int(combined_mask.sum())
    top_n = min(args.max_details, n_out)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>MedJEPA PCA outlier analysis</title>",
        f"<style>{CSS}</style></head><body><main>",
        "<h1>MedJEPA PCA / representation outlier analysis</h1>",
        "<section class='card'>",
        "<h2>Run summary</h2>",
        "<table>",
        f"<tr><th>Primary checkpoint</th><td>{html_escape(checkpoint)}</td></tr>",
        f"<tr><th>Reference checkpoint</th><td>{html_escape(reference_checkpoint) if reference_checkpoint else 'not provided'}</td></tr>",
        f"<tr><th>Split</th><td>{html_escape(args.split)}</td></tr>",
        f"<tr><th>Analyzed samples</th><td>{n:,}</td></tr>",
        f"<tr><th>Combined outlier rule</th><td>{html_escape(args.outlier_space)}</td></tr>",
        f"<tr><th>Combined detected outliers</th><td><b>{n_out}</b> ({100*n_out/max(1,n):.3f}%)</td></tr>",
        f"<tr><th>Detailed outliers shown</th><td>{top_n} / {n_out}</td></tr>",
        f"<tr><th>Global robust-PCA z threshold</th><td>{args.global_pca_z_threshold}</td></tr>",
        f"<tr><th>Global PCA components</th><td>{args.global_pca_components}</td></tr>",
        f"<tr><th>Global PCA max iterations</th><td>{args.global_pca_max_iterations}</td></tr>",
        f"<tr><th>Global safety cap</th><td>{100*args.max_outlier_fraction:.2f}% of samples</td></tr>",
        f"<tr><th>Local robust-z threshold</th><td>{args.robust_z_threshold}</td></tr>",
        f"<tr><th>Hard kNN-ratio threshold</th><td>{args.hard_knn_ratio}× median</td></tr>",
        f"<tr><th>k for kNN</th><td>{args.knn_k}</td></tr>",
        f"<tr><th>Runtime</th><td>{runtime_sec/60:.2f} min</td></tr>",
        "</table>",
        "<h3>Checkpoint architecture / training metadata</h3>",
        dict_table_html([arch]),
        "</section>",
    ]

    for name, space in spaces.items():
        parts.append(report_space_section(space, name))

    parts += [
        "<section class='card'><h2>Combined outlier population</h2>",
        f"<p>The detailed gallery below is capped at <b>{args.max_details}</b> examples. "
        "All remaining detected outliers appear only through the aggregate statistics in this section.</p>",
    ]

    if n_out == 0:
        parts += [
            "<p class='good'>No strong outliers were detected by either the global robust-PCA or local detector. "
            "The expensive detailed-outlier and reference comparison stages were skipped.</p>",
            "</section>",
            "</main></body></html>",
        ]
        return "".join(parts)

    for key, rows in aggregate_tables.items():
        parts += [f"<h3>Outliers aggregated by {html_escape(key)}</h3>", dict_table_html(rows)]
    parts.append("</section>")

    if reference_summary is not None:
        parts += [
            "<section class='card'><h2>Reference-checkpoint comparison</h2>",
            f"<p>The reference checkpoint is evaluated on the <b>same image rows</b>. "
            "Its own deterministic evaluation preprocessing is used, matching that checkpoint's saved augmentation configuration.</p>",
            "<table>",
            f"<tr><th>Reference checkpoint</th><td>{html_escape(reference_checkpoint)}</td></tr>",
            f"<tr><th>Primary combined outliers</th><td>{reference_summary['primary_outlier_count']}</td></tr>",
            f"<tr><th>Also detected as reference outliers</th><td>{reference_summary['also_reference_outlier_count']} "
            f"({100*reference_summary['also_reference_outlier_fraction']:.2f}%)</td></tr>",
            f"<tr><th>Median reference outlier-score percentile of primary outliers</th>"
            f"<td>{fmt_num(reference_summary['median_reference_score_percentile_for_primary_outliers'])}%</td></tr>",
            f"<tr><th>Median reference kNN ratio of primary outliers</th>"
            f"<td>{fmt_num(reference_summary['median_reference_knn_ratio_for_primary_outliers'])}×</td></tr>",
            "</table>",
        ]
        if reference_arch:
            parts += ["<h3>Reference architecture / metadata</h3>", dict_table_html([reference_arch])]
        if reference_summary.get("highlight_plot_uri"):
            parts += [
                "<h3>Reference PCA with primary outliers highlighted</h3>",
                f"<img class='plot' style='max-width:950px' src='{reference_summary['highlight_plot_uri']}'>",
            ]
        parts.append("</section>")

    parts += [
        "<section class='card'><h2>Detailed outlier gallery</h2>",
        f"<p>Showing the {top_n} highest-ranked combined outliers only. "
        "No individual details for the remaining outliers are included.</p>",
        "".join(detailed_cards),
        "</section>",
        "<section class='card'><h2>Interpretation guide</h2>",
        "<p><b>Large change after removing all outliers:</b> the visible PCA anisotropy was substantially driven by a small pathological population.</p>",
        "<p><b>Little change after removing all outliers:</b> the representation geometry is globally anisotropic/collapsed; the outliers are more likely a symptom than the sole cause.</p>",
        "<p><b>Primary outliers become normal under the reference checkpoint:</b> evidence for model/loss-induced pathology rather than unusual raw data alone.</p>",
        "<p><b>Only some stochastic SSL views become extreme:</b> evidence for an augmentation × model interaction. "
        "If every stochastic view remains extreme, the underlying image/domain is more suspicious.</p>",
        "</section>",
        "</main></body></html>",
    ]
    return "".join(parts)


# -----------------------------------------------------------------------------
# CLI / orchestration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedJEPA PCA / representation outlier analysis with self-contained HTML report.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--reference-checkpoint", default=None, type=Path)
    p.add_argument("--output-dir", default=None, type=Path)

    p.add_argument("--split", choices=["train", "val", "test", "full"], default="test")
    p.add_argument("--max-details", type=int, default=10)
    p.add_argument("--num-aug-views", type=int, default=4)

    p.add_argument(
        "--outlier-space",
        choices=["backbone", "projector", "union", "intersection"],
        default="union",
        help="How backbone/projector outlier flags are combined for the final removal/gallery.",
    )
    p.add_argument(
        "--global-pca-z-threshold",
        type=float,
        default=8.0,
        help="Flag a global PCA anomaly when max |robust PC z| reaches this threshold.",
    )
    p.add_argument(
        "--global-pca-components",
        type=int,
        default=30,
        help="Number of leading PCs used by the robust global-island detector.",
    )
    p.add_argument(
        "--global-pca-max-iterations",
        type=int,
        default=4,
        help="Maximum robust-PCA refit iterations for discovering compact remote groups.",
    )
    p.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=0.05,
        help="Safety cap on the fraction of samples that the global detector may remove.",
    )
    p.add_argument(
        "--robust-z-threshold",
        type=float,
        default=6.0,
        help="Threshold for the local radius/norm/kNN robust-z diagnostics.",
    )
    p.add_argument("--hard-knn-ratio", type=float, default=8.0)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--screen-pca-components", type=int, default=50)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Optional path overrides. Normally everything is inferred from the checkpoint.
    p.add_argument("--full-csv", type=Path, default=None)
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=None)
    p.add_argument("--bin", dest="bin_path", type=Path, default=None)
    return p.parse_args()


def resolve_paths(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Path]:
    def pick(override: Optional[Path], *keys: str) -> Optional[Path]:
        if override is not None:
            return Path(override)
        v = cfg_get(cfg, *keys, default=None)
        return Path(v) if v else None

    paths = {
        "full_csv": pick(args.full_csv, "full_csv_path", "full_csv"),
        "train_csv": pick(args.train_csv, "train_csv_path", "train_csv"),
        "val_csv": pick(args.val_csv, "val_csv_path", "val_csv"),
        "test_csv": pick(args.test_csv, "test_csv_path", "test_csv"),
        "bin": pick(args.bin_path, "bin_path", "bin"),
    }

    required = ["full_csv", "bin"]
    if args.split != "full":
        required.append(f"{args.split}_csv")

    missing = [k for k in required if paths.get(k) is None]
    if missing:
        raise ValueError(
            f"Could not infer required path(s) {missing} from the checkpoint. "
            "Provide the corresponding CLI override(s)."
        )

    for k, v in paths.items():
        if v is not None and k in required and not v.exists():
            raise FileNotFoundError(f"{k} not found: {v}")
    return paths  # type: ignore[return-value]


def load_analysis_df(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_raw = read_csv_clean(paths["full_csv"])
    full_df = add_metadata(prepare_labels(full_raw.reset_index(drop=False).rename(columns={"index": "original_index"})))

    if args.split == "full":
        split_df = full_df.copy()
    else:
        split_path = paths[f"{args.split}_csv"]
        split_raw = read_csv_clean(split_path)
        split_raw = ensure_original_index(split_raw, full_raw, args.split)
        split_df = add_metadata(prepare_labels(split_raw))

    split_df = split_df.reset_index(drop=True)
    return full_raw, split_df


def make_dataset(
    split_df: pd.DataFrame,
    full_num_rows: int,
    bin_path: Path,
    arch: dict[str, Any],
    aug_cfg: dict[str, Any],
    train: bool,
) -> MGAnalysisDataset:
    transform = ConfigurableMGAugmentation(aug_cfg, int(arch["image_size"]), train=train)
    return MGAnalysisDataset(
        split_df,
        bin_path,
        full_num_rows,
        (int(arch["image_height"]), int(arch["image_width"])),
        str(arch["memmap_dtype"]),
        transform,
        str(arch["normalize_mode"]),
        float(arch["percentile_low"]),
        float(arch["percentile_high"]),
    )


def verify_bin(full_num_rows: int, bin_path: Path, arch: dict[str, Any]) -> None:
    expected = (
        full_num_rows
        * int(arch["image_height"])
        * int(arch["image_width"])
        * np.dtype(str(arch["memmap_dtype"])).itemsize
    )
    actual = bin_path.stat().st_size
    if expected != actual:
        raise RuntimeError(
            "Full CSV / BIN size mismatch.\n"
            f"Expected {expected:,} bytes from {full_num_rows} rows and shape "
            f"{arch['image_height']}x{arch['image_width']} {arch['memmap_dtype']}, "
            f"but BIN has {actual:,} bytes."
        )


def combined_outlier_mask(spaces: dict[str, dict[str, Any]], mode: str) -> np.ndarray:
    b = spaces["Backbone embedding"]["is_outlier"]
    p = spaces["Raw projector output"]["is_outlier"]
    if mode == "backbone":
        return b.copy()
    if mode == "projector":
        return p.copy()
    if mode == "union":
        return b | p
    if mode == "intersection":
        return b & p
    raise ValueError(mode)


def apply_combined_removal_to_space(
    space: dict[str, Any],
    labels: np.ndarray,
    combined_mask: np.ndarray,
    combined_top_indices: list[int],
    args: argparse.Namespace,
) -> None:
    """Rebuild the visual before/after comparison using the SAME combined outlier set.

    This ensures that if X total outliers are found by the configured backbone/projector
    combination rule, all X are removed simultaneously from every after-analysis.
    """
    x = space["x"]
    keep = ~combined_mask
    space["removed_outliers"] = int(combined_mask.sum())
    space["combined_outlier_pca_uri"] = make_pca_outlier_plot(
        x,
        combined_mask,
        f"{space['name']}: same PCA, FINAL combined outliers highlighted",
        args.seed,
        annotate_indices=combined_top_indices[: args.max_details],
    )
    space["same_basis_cleaned_pca_uri"] = make_pca_same_basis_cleaned_plot(
        x,
        labels,
        combined_mask,
        f"{space['name']}: original PCA coordinates after removing ALL "
        f"{int(combined_mask.sum())} final outliers",
        args.seed,
    )

    if int(combined_mask.sum()) == 0:
        space["stats_after"] = space["stats_before"]
        space["after_pca_uri"] = None
        space["after_pca_stats"] = None
        return

    if int(keep.sum()) < 3:
        space["stats_after"] = {}
        space["after_pca_uri"] = None
        space["after_pca_stats"] = None
        return

    space["stats_after"] = representation_statistics(x[keep])
    space["after_pca_uri"], space["after_pca_stats"] = make_pca_colored_plot(
        x[keep],
        labels[keep],
        f"{space['name']}: PCA after removing ALL {int(combined_mask.sum())} final outliers",
        args.seed,
    )


def make_detailed_card(
    rank: int,
    idx: int,
    split_df: pd.DataFrame,
    dataset: MGAnalysisDataset,
    train_transform: ConfigurableMGAugmentation,
    model: ViTEncoder,
    device: torch.device,
    spaces: dict[str, dict[str, Any]],
    combined_score: np.ndarray,
    args: argparse.Namespace,
    reference_spaces: Optional[dict[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    row = split_df.iloc[idx]
    raw = dataset.load_raw_tensor(idx)
    processed = dataset.transform(raw.clone())

    # Nearest normal is chosen in backbone space when possible, because the PCA motivating
    # this analysis is a backbone-embedding PCA.
    b_metrics = spaces["Backbone embedding"]["metrics"]
    p_metrics = spaces["Raw projector output"]["metrics"]
    nn_idx = int(b_metrics.loc[idx, "nearest_normal_index"])
    if nn_idx < 0:
        nn_idx = int(p_metrics.loc[idx, "nearest_normal_index"])
    nearest_processed = dataset.transform(dataset.load_raw_tensor(nn_idx)) if nn_idx >= 0 else None

    metadata = metadata_for_row(row)
    nearest_meta = metadata_for_row(split_df.iloc[nn_idx]) if nn_idx >= 0 else {}

    raw_uri = tensor_to_data_uri(raw, "Raw image")
    proc_uri = tensor_to_data_uri(processed, "Deterministic preprocessing")
    nn_uri = tensor_to_data_uri(nearest_processed, "Nearest normal") if nearest_processed is not None else None

    # Stochastic SSL-view analysis.
    aug_rows = []
    aug_imgs = []
    if args.num_aug_views > 0:
        views = [train_transform(raw.clone()) for _ in range(args.num_aug_views)]
        batch = torch.stack(views).to(device)
        with torch.inference_mode():
            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=bool((not args.no_amp) and device.type == "cuda"),
            ):
                emb, proj = model.encode_one(batch)
        emb_np = emb.float().cpu().numpy()
        proj_np = proj.float().cpu().numpy()

        emb_scores = score_new_points(emb_np, spaces["Backbone embedding"]["screen_model"])
        proj_scores = score_new_points(proj_np, spaces["Raw projector output"]["screen_model"])

        for j, view in enumerate(views):
            aug_imgs.append(tensor_to_data_uri(view, f"SSL view {j+1}"))
            aug_rows.append({
                "view": j + 1,
                "backbone_outlier": bool(emb_scores.loc[j, "is_outlier"]),
                "backbone_global_pc_z": float(emb_scores.loc[j, "global_max_abs_pc_z"]),
                "backbone_knn_ratio": float(emb_scores.loc[j, "knn_ratio_to_median"]),
                "projector_outlier": bool(proj_scores.loc[j, "is_outlier"]),
                "projector_global_pc_z": float(proj_scores.loc[j, "global_max_abs_pc_z"]),
                "projector_knn_ratio": float(proj_scores.loc[j, "knn_ratio_to_median"]),
            })

    ref_info = {}
    if reference_spaces is not None:
        for space_name in ["Backbone embedding", "Raw projector output"]:
            m = reference_spaces[space_name]["metrics"]
            ref_info[space_name] = {
                "detected_outlier": bool(m.loc[idx, "is_outlier"]),
                "score_percentile": float(m.loc[idx, "outlier_score_percentile"]),
                "knn_ratio": float(m.loc[idx, "knn_ratio_to_median"]),
            }

    metric_rows = [
        {
            "space": "Backbone",
            "outlier": bool(b_metrics.loc[idx, "is_outlier"]),
            "score_percentile": float(b_metrics.loc[idx, "outlier_score_percentile"]),
            "global_outlier": bool(b_metrics.loc[idx, "global_is_outlier"]),
            "global_iteration_found": int(b_metrics.loc[idx, "global_iteration_found"]),
            "global_max_abs_pc_z": float(b_metrics.loc[idx, "global_max_abs_pc_z"]),
            "global_dominant_pc": int(b_metrics.loc[idx, "global_dominant_pc"]),
            "local_outlier": bool(b_metrics.loc[idx, "local_is_outlier"]),
            "radius_z": float(b_metrics.loc[idx, "radius_robust_z"]),
            "norm_z": float(b_metrics.loc[idx, "norm_robust_z"]),
            "knn_z": float(b_metrics.loc[idx, "knn_robust_z"]),
            "knn_ratio": float(b_metrics.loc[idx, "knn_ratio_to_median"]),
            "relative_normal_separation": float(b_metrics.loc[idx, "relative_normal_separation"]),
        },
        {
            "space": "Projector",
            "outlier": bool(p_metrics.loc[idx, "is_outlier"]),
            "score_percentile": float(p_metrics.loc[idx, "outlier_score_percentile"]),
            "global_outlier": bool(p_metrics.loc[idx, "global_is_outlier"]),
            "global_iteration_found": int(p_metrics.loc[idx, "global_iteration_found"]),
            "global_max_abs_pc_z": float(p_metrics.loc[idx, "global_max_abs_pc_z"]),
            "global_dominant_pc": int(p_metrics.loc[idx, "global_dominant_pc"]),
            "local_outlier": bool(p_metrics.loc[idx, "local_is_outlier"]),
            "radius_z": float(p_metrics.loc[idx, "radius_robust_z"]),
            "norm_z": float(p_metrics.loc[idx, "norm_robust_z"]),
            "knn_z": float(p_metrics.loc[idx, "knn_robust_z"]),
            "knn_ratio": float(p_metrics.loc[idx, "knn_ratio_to_median"]),
            "relative_normal_separation": float(p_metrics.loc[idx, "relative_normal_separation"]),
        },
    ]

    nn_html = (
        f"<img class='mammo' src='{nn_uri}'>"
        if nn_uri
        else "<p>No normal neighbour available.</p>"
    )

    card = [
        f"<article class='card outlier-card'><h3>#{rank} — combined score {combined_score[idx]:.2f}</h3>",
        "<div class='grid3'>",
        f"<div><img class='mammo' src='{raw_uri}'></div>",
        f"<div><img class='mammo' src='{proc_uri}'></div>",
        f"<div>{nn_html}</div>",
        "</div>",
        "<h4>Metadata</h4>",
        dict_table_html([metadata]),
        "<h4>Representation scores</h4>",
        dict_table_html(metric_rows),
    ]

    if nearest_meta:
        card += ["<h4>Nearest-normal metadata</h4>", dict_table_html([nearest_meta])]

    if aug_rows:
        card.append("<h4>Stochastic SSL views</h4>")
        card.append(
            "<p class='small muted'>These views use the checkpoint's saved training augmentation policy. "
            "If only some views become extreme, augmentation sensitivity is a plausible contributor.</p>"
        )
        card.append("<div class='grid3'>")
        for uri in aug_imgs:
            card.append(f"<div><img class='mammo' src='{uri}'></div>")
        card.append("</div>")
        card.append(dict_table_html(aug_rows))

    if ref_info:
        ref_rows = []
        for k, v in ref_info.items():
            ref_rows.append({"space": k, **v})
        card += ["<h4>Same image under reference checkpoint</h4>", dict_table_html(ref_rows)]

    card.append("</article>")

    record = {
        "rank": rank,
        "dataset_index": int(idx),
        "combined_score": float(combined_score[idx]),
        **metadata,
        "backbone_is_outlier": bool(b_metrics.loc[idx, "is_outlier"]),
        "backbone_global_is_outlier": bool(b_metrics.loc[idx, "global_is_outlier"]),
        "backbone_global_iteration_found": int(b_metrics.loc[idx, "global_iteration_found"]),
        "backbone_global_max_abs_pc_z": float(b_metrics.loc[idx, "global_max_abs_pc_z"]),
        "backbone_global_dominant_pc": int(b_metrics.loc[idx, "global_dominant_pc"]),
        "backbone_local_is_outlier": bool(b_metrics.loc[idx, "local_is_outlier"]),
        "backbone_score_percentile": float(b_metrics.loc[idx, "outlier_score_percentile"]),
        "backbone_knn_ratio": float(b_metrics.loc[idx, "knn_ratio_to_median"]),
        "backbone_relative_normal_separation": float(b_metrics.loc[idx, "relative_normal_separation"]),
        "projector_is_outlier": bool(p_metrics.loc[idx, "is_outlier"]),
        "projector_global_is_outlier": bool(p_metrics.loc[idx, "global_is_outlier"]),
        "projector_global_iteration_found": int(p_metrics.loc[idx, "global_iteration_found"]),
        "projector_global_max_abs_pc_z": float(p_metrics.loc[idx, "global_max_abs_pc_z"]),
        "projector_global_dominant_pc": int(p_metrics.loc[idx, "global_dominant_pc"]),
        "projector_local_is_outlier": bool(p_metrics.loc[idx, "local_is_outlier"]),
        "projector_score_percentile": float(p_metrics.loc[idx, "outlier_score_percentile"]),
        "projector_knn_ratio": float(p_metrics.loc[idx, "knn_ratio_to_median"]),
        "projector_relative_normal_separation": float(p_metrics.loc[idx, "relative_normal_separation"]),
        "nearest_normal_dataset_index": int(nn_idx),
    }
    return "".join(card), record


def main() -> None:
    args = parse_args()
    if args.max_details < 0:
        raise ValueError("--max-details must be >= 0")
    if args.num_aug_views < 0:
        raise ValueError("--num-aug-views must be >= 0")
    if args.knn_k < 1:
        raise ValueError("--knn-k must be >= 1")
    if args.screen_pca_components < 2:
        raise ValueError("--screen-pca-components must be >= 2")
    if args.global_pca_components < 2:
        raise ValueError("--global-pca-components must be >= 2")
    if args.global_pca_max_iterations < 1:
        raise ValueError("--global-pca-max-iterations must be >= 1")
    if args.global_pca_z_threshold <= 0:
        raise ValueError("--global-pca-z-threshold must be > 0")
    if not (0.0 < args.max_outlier_fraction < 1.0):
        raise ValueError("--max-outlier-fraction must be between 0 and 1")

    set_seed(args.seed)
    start = time.perf_counter()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    payload = checkpoint_payload(checkpoint)
    cfg = dict(payload.get("config", {}) or {})
    paths = resolve_paths(args, cfg)

    output_dir = args.output_dir
    if output_dir is None:
        # final checkpoint is normally <run>/models/final_lejepa_checkpoint.pt
        run_dir = checkpoint.parent.parent if checkpoint.parent.name == "models" else checkpoint.parent
        output_dir = run_dir / "pca_outlier_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    model, arch, load_info = build_model(payload, device)
    aug_cfg = dict(payload.get("augmentation_config", {}) or {})
    if not aug_cfg:
        aug_path = cfg_get(cfg, "aug_config_path", "augmentation_config", default=None)
        if aug_path:
            with open(aug_path, "r", encoding="utf-8") as f:
                aug_cfg = json.load(f)

    full_raw, split_df = load_analysis_df(args, paths)
    verify_bin(len(full_raw), paths["bin"], arch)

    eval_ds = make_dataset(split_df, len(full_raw), paths["bin"], arch, aug_cfg, train=False)
    train_transform = ConfigurableMGAugmentation(aug_cfg, int(arch["image_size"]), train=True)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(eval_ds, **loader_kwargs)

    print(f"Analyzing split={args.split} with {len(split_df):,} rows.")
    emb, proj = extract_representations(model, loader, device, not args.no_amp, "Primary checkpoint")

    labels = split_df["collapsed_birads"].astype(str).to_numpy()
    spaces = {
        "Backbone embedding": analyze_space("Backbone embedding", emb, labels, args),
        "Raw projector output": analyze_space("Raw projector output", proj, labels, args),
    }

    combined_mask = combined_outlier_mask(spaces, args.outlier_space)
    combined_score = np.maximum(
        spaces["Backbone embedding"]["metrics"]["outlier_score"].to_numpy(),
        spaces["Raw projector output"]["metrics"]["outlier_score"].to_numpy(),
    )

    combined_idx = np.where(combined_mask)[0]
    top_combined_indices = combined_idx[np.argsort(combined_score[combined_idx])[::-1]].tolist() if len(combined_idx) else []

    # Crucial: the before/after comparison removes the FINAL combined set all at once,
    # rather than removing a different set separately for backbone/projector analyses.
    for space in spaces.values():
        apply_combined_removal_to_space(
            space,
            labels,
            combined_mask,
            top_combined_indices,
            args,
        )

    aggregate_tables = {
        col: aggregate_outliers(split_df, combined_mask, col)
        for col in ["collapsed_birads", "birads_numeric", "dataset", "machine_family", "machine", "view", "laterality"]
        if col in split_df.columns
    }

    # If there are no outliers, intentionally skip reference inference and detailed augmentation work.
    reference_spaces = None
    reference_arch = None
    reference_summary = None

    if int(combined_mask.sum()) > 0 and args.reference_checkpoint is not None:
        ref_path = args.reference_checkpoint.resolve()
        if not ref_path.exists():
            raise FileNotFoundError(ref_path)

        print("Loading reference checkpoint...")
        ref_payload = checkpoint_payload(ref_path)
        reference_arch = infer_architecture(ref_payload)
        ref_aug_cfg = dict(ref_payload.get("augmentation_config", {}) or {})

        # Only the ViT backbone must match. The projector may legitimately differ
        # between checkpoints (e.g. hidden dim 512 vs 2048), so rebuild only that
        # small MLP in place without calling timm.create_model() again.
        backbone_compatible_keys = [
            "backbone_name",
            "image_size",
            "backbone_output_dim",
        ]
        backbone_differences = {
            k: (arch.get(k), reference_arch.get(k))
            for k in backbone_compatible_keys
            if arch.get(k) != reference_arch.get(k)
        }
        if backbone_differences:
            raise RuntimeError(
                "Reference checkpoint uses a different BACKBONE architecture and cannot "
                "reuse the already-created ViT safely. "
                f"Differences: {backbone_differences}"
            )

        projector_differences = {
            k: (arch.get(k), reference_arch.get(k))
            for k in ["projector_hidden_dim", "projection_dim"]
            if arch.get(k) != reference_arch.get(k)
        }
        if projector_differences:
            print(
                "Reference checkpoint uses a different projector architecture; "
                f"rebuilding projector in place: {projector_differences}",
                flush=True,
            )

        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if projector_differences:
            rebuild_projector_in_place(model, reference_arch, device)

        load_weights_into_existing_model(model, ref_payload, "Reference")

        # Same raw rows, deterministic preprocessing saved with the reference checkpoint.
        ref_ds = make_dataset(
            split_df,
            len(full_raw),
            paths["bin"],
            reference_arch,
            ref_aug_cfg,
            train=False,
        )
        ref_loader = DataLoader(ref_ds, **loader_kwargs)
        ref_emb, ref_proj = extract_representations(
            model,
            ref_loader,
            device,
            not args.no_amp,
            "Reference checkpoint",
        )
        reference_spaces = {
            "Backbone embedding": analyze_space(
                "Reference backbone embedding", ref_emb, labels, args
            ),
            "Raw projector output": analyze_space(
                "Reference raw projector output", ref_proj, labels, args
            ),
        }
        ref_combined = combined_outlier_mask(reference_spaces, args.outlier_space)

        # Aggregate comparison uses the same combined rule.
        primary_idx = np.where(combined_mask)[0]
        ref_metrics_for_combined = np.maximum(
            reference_spaces["Backbone embedding"]["metrics"]["outlier_score_percentile"].to_numpy(),
            reference_spaces["Raw projector output"]["metrics"]["outlier_score_percentile"].to_numpy(),
        )
        ref_knn_ratio_for_combined = np.maximum(
            reference_spaces["Backbone embedding"]["metrics"]["knn_ratio_to_median"].to_numpy(),
            reference_spaces["Raw projector output"]["metrics"]["knn_ratio_to_median"].to_numpy(),
        )

        reference_summary = {
            "primary_outlier_count": int(len(primary_idx)),
            "also_reference_outlier_count": int(ref_combined[primary_idx].sum()),
            "also_reference_outlier_fraction": float(ref_combined[primary_idx].mean()) if len(primary_idx) else 0.0,
            "median_reference_score_percentile_for_primary_outliers": (
                float(np.median(ref_metrics_for_combined[primary_idx])) if len(primary_idx) else None
            ),
            "median_reference_knn_ratio_for_primary_outliers": (
                float(np.median(ref_knn_ratio_for_combined[primary_idx])) if len(primary_idx) else None
            ),
            "highlight_plot_uri": make_pca_outlier_plot(
                ref_emb,
                combined_mask,
                "Reference backbone PCA: PRIMARY-checkpoint outliers highlighted",
                args.seed,
                annotate_indices=top_combined_indices[: args.max_details],
            ),
        }

        del ref_loader, ref_ds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Detailed cards below need the primary checkpoint again.
        if (
            int(reference_arch["projector_hidden_dim"]) != int(arch["projector_hidden_dim"])
            or int(reference_arch["projection_dim"]) != int(arch["projection_dim"])
        ):
            rebuild_projector_in_place(model, arch, device)

        load_weights_into_existing_model(model, payload, "Primary restore")

    detailed_cards: list[str] = []
    detailed_records: list[dict[str, Any]] = []

    if int(combined_mask.sum()) > 0 and args.max_details > 0:
        print(f"Creating detailed gallery for top {min(args.max_details, len(top_combined_indices))} outliers...")
        for rank, idx in enumerate(top_combined_indices[: args.max_details], start=1):
            card, record = make_detailed_card(
                rank=rank,
                idx=int(idx),
                split_df=split_df,
                dataset=eval_ds,
                train_transform=train_transform,
                model=model,
                device=device,
                spaces=spaces,
                combined_score=combined_score,
                args=args,
                reference_spaces=reference_spaces,
            )
            detailed_cards.append(card)
            detailed_records.append(record)

    runtime_sec = time.perf_counter() - start

    summary = {
        "checkpoint": str(checkpoint),
        "reference_checkpoint": None if args.reference_checkpoint is None else str(args.reference_checkpoint),
        "split": args.split,
        "rows": int(len(split_df)),
        "outlier_space": args.outlier_space,
        "global_pca_z_threshold": float(args.global_pca_z_threshold),
        "global_pca_components": int(args.global_pca_components),
        "global_pca_max_iterations": int(args.global_pca_max_iterations),
        "max_outlier_fraction": float(args.max_outlier_fraction),
        "robust_z_threshold": float(args.robust_z_threshold),
        "hard_knn_ratio": float(args.hard_knn_ratio),
        "knn_k": int(args.knn_k),
        "screen_pca_components": int(args.screen_pca_components),
        "combined_outlier_count": int(combined_mask.sum()),
        "combined_outlier_fraction": float(combined_mask.mean()),
        "max_details": int(args.max_details),
        "details_written": int(len(detailed_records)),
        "primary_architecture": arch,
        "reference_architecture": reference_arch,
        "spaces": {
            name: {
                "num_outliers": int(space["num_outliers"]),
                "outlier_fraction": float(space["outlier_fraction"]),
                "global_outlier_count": int(space["global_outlier_count"]),
                "local_outlier_count": int(space["local_outlier_count"]),
                "global_local_overlap_count": int(space["global_local_overlap_count"]),
                "global_detection_capped": bool(space["global_detection_capped"]),
                "max_global_pc_z": float(space["max_global_pc_z"]),
                "global_pc12_outlier_count": int(space["global_pc12_outlier_count"]),
                "global_higher_pc_outlier_count": int(space["global_higher_pc_outlier_count"]),
                "local_only_outlier_count": int(space["local_only_outlier_count"]),
                "global_iteration_history": space["global_iteration_history"],
                "stats_before": space["stats_before"],
                "stats_after": space["stats_after"],
            }
            for name, space in spaces.items()
        },
        "aggregate_outliers": aggregate_tables,
        "reference_summary": (
            None
            if reference_summary is None
            else {k: v for k, v in reference_summary.items() if k != "highlight_plot_uri"}
        ),
        "runtime_sec": float(runtime_sec),
        "model_load_info": load_info,
    }

    save_json(summary, output_dir / "analysis_summary.json")
    if detailed_records:
        pd.DataFrame(detailed_records).to_csv(output_dir / "top_outliers.csv", index=False)

    report_html = build_report(
        args=args,
        checkpoint=checkpoint,
        reference_checkpoint=args.reference_checkpoint.resolve() if args.reference_checkpoint else None,
        arch=arch,
        reference_arch=reference_arch,
        split_df=split_df,
        spaces=spaces,
        combined_mask=combined_mask,
        top_combined_indices=top_combined_indices,
        detailed_cards=detailed_cards,
        aggregate_tables=aggregate_tables,
        reference_summary=reference_summary,
        runtime_sec=runtime_sec,
    )
    report_path = output_dir / "pca_outlier_analysis.html"
    report_path.write_text(report_html, encoding="utf-8")

    print("\nAnalysis complete.")
    print(f"  Report:  {report_path}")
    print(f"  Summary: {output_dir / 'analysis_summary.json'}")
    if detailed_records:
        print(f"  Top-N:   {output_dir / 'top_outliers.csv'}")
    print(
        f"  Combined outliers: {int(combined_mask.sum())}/{len(combined_mask)} "
        f"({100*combined_mask.mean():.3f}%)"
    )
    if int(combined_mask.sum()) == 0:
        print("  No strong outliers found; reference/detailed stages were skipped as intended.")


if __name__ == "__main__":
    main()
