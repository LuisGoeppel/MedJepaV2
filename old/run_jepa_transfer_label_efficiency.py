#!/usr/bin/env python3
"""
MedJEPA transfer-learning / label-efficiency experiment.

Compares three supervised training regimes on fixed MG train/val/test CSVs:

1. random
   Randomly initialized ViT backbone + classification head, trained end-to-end.

2. frozen_jepa
   Load a LeJEPA checkpoint, keep the backbone frozen, train only the same
   classification head.

3. finetune_jepa
   Load a LeJEPA checkpoint, first train only the head for a configurable warmup,
   then unfreeze the backbone and fine-tune end-to-end.

The script loops over user-provided label budgets, e.g.
  --subset-sizes 2000 5000 10000 25000 50000 full

It saves one folder per budget/mode and writes an aggregate summary CSV/JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
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
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
except ImportError as exc:
    raise ImportError("Missing dependency: scikit-learn. Install with: pip install scikit-learn") from exc


CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class ExperimentConfig:
    checkpoint: str
    full_csv: str
    train_csv: str
    val_csv: str
    test_csv: str
    bin_path: str
    aug_config: str
    output_dir: str

    backbone: str = "vit_small_patch8_224"
    image_size: int = 224
    # embedding_dim is the output dimension of the timm ViT head when backbone_num_classes > 0.
    # For raw ViT features (backbone_num_classes=0), the actual embedding dimension is inferred from timm.
    embedding_dim: int = 512
    # timm num_classes for the backbone. None means: use embedding_dim as before.
    # Use 0 to remove the timm classification/projection head and use raw ViT features.
    backbone_num_classes: Optional[int] = None
    # If enabled, infer backbone/patch size/image size/head dim from the JEPA checkpoint config/state.
    auto_model_from_checkpoint: bool = True
    drop_path_rate: float = 0.1
    image_height: int = 512
    image_width: int = 512
    memmap_dtype: str = "uint16"
    normalize_mode: str = "uint16"
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    subset_sizes: tuple[str, ...] = ("2000", "5000", "10000", "25000", "50000", "full")
    subset_strategy: str = "balanced"
    modes: tuple[str, ...] = ("random", "frozen_jepa", "finetune_jepa")

    epochs: int = 100
    patience: int = 20
    head_warmup_epochs: int = 10
    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 4

    random_learning_rate: float = 3e-4
    head_learning_rate: float = 3e-4
    backbone_learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 5e-2
    grad_clip_norm: float = 0.0
    label_smoothing: float = 0.0
    use_class_weights: bool = True

    # For supervised transfer, the SSL augmentation recipe can be too strong.
    # config = use full LeJEPA aug JSON; mild = crop/mask/resize + optional hflip/small rotation; none = deterministic crop/mask/resize.
    supervised_aug_mode: str = "mild"
    hflip_p: float = 0.5
    max_rotation_deg: float = 5.0

    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    save_train_used_csv: bool = True


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_budget(s: str) -> Optional[int]:
    s = str(s).strip().lower().replace("_", "")
    if s in {"full", "all", "none", "0", "-1"}:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k)?", s)
    if not m:
        raise ValueError(f"Invalid subset size: {s}. Use e.g. 2000, 2k, 50000, full.")
    val = float(m.group(1))
    if m.group(2) == "k":
        val *= 1000
    n = int(round(val))
    if n <= 0:
        raise ValueError(f"Subset size must be positive or full, got {s}")
    return n


def budget_name(s: str) -> str:
    n = parse_budget(s)
    return "full" if n is None else str(n)


def stable_int_from_text(text: str, modulo: int = 1_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


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


def read_csv_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


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
    if "collapsed_birads" in df.columns and set(df["collapsed_birads"].dropna().astype(str).unique()).issubset(set(CLASS_NAMES)):
        df = df[df["collapsed_birads"].isin(CLASS_NAMES)].copy()
        df["target_collapsed"] = df["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
        return df

    label_col = "original_birads" if "original_birads" in df.columns else "birads"
    if label_col not in df.columns:
        raise ValueError("CSV must contain original_birads, birads, or collapsed_birads.")
    df["birads_numeric"] = df[label_col].apply(normalize_birads_value)
    df = df[df["birads_numeric"].isin([1, 2, 3, 4, 5])].copy()
    df["birads_numeric"] = df["birads_numeric"].astype(int)
    df["collapsed_birads"] = df["birads_numeric"].apply(collapse_birads_numeric)
    df["target_collapsed"] = df["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
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


def verify_no_patient_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    if "patient" not in train_df.columns:
        return {"warning": "no patient column found; cannot verify leakage"}  # type: ignore[return-value]
    train_p = set(train_df["patient"].astype(str))
    val_p = set(val_df["patient"].astype(str))
    test_p = set(test_df["patient"].astype(str))
    overlaps = {"train_val": len(train_p & val_p), "train_test": len(train_p & test_p), "val_test": len(val_p & test_p)}
    if any(v != 0 for v in overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")
    return overlaps


def load_splits(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_raw = read_csv_clean(cfg.full_csv)
    full_df = prepare_labels(full_raw)
    if "original_index" not in full_df.columns:
        full_df = full_df.reset_index(drop=False).rename(columns={"index": "original_index"})

    train_df = prepare_labels(ensure_original_index(read_csv_clean(cfg.train_csv), full_raw, "train"))
    val_df = prepare_labels(ensure_original_index(read_csv_clean(cfg.val_csv), full_raw, "val"))
    test_df = prepare_labels(ensure_original_index(read_csv_clean(cfg.test_csv), full_raw, "test"))
    return full_df, train_df, val_df, test_df


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.as_tensor(weights, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Augmentation copied/refactored from the v6 MG augmentation path
# -----------------------------------------------------------------------------

def deep_get(dct: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class ConfigurableMGAugmentation(nn.Module):
    """Config-driven mammography augmentation compatible with the v6 aug JSON."""

    _warned_no_cv2 = False

    def __init__(self, aug_cfg: dict[str, Any], image_size: int, train: bool):
        super().__init__()
        self.cfg = aug_cfg
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
            return x.clamp(0, 1)

        x = self._resize(x, self.image_size)
        return x.clamp(0, 1)

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
        mh, mw = int((y1 - y0) * margin_frac), int((x1 - x0) * margin_frac)
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
                relevant_components = 0
                for label_idx in range(1, num_labels):
                    if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_area:
                        relevant_components += 1
                if relevant_components <= 1:
                    return x
            except Exception:
                pass
        left_half = foreground[:, : w // 2]
        right_half = foreground[:, w // 2:]
        left_foreground = left_half.float().sum().item()
        right_foreground = right_half.float().sum().item()
        x = x.clone()
        if left_foreground <= right_foreground:
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
        return TF.resized_crop(x, i, j, h, w, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)

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
        return TF.affine(x, angle=angle, translate=[tx, ty], scale=scale, shear=shear,
                         interpolation=InterpolationMode.BILINEAR, fill=[float(c.get("fill", 0.0))])

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
        b, co = c.get("brightness", [0.95, 1.05]), c.get("contrast", [0.9, 1.1])
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
                print("WARNING: CLAHE requested but OpenCV/cv2 is unavailable or failed. Skipping CLAHE.", flush=True)
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


class MildSupervisedMGAugmentation(ConfigurableMGAugmentation):
    """Milder supervised augmentation for label-efficiency experiments.

    The SSL/LeJEPA augmentation config can be intentionally aggressive and stochastic.
    That is useful for invariance learning but can make small supervised subsets very hard
    to fit. This transform keeps the mammography-specific preprocessing from v6
    (foreground crop and top-corner marker masking), then applies only mild supervised
    augmentation.
    """

    def __init__(self, aug_cfg: dict[str, Any], image_size: int, train: bool, hflip_p: float = 0.5, max_rotation_deg: float = 5.0):
        super().__init__(aug_cfg, image_size, train)
        self.hflip_p = float(hflip_p)
        self.max_rotation_deg = float(max_rotation_deg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._foreground_crop(x)
        x = self._mask_top_corner(x)
        x = TF.resize(x, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if self.train:
            if self.hflip_p > 0 and random.random() < self.hflip_p:
                x = TF.hflip(x)
            if self.max_rotation_deg > 0:
                angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                x = TF.rotate(x, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        return x.float().clamp(0, 1)


def make_supervised_transform(aug_cfg: dict[str, Any], cfg: ExperimentConfig, train: bool) -> nn.Module:
    mode = str(cfg.supervised_aug_mode).lower().strip()
    if mode == "config":
        return ConfigurableMGAugmentation(aug_cfg, cfg.image_size, train=train)
    if mode == "mild":
        return MildSupervisedMGAugmentation(aug_cfg, cfg.image_size, train=train, hflip_p=cfg.hflip_p, max_rotation_deg=cfg.max_rotation_deg)
    if mode == "none":
        return MildSupervisedMGAugmentation(aug_cfg, cfg.image_size, train=train, hflip_p=0.0, max_rotation_deg=0.0)
    raise ValueError(f"Unknown supervised_aug_mode: {cfg.supervised_aug_mode}")


# -----------------------------------------------------------------------------
# Dataset and model
# -----------------------------------------------------------------------------

class MGSupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, bin_path: str | Path, full_num_rows: int,
                 image_shape: tuple[int, int], dtype: str, transform: nn.Module,
                 normalize_mode: str = "uint16", percentile_low: float = 1.0, percentile_high: float = 99.0):
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
        if "original_index" not in self.df.columns:
            raise ValueError("Split dataframe requires original_index.")

    def __len__(self) -> int:
        return len(self.df)

    def _open(self) -> np.memmap:
        if self._imgs is None:
            self._imgs = np.memmap(self.bin_path, dtype=self.dtype, mode="r", shape=(self.full_num_rows, *self.image_shape))
        return self._imgs

    def _load_tensor(self, original_index: int) -> torch.Tensor:
        arr = self._open()[int(original_index)].astype(np.float32)
        if self.normalize_mode == "uint16":
            arr = arr / 65535.0 if self.dtype == np.dtype("uint16") else arr / max(float(arr.max()), 1.0)
        elif self.normalize_mode == "per_image_percentile":
            lo, hi = np.percentile(arr, [self.percentile_low, self.percentile_high])
            arr = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[int(idx)]
        x = self._load_tensor(int(row["original_index"]))
        x = self.transform(x)
        y = int(row["target_collapsed"])
        return x, torch.tensor(y, dtype=torch.long)


class ViTBackboneClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        embedding_dim: int,
        drop_path_rate: float,
        num_classes: int = 3,
        backbone_num_classes: Optional[int] = None,
    ):
        super().__init__()
        timm_num_classes = int(embedding_dim if backbone_num_classes is None else backbone_num_classes)
        self.backbone_num_classes = timm_num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=timm_num_classes,
            drop_path_rate=drop_path_rate,
            img_size=image_size,
            in_chans=1,
        )
        if timm_num_classes == 0:
            actual_embedding_dim = int(getattr(self.backbone, "num_features", 0))
            if actual_embedding_dim <= 0:
                raise RuntimeError(f"Could not infer num_features for raw backbone {backbone_name}.")
        else:
            actual_embedding_dim = int(embedding_dim)
        self.embedding_dim = actual_embedding_dim
        self.head = nn.Sequential(nn.LayerNorm(actual_embedding_dim), nn.Linear(actual_embedding_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)

    @torch.inference_mode()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def strip_module_prefix(k: str) -> str:
    return k[len("module."):] if k.startswith("module.") else k


def extract_backbone_state(payload: Any) -> tuple[dict[str, torch.Tensor], int, dict[str, torch.Tensor]]:
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("Could not find a state dict in checkpoint payload.")
    backbone_state: dict[str, torch.Tensor] = {}
    full_backbone_state: dict[str, torch.Tensor] = {}
    ignored_projector = 0
    for k, v in state.items():
        k = strip_module_prefix(str(k))
        if k.startswith("backbone."):
            kk = k[len("backbone."):]
            backbone_state[kk] = v
            full_backbone_state[kk] = v
        elif k.startswith("proj."):
            ignored_projector += 1
    if not backbone_state:
        raise ValueError("No backbone.* keys found in checkpoint. Expected v6/v7 LeJEPA checkpoint.")
    return backbone_state, ignored_projector, full_backbone_state


def _as_dict_maybe(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        return dict(vars(x))
    return {}


def infer_model_hints_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Infer ViT architecture/head information from a LeJEPA checkpoint.

    This prevents the common patch8/patch16 mismatch, e.g. checkpoint pos_embed [1,197,384]
    but current model [1,785,384].
    """
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    backbone_state, ignored_projector, _ = extract_backbone_state(payload)
    ckpt_cfg = _as_dict_maybe(payload.get("config") if isinstance(payload, dict) else {})

    hints: dict[str, Any] = {"checkpoint": str(checkpoint_path), "ignored_projector_keys": int(ignored_projector)}
    for key in ["backbone_name", "image_size", "backbone_output_dim", "backbone_num_classes"]:
        if key in ckpt_cfg and ckpt_cfg[key] is not None:
            hints[key] = ckpt_cfg[key]

    patch_w = backbone_state.get("patch_embed.proj.weight")
    if isinstance(patch_w, torch.Tensor) and patch_w.ndim == 4:
        hints["patch_size"] = int(patch_w.shape[-1])
        hints["vit_feature_dim"] = int(patch_w.shape[0])

    pos = backbone_state.get("pos_embed")
    if isinstance(pos, torch.Tensor) and pos.ndim == 3 and "patch_size" in hints:
        # ViT position embedding length is usually 1 + grid_h * grid_w.
        n_patches = int(pos.shape[1]) - 1
        grid = int(round(math.sqrt(max(1, n_patches))))
        if grid * grid == n_patches and "image_size" not in hints:
            hints["image_size"] = int(grid * int(hints["patch_size"]))

    head_w = backbone_state.get("head.weight")
    if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
        hints["backbone_num_classes"] = int(head_w.shape[0])
        hints["backbone_output_dim"] = int(head_w.shape[0])
        hints["vit_feature_dim"] = int(head_w.shape[1])
    else:
        # Raw timm backbone with num_classes=0 usually has no head.weight.
        if "backbone_num_classes" not in hints:
            hints["backbone_num_classes"] = 0
        if "backbone_output_dim" not in hints and "vit_feature_dim" in hints:
            hints["backbone_output_dim"] = int(hints["vit_feature_dim"])

    return hints


def maybe_update_config_from_checkpoint(cfg: ExperimentConfig) -> dict[str, Any]:
    hints = infer_model_hints_from_checkpoint(cfg.checkpoint)

    # Prefer explicit checkpoint config if present. Otherwise patch backbone name by detected patch size.
    if "backbone_name" in hints:
        cfg.backbone = str(hints["backbone_name"])
    elif "patch_size" in hints:
        ps = int(hints["patch_size"])
        if re.search(r"patch\d+", cfg.backbone):
            cfg.backbone = re.sub(r"patch\d+", f"patch{ps}", cfg.backbone)

    if "image_size" in hints:
        cfg.image_size = int(hints["image_size"])
    if "backbone_num_classes" in hints:
        cfg.backbone_num_classes = int(hints["backbone_num_classes"])
    if "backbone_output_dim" in hints:
        cfg.embedding_dim = int(hints["backbone_output_dim"])

    return hints


def load_jepa_backbone(model: ViTBackboneClassifier, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    backbone_state, ignored_projector, _ = extract_backbone_state(payload)

    try:
        load_info = model.backbone.load_state_dict(backbone_state, strict=False)
    except RuntimeError as exc:
        hints = infer_model_hints_from_checkpoint(checkpoint_path)
        raise RuntimeError(
            "Failed to load JEPA backbone. This is usually a model mismatch.\n"
            f"Checkpoint hints: {json.dumps(hints, indent=2, default=str)}\n"
            f"Current model: backbone={model.backbone.__class__.__name__}, "
            f"backbone_num_classes={model.backbone_num_classes}, embedding_dim={model.embedding_dim}.\n"
            "Check --backbone, --image-size, --embedding-dim and --backbone-num-classes, "
            "or leave --auto-model-from-checkpoint enabled.\n"
            f"Original error: {exc}"
        ) from exc

    return {
        "checkpoint": str(checkpoint_path),
        "loaded_backbone_keys": int(len(backbone_state)),
        "ignored_projector_keys": int(ignored_projector),
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "checkpoint_config": payload.get("config") if isinstance(payload, dict) else None,
        "model_backbone_num_classes": int(model.backbone_num_classes),
        "model_embedding_dim": int(model.embedding_dim),
    }


def set_backbone_trainable(model: ViTBackboneClassifier, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)


def make_optimizer(model: ViTBackboneClassifier, cfg: ExperimentConfig, mode: str, stage: str) -> torch.optim.Optimizer:
    if mode == "random":
        return torch.optim.AdamW(model.parameters(), lr=cfg.random_learning_rate, weight_decay=cfg.weight_decay)
    if mode == "frozen_jepa" or stage == "head_warmup":
        return torch.optim.AdamW(model.head.parameters(), lr=cfg.head_learning_rate, weight_decay=cfg.weight_decay)
    if mode == "finetune_jepa" and stage == "finetune":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg.backbone_learning_rate},
                {"params": model.head.parameters(), "lr": cfg.head_learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer mode/stage: {mode}/{stage}")


def make_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, cfg: ExperimentConfig):
    # Simple epoch-level cosine scheduler. Kept intentionally simple for fair baseline comparisons.
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(total_epochs)), eta_min=cfg.min_learning_rate)


# -----------------------------------------------------------------------------
# Subset selection
# -----------------------------------------------------------------------------

def _sample_indices(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    n = min(n, len(indices))
    return rng.choice(indices, size=n, replace=False)


def select_train_subset(train_df: pd.DataFrame, budget_text: str, strategy: str, seed: int) -> pd.DataFrame:
    budget = parse_budget(budget_text)
    if budget is None or budget >= len(train_df):
        return train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if budget < len(CLASS_NAMES):
        raise ValueError(f"Budget {budget} is smaller than number of classes {len(CLASS_NAMES)}.")

    rng = np.random.default_rng(seed)
    labels = train_df["target_collapsed"].to_numpy(dtype=np.int64)
    class_indices = {c: np.where(labels == c)[0] for c in range(len(CLASS_NAMES))}

    if strategy == "natural":
        chosen = rng.choice(np.arange(len(train_df)), size=budget, replace=False)

    elif strategy == "balanced":
        base = budget // len(CLASS_NAMES)
        rem = budget % len(CLASS_NAMES)
        parts = []
        for i, c in enumerate(range(len(CLASS_NAMES))):
            n = base + (1 if i < rem else 0)
            parts.append(_sample_indices(class_indices[c], n, rng))
        chosen = np.concatenate(parts)
        # If a class has fewer available rows than requested, fill with remaining rows.
        if len(chosen) < budget:
            remaining = np.setdiff1d(np.arange(len(train_df)), chosen, assume_unique=False)
            fill = _sample_indices(remaining, budget - len(chosen), rng)
            chosen = np.concatenate([chosen, fill])

    elif strategy == "minority_inclusive":
        minority = np.concatenate([class_indices[1], class_indices[2]])
        if len(minority) <= budget:
            routine_needed = budget - len(minority)
            routine = _sample_indices(class_indices[0], routine_needed, rng)
            chosen = np.concatenate([minority, routine])
        else:
            # If the budget cannot include all minority labels, fall back to balanced sampling.
            base = budget // len(CLASS_NAMES)
            rem = budget % len(CLASS_NAMES)
            parts = []
            for i, c in enumerate(range(len(CLASS_NAMES))):
                n = base + (1 if i < rem else 0)
                parts.append(_sample_indices(class_indices[c], n, rng))
            chosen = np.concatenate(parts)
            if len(chosen) < budget:
                remaining = np.setdiff1d(np.arange(len(train_df)), chosen, assume_unique=False)
                chosen = np.concatenate([chosen, _sample_indices(remaining, budget - len(chosen), rng)])
    else:
        raise ValueError(f"Unknown subset strategy: {strategy}")

    rng.shuffle(chosen)
    return train_df.iloc[chosen].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Training/evaluation
# -----------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    total = 0
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(amp and use_cuda)):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
            pred = logits.argmax(dim=1)
            preds.append(pred.detach().cpu())
            trues.append(y.detach().cpu())
            total_loss += float(loss.item()) * x.size(0)
            total += int(x.size(0))
    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "loss": float(total_loss / max(1, total)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }


def plot_history(history: dict[str, list[float]], out_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 8))
    for i, key in enumerate(["train_loss", "train_accuracy", "val_loss", "val_balanced_accuracy", "val_macro_f1", "lr_head"]):
        ax = plt.subplot(2, 3, i + 1)
        if key in history:
            ax.plot(epochs, history[key])
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close()


def plot_confusion(cm: list[list[int]], title: str, out_path: Path) -> None:
    arr = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(3), CLASS_NAMES)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(int(arr[i, j])), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path) -> None:
    splits = {"train_used": train_df, "val": val_df, "test": test_df}
    counts = np.array([
        [int((df["target_collapsed"] == c).sum()) for c in range(3)]
        for df in splits.values()
    ])
    x = np.arange(len(splits))
    bottom = np.zeros(len(splits))
    plt.figure(figsize=(7, 5))
    for c, name in enumerate(CLASS_NAMES):
        plt.bar(x, counts[:, c], bottom=bottom, label=name)
        bottom += counts[:, c]
    plt.xticks(x, list(splits.keys()))
    plt.ylabel("Rows")
    plt.title("Class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_one_run(cfg: ExperimentConfig, mode: str, budget_text: str, train_df: pd.DataFrame,
                  val_df: pd.DataFrame, test_df: pd.DataFrame, full_num_rows: int,
                  aug_cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    run_name = f"budget_{budget_name(budget_text)}__{mode}"
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.seed + stable_int_from_text(f"{budget_name(budget_text)}::{mode}")
    set_seed(seed)

    train_used = select_train_subset(train_df, budget_text, cfg.subset_strategy, cfg.seed)
    if cfg.save_train_used_csv:
        train_used.to_csv(out_dir / "train_used.csv", index=False)

    train_transform = make_supervised_transform(aug_cfg, cfg, train=True)
    eval_transform = make_supervised_transform(aug_cfg, cfg, train=False)
    shape = (cfg.image_height, cfg.image_width)

    train_ds = MGSupDataset(
        train_used, cfg.bin_path, full_num_rows, shape, cfg.memmap_dtype, train_transform,
        cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high,
    )
    val_ds = MGSupDataset(
        val_df, cfg.bin_path, full_num_rows, shape, cfg.memmap_dtype, eval_transform,
        cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high,
    )
    test_ds = MGSupDataset(
        test_df, cfg.bin_path, full_num_rows, shape, cfg.memmap_dtype, eval_transform,
        cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high,
    )

    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    model = ViTBackboneClassifier(
        cfg.backbone, cfg.image_size, cfg.embedding_dim, cfg.drop_path_rate, len(CLASS_NAMES), cfg.backbone_num_classes
    ).to(device)
    # For raw timm features (backbone_num_classes=0), the classifier infers the true feature dim.
    cfg.embedding_dim = int(model.embedding_dim)

    load_info: Optional[dict[str, Any]] = None
    if mode in {"frozen_jepa", "finetune_jepa"}:
        load_info = load_jepa_backbone(model, cfg.checkpoint, device)

    if mode == "frozen_jepa":
        set_backbone_trainable(model, False)
        stage = "head_warmup"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, cfg.epochs, cfg)
    elif mode == "finetune_jepa":
        set_backbone_trainable(model, False)
        stage = "head_warmup"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, max(1, min(cfg.head_warmup_epochs, cfg.epochs)), cfg)
    elif mode == "random":
        set_backbone_trainable(model, True)
        stage = "finetune"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, cfg.epochs, cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    labels_np = train_used["target_collapsed"].to_numpy(dtype=np.int64)
    class_weights = compute_class_weights(labels_np, len(CLASS_NAMES)).to(device) if cfg.use_class_weights else None

    run_config = {
        "mode": mode,
        "budget": budget_text,
        "budget_resolved": budget_name(budget_text),
        "subset_strategy": cfg.subset_strategy,
        "train_used_rows": int(len(train_used)),
        "train_used_class_counts": {CLASS_NAMES[i]: int((labels_np == i).sum()) for i in range(3)},
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "class_weights": None if class_weights is None else {CLASS_NAMES[i]: float(class_weights.detach().cpu()[i]) for i in range(3)},
        "config": asdict(cfg),
        "load_info": load_info,
    }
    save_json(run_config, out_dir / "config.json")
    plot_class_distribution(train_used, val_df, test_df, out_dir / "class_distribution.png")

    history = {
        "epoch": [], "stage": [], "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [], "val_balanced_accuracy": [], "val_macro_f1": [], "val_weighted_f1": [],
        "lr_backbone": [], "lr_head": [], "epoch_time_sec": [],
    }

    best_val_ba = -1.0
    best_epoch = 0
    best_path = out_dir / "best_by_val_balanced_accuracy.pt"
    no_improve = 0
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f"\n=== Run: {run_name} ===", flush=True)
    print(f"Train rows: {len(train_used):,} | counts: {run_config['train_used_class_counts']}", flush=True)
    print(f"Model: backbone={cfg.backbone} image_size={cfg.image_size} backbone_num_classes={cfg.backbone_num_classes} embedding_dim={cfg.embedding_dim} aug_mode={cfg.supervised_aug_mode}", flush=True)
    if load_info is not None:
        print(f"Loaded JEPA backbone keys: {load_info['loaded_backbone_keys']} | ignored projector: {load_info['ignored_projector_keys']}", flush=True)
        if load_info["missing_keys"] or load_info["unexpected_keys"]:
            print(f"Backbone load missing keys: {len(load_info['missing_keys'])}, unexpected: {len(load_info['unexpected_keys'])}", flush=True)

    for epoch in range(1, cfg.epochs + 1):
        if mode == "finetune_jepa" and epoch == cfg.head_warmup_epochs + 1:
            print("Switching finetune_jepa from head warm-up to end-to-end fine-tuning.", flush=True)
            set_backbone_trainable(model, True)
            stage = "finetune"
            optimizer = make_optimizer(model, cfg, mode, stage)
            scheduler = make_scheduler(optimizer, max(1, cfg.epochs - cfg.head_warmup_epochs), cfg)

        model.train()
        # During frozen phases, keep the frozen backbone in eval mode so dropout/drop-path
        # does not inject noise into fixed features while the head is learning.
        if mode in {"frozen_jepa", "finetune_jepa"} and stage == "head_warmup":
            model.backbone.eval()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(cfg.amp and use_cuda)):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float(), y,
                    weight=class_weights,
                    label_smoothing=float(cfg.label_smoothing),
                )
            loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, cfg.amp)

        lr_backbone = None
        lr_head = None
        if len(optimizer.param_groups) == 1:
            lr_head = float(optimizer.param_groups[0]["lr"])
            lr_backbone = float(optimizer.param_groups[0]["lr"] if stage == "finetune" else 0.0)
        else:
            lr_backbone = float(optimizer.param_groups[0]["lr"])
            lr_head = float(optimizer.param_groups[1]["lr"])

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        elapsed = time.perf_counter() - epoch_start

        history["epoch"].append(epoch)
        history["stage"].append(stage)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["val_weighted_f1"].append(float(val_metrics["weighted_f1"]))
        history["lr_backbone"].append(float(lr_backbone if lr_backbone is not None else 0.0))
        history["lr_head"].append(float(lr_head if lr_head is not None else 0.0))
        history["epoch_time_sec"].append(float(elapsed))

        save_json(history, out_dir / "training_history.json")

        improved = float(val_metrics["balanced_accuracy"]) > best_val_ba + 1e-8
        if improved:
            best_val_ba = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": int(epoch),
                    "mode": mode,
                    "budget": budget_text,
                    "model_state_dict": model.state_dict(),
                    "config": run_config,
                    "history": history,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | stage={stage} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | best={best_val_ba:.4f}@{best_epoch} | "
            f"lr_backbone={lr_backbone:.2e} lr_head={lr_head:.2e} | time={elapsed/60:.2f} min",
            flush=True,
        )

        if no_improve >= cfg.patience:
            print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was written for {run_name}")

    payload = torch.load(best_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    val_best_metrics = evaluate(model, val_loader, device, cfg.amp)
    test_metrics = evaluate(model, test_loader, device, cfg.amp)

    result = {
        "mode": mode,
        "budget": budget_text,
        "budget_resolved": budget_name(budget_text),
        "subset_strategy": cfg.subset_strategy,
        "train_used_rows": int(len(train_used)),
        "train_used_class_counts": run_config["train_used_class_counts"],
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy_during_training": float(best_val_ba),
        "val_best": val_best_metrics,
        "test": test_metrics,
        "class_names": CLASS_NAMES,
        "output_dir": str(out_dir),
    }
    save_json(result, out_dir / "metrics.json")
    plot_history(history, out_dir)
    plot_confusion(val_best_metrics["confusion_matrix"], f"Val confusion matrix: {run_name}", out_dir / "confusion_matrix_val_best.png")
    plot_confusion(test_metrics["confusion_matrix"], f"Test confusion matrix: {run_name}", out_dir / "confusion_matrix_test.png")

    print("Test summary:", json.dumps({k: test_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2), flush=True)
    return result


def write_aggregate_outputs(results: list[dict[str, Any]], cfg: ExperimentConfig) -> None:
    out_dir = Path(cfg.output_dir)
    rows = []
    for r in results:
        row = {
            "budget": r["budget_resolved"],
            "mode": r["mode"],
            "subset_strategy": r["subset_strategy"],
            "train_used_rows": r["train_used_rows"],
            "best_epoch": r["best_epoch"],
            "val_balanced_accuracy": r["val_best"]["balanced_accuracy"],
            "val_macro_f1": r["val_best"]["macro_f1"],
            "val_accuracy": r["val_best"]["accuracy"],
            "test_balanced_accuracy": r["test"]["balanced_accuracy"],
            "test_macro_f1": r["test"]["macro_f1"],
            "test_accuracy": r["test"]["accuracy"],
            "test_weighted_f1": r["test"]["weighted_f1"],
            "routine_recall": r["test"]["per_class"]["routine"]["recall"],
            "follow_up_recall": r["test"]["per_class"]["follow_up"]["recall"],
            "biopsy_recall": r["test"]["per_class"]["biopsy"]["recall"],
            "output_dir": r["output_dir"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_table.csv", index=False)
    save_json(results, out_dir / "summary_results.json")

    # Plot label-efficiency curves for the numeric budgets only.
    try:
        plot_df = df[df["budget"] != "full"].copy()
        plot_df["budget_numeric"] = plot_df["budget"].astype(int)
        if len(plot_df):
            plt.figure(figsize=(8, 5))
            for mode, sub in plot_df.groupby("mode"):
                sub = sub.sort_values("budget_numeric")
                plt.plot(sub["budget_numeric"], sub["test_balanced_accuracy"], marker="o", label=mode)
            plt.xscale("log")
            plt.xlabel("Labeled train samples")
            plt.ylabel("Test balanced accuracy")
            plt.title("Label-efficiency comparison")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "label_efficiency_test_balanced_accuracy.png", dpi=200)
            plt.close()
    except Exception as exc:
        print(f"WARNING: failed to plot aggregate label-efficiency curve: {exc}", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedJEPA supervised transfer / label-efficiency experiment.")
    p.add_argument("--checkpoint", required=True, type=str, help="LeJEPA checkpoint for frozen_jepa / finetune_jepa modes.")
    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--val-csv", required=True, type=str)
    p.add_argument("--test-csv", required=True, type=str)
    p.add_argument("--bin", required=True, dest="bin_path", type=str)
    p.add_argument("--aug-config", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument("--backbone", default="vit_small_patch8_224", type=str)
    p.add_argument("--image-size", default=224, type=int)
    p.add_argument("--embedding-dim", default=512, type=int)
    p.add_argument("--backbone-num-classes", default=None, type=int,
                   help="timm num_classes for backbone. Default: embedding_dim. Use 0 for raw ViT features.")
    p.add_argument("--auto-model-from-checkpoint", dest="auto_model_from_checkpoint", action="store_true", default=True,
                   help="Infer backbone/patch size/head dim from the JEPA checkpoint before running all modes. Default: on.")
    p.add_argument("--no-auto-model-from-checkpoint", dest="auto_model_from_checkpoint", action="store_false",
                   help="Disable architecture inference from checkpoint.")
    p.add_argument("--drop-path-rate", default=0.1, type=float)
    p.add_argument("--image-height", default=512, type=int)
    p.add_argument("--image-width", default=512, type=int)
    p.add_argument("--memmap-dtype", default="uint16", type=str)
    p.add_argument("--normalize-mode", default="uint16", choices=["uint16", "per_image_percentile"])
    p.add_argument("--percentile-low", default=1.0, type=float)
    p.add_argument("--percentile-high", default=99.0, type=float)

    p.add_argument("--subset-sizes", nargs="+", required=True,
                   help="Label budgets, e.g. 2000 5000 10k 50k full. Not hard-coded.")
    p.add_argument("--subset-strategy", default="balanced", choices=["balanced", "minority_inclusive", "natural"],
                   help="How to sample each labeled subset from the train split.")
    p.add_argument("--modes", nargs="+", default=["random", "frozen_jepa", "finetune_jepa"],
                   choices=["random", "frozen_jepa", "finetune_jepa"])

    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--patience", default=20, type=int)
    p.add_argument("--head-warmup-epochs", default=10, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--eval-batch-size", default=256, type=int)
    p.add_argument("--num-workers", default=4, type=int)

    p.add_argument("--random-learning-rate", default=3e-4, type=float)
    p.add_argument("--head-learning-rate", default=3e-4, type=float)
    p.add_argument("--backbone-learning-rate", default=3e-5, type=float)
    p.add_argument("--min-learning-rate", default=1e-6, type=float)
    p.add_argument("--weight-decay", default=5e-2, type=float)
    p.add_argument("--grad-clip-norm", default=0.0, type=float)
    p.add_argument("--label-smoothing", default=0.0, type=float)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--supervised-aug-mode", default="mild", choices=["config", "mild", "none"],
                   help="config uses the full LeJEPA aug JSON; mild uses crop/mask/resize + hflip/small rotation; none is deterministic crop/mask/resize.")
    p.add_argument("--hflip-p", default=0.5, type=float)
    p.add_argument("--max-rotation-deg", default=5.0, type=float)

    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-save-train-used-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()

    cfg = ExperimentConfig(
        checkpoint=args.checkpoint,
        full_csv=args.full_csv,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        bin_path=args.bin_path,
        aug_config=args.aug_config,
        output_dir=args.output_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        backbone_num_classes=args.backbone_num_classes,
        auto_model_from_checkpoint=bool(args.auto_model_from_checkpoint),
        drop_path_rate=args.drop_path_rate,
        image_height=args.image_height,
        image_width=args.image_width,
        memmap_dtype=args.memmap_dtype,
        normalize_mode=args.normalize_mode,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        subset_sizes=tuple(args.subset_sizes),
        subset_strategy=args.subset_strategy,
        modes=tuple(args.modes),
        epochs=args.epochs,
        patience=args.patience,
        head_warmup_epochs=args.head_warmup_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        random_learning_rate=args.random_learning_rate,
        head_learning_rate=args.head_learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        use_class_weights=not bool(args.no_class_weights),
        supervised_aug_mode=args.supervised_aug_mode,
        hflip_p=args.hflip_p,
        max_rotation_deg=args.max_rotation_deg,
        seed=args.seed,
        device=args.device,
        amp=not bool(args.no_amp),
        save_train_used_csv=not bool(args.no_save_train_used_csv),
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hints: dict[str, Any] = {}
    if cfg.auto_model_from_checkpoint:
        checkpoint_hints = maybe_update_config_from_checkpoint(cfg)
        print("Auto model config from checkpoint:", json.dumps(checkpoint_hints, indent=2, default=str), flush=True)

    save_json(asdict(cfg), out_dir / "experiment_config.json")
    if checkpoint_hints:
        save_json(checkpoint_hints, out_dir / "checkpoint_model_hints.json")

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device), flush=True)

    with open(cfg.aug_config, "r", encoding="utf-8") as f:
        aug_cfg = json.load(f)
    save_json(aug_cfg, out_dir / "augmentation_config_used.json")

    full_df, train_df, val_df, test_df = load_splits(cfg)
    leakage = verify_no_patient_leakage(train_df, val_df, test_df)

    expected_bytes = len(full_df) * cfg.image_height * cfg.image_width * np.dtype(cfg.memmap_dtype).itemsize
    actual_bytes = Path(cfg.bin_path).stat().st_size
    print(f"BIN check: expected={expected_bytes/1024**3:.3f} GiB actual={actual_bytes/1024**3:.3f} GiB match={expected_bytes == actual_bytes}", flush=True)
    if expected_bytes != actual_bytes:
        raise RuntimeError("Full CSV and BIN do not match. Refusing to train.")

    split_summary = {
        "patient_leakage": leakage,
        "train": {"rows": len(train_df), "classes": train_df["collapsed_birads"].value_counts().to_dict()},
        "val": {"rows": len(val_df), "classes": val_df["collapsed_birads"].value_counts().to_dict()},
        "test": {"rows": len(test_df), "classes": test_df["collapsed_birads"].value_counts().to_dict()},
    }
    save_json(split_summary, out_dir / "split_summary.json")
    print(json.dumps(split_summary, indent=2), flush=True)

    results = []
    start = time.perf_counter()
    for budget in cfg.subset_sizes:
        for mode in cfg.modes:
            result = train_one_run(cfg, mode, budget, train_df, val_df, test_df, len(full_df), aug_cfg, device)
            results.append(result)
            write_aggregate_outputs(results, cfg)

    elapsed = time.perf_counter() - start
    save_json({"wall_time_sec": elapsed, "num_runs": len(results)}, out_dir / "timing.json")
    write_aggregate_outputs(results, cfg)
    print("Finished all transfer experiments.", flush=True)
    print(pd.read_csv(out_dir / "summary_table.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
