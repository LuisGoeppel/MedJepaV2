#!/usr/bin/env python3
"""
MedJEPA transfer-learning / label-efficiency experiment -- V2.

Main V2 changes relative to run_jepa_transfer_label_efficiency.py:
- Progressive, nested label subsets:
    * smallest numeric budget starts fully balanced (balance degree = 1)
    * full data uses the natural train distribution (balance degree = 0)
    * intermediate budgets interpolate monotonically in log-budget space
    * requested balance is automatically reduced when a minority class is exhausted
    * subsets are nested within a seed: D_2k subset D_5k subset ... subset D_full
- Seed handling:
    * user provides one base --seed
    * optional --num-seeds uses seed, seed+1, seed+2, ...
    * the same seed/subset is used for all modes within a repetition
- Random-from-scratch optimization:
    * configurable LR warm-up
    * default LR raised to 3e-4
    * random mode cannot early-stop before a configurable fraction of max epochs
- JEPA fine-tuning:
    * head warm-up uses a constant LR instead of decaying to eta_min and jumping back up
    * after unfreezing, a separate (lower) head LR is used
    * early-stopping patience is reset at the unfreeze boundary
- Better diagnostics:
    * per-epoch class recalls and predicted-class fractions
    * collapse diagnostics in metrics/summary tables
    * more aggregate/root-level summary plots, without adding new run-specific plot types
- LR history now records the LR actually used during an epoch (before scheduler.step()).
"""

from __future__ import annotations

import argparse
import gc
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
    embedding_dim: int = 512
    backbone_num_classes: Optional[int] = None
    auto_model_from_checkpoint: bool = True
    drop_path_rate: float = 0.1
    image_height: int = 512
    image_width: int = 512
    memmap_dtype: str = "uint16"
    normalize_mode: str = "uint16"
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    subset_sizes: tuple[str, ...] = ("2000", "5000", "10000", "25000", "50000", "full")
    subset_strategy: str = "progressive"
    balance_curve: str = "log"
    modes: tuple[str, ...] = ("random", "frozen_jepa", "finetune_jepa")

    epochs: int = 100
    patience: int = 15
    head_warmup_epochs: int = 5
    random_min_epoch_fraction: float = 0.5
    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 4

    random_learning_rate: float = 3e-4
    random_warmup_epochs: int = 5
    random_warmup_start_factor: float = 0.1
    head_learning_rate: float = 3e-4
    finetune_head_learning_rate: float = 1e-4
    backbone_learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.0
    label_smoothing: float = 0.0
    use_class_weights: bool = True
    selection_metric: str = "balanced_accuracy"

    supervised_aug_mode: str = "mild"
    hflip_p: float = 0.5
    max_rotation_deg: float = 5.0

    seed: int = 42
    num_seeds: int = 1
    device: str = "cuda"
    amp: bool = True
    save_train_used_csv: bool = True
    save_run_plots: bool = True


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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    overlaps = {
        "train_val": len(train_p & val_p),
        "train_test": len(train_p & test_p),
        "val_test": len(val_p & test_p),
    }
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
# Augmentation (same mammography path as V1)
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
    def __init__(
        self,
        aug_cfg: dict[str, Any],
        image_size: int,
        train: bool,
        hflip_p: float = 0.5,
        max_rotation_deg: float = 5.0,
    ):
        super().__init__(aug_cfg, image_size, train)
        self.hflip_p = float(hflip_p)
        self.max_rotation_deg = float(max_rotation_deg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._foreground_crop(x)
        x = self._mask_top_corner(x)
        x = TF.resize(
            x, [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True
        )
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
        return MildSupervisedMGAugmentation(
            aug_cfg, cfg.image_size, train=train,
            hflip_p=cfg.hflip_p, max_rotation_deg=cfg.max_rotation_deg
        )
    if mode == "none":
        return MildSupervisedMGAugmentation(
            aug_cfg, cfg.image_size, train=train,
            hflip_p=0.0, max_rotation_deg=0.0
        )
    raise ValueError(f"Unknown supervised_aug_mode: {cfg.supervised_aug_mode}")


# -----------------------------------------------------------------------------
# Dataset and model
# -----------------------------------------------------------------------------

class MGSupDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bin_path: str | Path,
        full_num_rows: int,
        image_shape: tuple[int, int],
        dtype: str,
        transform: nn.Module,
        normalize_mode: str = "uint16",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
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
        if "original_index" not in self.df.columns:
            raise ValueError("Split dataframe requires original_index.")

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
        self.head = nn.Sequential(
            nn.LayerNorm(actual_embedding_dim),
            nn.Linear(actual_embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a model state to CPU so the same single model object can be reset safely.

    This avoids repeatedly calling timm.create_model() between experiment runs.
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def restore_initial_model_state(
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
) -> None:
    """Restore the exact initial random weights before a new transfer run."""
    model.load_state_dict(initial_state, strict=True)
    model.zero_grad(set_to_none=True)


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
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    backbone_state, ignored_projector, _ = extract_backbone_state(payload)
    ckpt_cfg = _as_dict_maybe(payload.get("config") if isinstance(payload, dict) else {})

    hints: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "ignored_projector_keys": int(ignored_projector),
    }
    for key in ["backbone_name", "image_size", "backbone_output_dim", "backbone_num_classes"]:
        if key in ckpt_cfg and ckpt_cfg[key] is not None:
            hints[key] = ckpt_cfg[key]

    patch_w = backbone_state.get("patch_embed.proj.weight")
    if isinstance(patch_w, torch.Tensor) and patch_w.ndim == 4:
        hints["patch_size"] = int(patch_w.shape[-1])
        hints["vit_feature_dim"] = int(patch_w.shape[0])

    pos = backbone_state.get("pos_embed")
    if isinstance(pos, torch.Tensor) and pos.ndim == 3 and "patch_size" in hints:
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
        if "backbone_num_classes" not in hints:
            hints["backbone_num_classes"] = 0
        if "backbone_output_dim" not in hints and "vit_feature_dim" in hints:
            hints["backbone_output_dim"] = int(hints["vit_feature_dim"])

    return hints


def maybe_update_config_from_checkpoint(cfg: ExperimentConfig) -> dict[str, Any]:
    hints = infer_model_hints_from_checkpoint(cfg.checkpoint)

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


def load_jepa_backbone(
    model: ViTBackboneClassifier,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
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


# -----------------------------------------------------------------------------
# V2 nested / progressive subset planning
# -----------------------------------------------------------------------------

def _budget_to_n(budget_text: str, full_n: int) -> int:
    n = parse_budget(budget_text)
    return full_n if n is None else min(int(n), full_n)


def raw_balance_degree(n: int, min_n: int, full_n: int, curve: str) -> float:
    if full_n <= min_n:
        return 0.0 if n >= full_n else 1.0
    if n <= min_n:
        return 1.0
    if n >= full_n:
        return 0.0
    if curve == "log":
        val = 1.0 - math.log(float(n) / float(min_n)) / math.log(float(full_n) / float(min_n))
    elif curve == "linear":
        val = 1.0 - (float(n) - float(min_n)) / (float(full_n) - float(min_n))
    else:
        raise ValueError(f"Unknown balance curve: {curve}")
    return float(np.clip(val, 0.0, 1.0))


def feasible_balance_cap(n: int, capacities: np.ndarray, natural_probs: np.ndarray) -> float:
    """Maximum b in p(b)=b*(1/K)+(1-b)*p_natural that does not request
    more unique samples from any class than are available.
    """
    if n <= 0:
        return 0.0
    k = len(capacities)
    balanced_p = 1.0 / float(k)
    caps = [1.0]
    for c in range(k):
        denom = balanced_p - float(natural_probs[c])
        if denom > 1e-12:
            upper = (float(capacities[c]) / float(n) - float(natural_probs[c])) / denom
            caps.append(upper)
    return float(np.clip(min(caps), 0.0, 1.0))


def blended_probs(balance_degree: float, natural_probs: np.ndarray) -> np.ndarray:
    k = len(natural_probs)
    return balance_degree * np.full(k, 1.0 / k, dtype=np.float64) + (1.0 - balance_degree) * natural_probs


def allocate_counts(
    n: int,
    probs: np.ndarray,
    capacities: np.ndarray,
    minimum_counts: np.ndarray,
) -> np.ndarray:
    """Integer allocation close to n*probs, respecting capacity and nested minima."""
    desired = np.asarray(probs, dtype=np.float64) * float(n)
    counts = np.floor(desired).astype(np.int64)
    counts = np.minimum(counts, capacities)
    counts = np.maximum(counts, minimum_counts)

    # If monotonic minima pushed us over budget, reduce only above minima.
    while int(counts.sum()) > n:
        candidates = np.where(counts > minimum_counts)[0]
        if len(candidates) == 0:
            raise RuntimeError(
                f"Cannot satisfy nested allocation: minima sum={int(minimum_counts.sum())} > budget={n}"
            )
        # Remove from the class furthest above its desired count.
        surplus = counts[candidates].astype(np.float64) - desired[candidates]
        c = int(candidates[int(np.argmax(surplus))])
        counts[c] -= 1

    # Fill remaining slots toward desired counts first, then toward remaining capacity.
    while int(counts.sum()) < n:
        candidates = np.where(counts < capacities)[0]
        if len(candidates) == 0:
            raise RuntimeError(f"Ran out of capacity while allocating budget {n}.")
        deficits = desired[candidates] - counts[candidates].astype(np.float64)
        if float(deficits.max()) > 0:
            c = int(candidates[int(np.argmax(deficits))])
        else:
            remaining = capacities[candidates] - counts[candidates]
            c = int(candidates[int(np.argmax(remaining))])
        counts[c] += 1

    return counts


def build_subset_plan(
    train_df: pd.DataFrame,
    budget_texts: tuple[str, ...],
    strategy: str,
    curve: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build all subsets for one seed at once, guaranteeing nestedness."""
    full_n = len(train_df)
    labels = train_df["target_collapsed"].to_numpy(dtype=np.int64)
    capacities = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.int64)
    natural_probs = capacities.astype(np.float64) / float(full_n)

    budget_pairs = [(b, _budget_to_n(b, full_n)) for b in budget_texts]
    numeric_nonfull = sorted({n for b, n in budget_pairs if parse_budget(b) is not None})
    min_n = numeric_nonfull[0] if numeric_nonfull else full_n

    # One class-specific random order per seed. Prefixes create nested subsets.
    rng = np.random.default_rng(seed)
    class_orders: dict[int, np.ndarray] = {}
    for c in range(len(CLASS_NAMES)):
        idx = np.where(labels == c)[0]
        class_orders[c] = rng.permutation(idx)

    # Build target counts in increasing budget order.
    unique_ns = sorted(set(n for _, n in budget_pairs))
    previous = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    specs_by_n: dict[int, dict[str, Any]] = {}

    for n in unique_ns:
        if n >= full_n:
            raw_b = 0.0
            effective_b = 0.0
            probs = natural_probs.copy()
            target_counts = capacities.copy()
        elif strategy == "natural":
            raw_b = 0.0
            effective_b = 0.0
            probs = natural_probs.copy()
            target_counts = allocate_counts(n, probs, capacities, previous)
        elif strategy == "balanced":
            raw_b = 1.0
            effective_b = feasible_balance_cap(n, capacities, natural_probs)
            probs = blended_probs(effective_b, natural_probs)
            target_counts = allocate_counts(n, probs, capacities, previous)
        elif strategy == "progressive":
            raw_b = raw_balance_degree(n, min_n, full_n, curve)
            effective_b = min(raw_b, feasible_balance_cap(n, capacities, natural_probs))
            probs = blended_probs(effective_b, natural_probs)
            target_counts = allocate_counts(n, probs, capacities, previous)
        else:
            raise ValueError(f"Unknown subset strategy: {strategy}")

        if np.any(target_counts < previous):
            raise RuntimeError("Internal error: subset target counts are not nested.")
        previous = target_counts.copy()

        specs_by_n[n] = {
            "budget_n": int(n),
            "raw_balance_degree": float(raw_b),
            "effective_balance_degree": float(effective_b),
            "target_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "target_counts": {CLASS_NAMES[i]: int(target_counts[i]) for i in range(len(CLASS_NAMES))},
        }

    plan: dict[str, dict[str, Any]] = {}
    previous_set: set[int] = set()

    for budget_text, n in sorted(budget_pairs, key=lambda x: x[1]):
        spec = dict(specs_by_n[n])
        counts_arr = np.array([spec["target_counts"][name] for name in CLASS_NAMES], dtype=np.int64)
        selected_parts = [class_orders[c][:counts_arr[c]] for c in range(len(CLASS_NAMES))]
        selected = np.concatenate(selected_parts)

        # Deterministic order per budget, without affecting which rows are selected.
        order_rng = np.random.default_rng(seed + stable_int_from_text(f"subset-order::{budget_name(budget_text)}"))
        selected = selected.copy()
        order_rng.shuffle(selected)
        subset_df = train_df.iloc[selected].reset_index(drop=True)

        current_set = set(int(x) for x in selected.tolist())
        nested_ok = previous_set.issubset(current_set)
        if not nested_ok:
            raise RuntimeError(f"Nested-subset invariant failed at budget {budget_text}.")
        previous_set = current_set

        spec.update({
            "budget_text": budget_text,
            "budget_resolved": budget_name(budget_text),
            "strategy": strategy,
            "balance_curve": curve,
            "seed": int(seed),
            "nested_ok": True,
            "rows": int(len(subset_df)),
            "actual_counts": {
                CLASS_NAMES[i]: int((subset_df["target_collapsed"].to_numpy() == i).sum())
                for i in range(len(CLASS_NAMES))
            },
            "df": subset_df,
        })
        plan[budget_name(budget_text)] = spec

    return plan


# -----------------------------------------------------------------------------
# Optimizers / schedulers
# -----------------------------------------------------------------------------

def make_optimizer(
    model: ViTBackboneClassifier,
    cfg: ExperimentConfig,
    mode: str,
    stage: str,
) -> torch.optim.Optimizer:
    if mode == "random":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.random_learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if mode == "frozen_jepa" or stage == "head_warmup":
        return torch.optim.AdamW(
            model.head.parameters(),
            lr=cfg.head_learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if mode == "finetune_jepa" and stage == "finetune":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg.backbone_learning_rate},
                {"params": model.head.parameters(), "lr": cfg.finetune_head_learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer mode/stage: {mode}/{stage}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    mode: str,
    stage: str,
    total_epochs: int,
):
    total_epochs = max(1, int(total_epochs))

    if mode == "finetune_jepa" and stage == "head_warmup":
        # V2: keep head warm-up LR constant. Do not decay to eta_min just to reset it.
        return None

    if mode == "random" and cfg.random_warmup_epochs > 0 and total_epochs > 1:
        warmup_epochs = min(int(cfg.random_warmup_epochs), max(1, total_epochs - 1))
        start_factor = float(np.clip(cfg.random_warmup_start_factor, 1e-4, 1.0))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=cfg.min_learning_rate,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=cfg.min_learning_rate,
    )


def get_current_lrs(
    optimizer: torch.optim.Optimizer,
    mode: str,
    stage: str,
) -> tuple[float, float]:
    if len(optimizer.param_groups) == 1:
        lr = float(optimizer.param_groups[0]["lr"])
        if mode == "frozen_jepa" or stage == "head_warmup":
            return 0.0, lr
        return lr, lr
    return float(optimizer.param_groups[0]["lr"]), float(optimizer.param_groups[1]["lr"])


# -----------------------------------------------------------------------------
# Evaluation / plotting
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
            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp and use_cuda),
            ):
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pred_counts = cm.sum(axis=0)
    pred_fracs = pred_counts / max(1, int(pred_counts.sum()))
    min_recall = float(np.min(recall))
    missing_predicted_classes = [CLASS_NAMES[i] for i in range(3) if int(pred_counts[i]) == 0]

    return {
        "loss": float(total_loss / max(1, total)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "min_class_recall": min_recall,
        "missing_predicted_classes": missing_predicted_classes,
        "prediction_counts": {CLASS_NAMES[i]: int(pred_counts[i]) for i in range(3)},
        "prediction_fractions": {CLASS_NAMES[i]: float(pred_fracs[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
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
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


def plot_history(history: dict[str, list[Any]], out_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 8))
    keys = [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_balanced_accuracy",
        "val_macro_f1",
        "lr_head",
    ]
    for i, key in enumerate(keys):
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


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: Path,
) -> None:
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


def selection_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unknown selection metric: {metric_name}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_one_run(
    cfg: ExperimentConfig,
    mode: str,
    budget_text: str,
    train_used: pd.DataFrame,
    subset_meta: dict[str, Any],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_num_rows: int,
    aug_cfg: dict[str, Any],
    device: torch.device,
    run_seed: int,
    seed_index: int,
    model: ViTBackboneClassifier,
    initial_model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    run_name = f"budget_{budget_name(budget_text)}__{mode}"
    if cfg.num_seeds > 1:
        out_dir = Path(cfg.output_dir) / f"seed_{run_seed}" / run_name
    else:
        out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same seed for all modes of this repetition. The subset was also generated from this seed.
    set_seed(run_seed)

    if cfg.save_train_used_csv:
        train_used.to_csv(out_dir / "train_used.csv", index=False)

    train_transform = make_supervised_transform(aug_cfg, cfg, train=True)
    eval_transform = make_supervised_transform(aug_cfg, cfg, train=False)
    shape = (cfg.image_height, cfg.image_width)

    train_ds = MGSupDataset(
        train_used,
        cfg.bin_path,
        full_num_rows,
        shape,
        cfg.memmap_dtype,
        train_transform,
        cfg.normalize_mode,
        cfg.percentile_low,
        cfg.percentile_high,
    )
    val_ds = MGSupDataset(
        val_df,
        cfg.bin_path,
        full_num_rows,
        shape,
        cfg.memmap_dtype,
        eval_transform,
        cfg.normalize_mode,
        cfg.percentile_low,
        cfg.percentile_high,
    )
    test_ds = MGSupDataset(
        test_df,
        cfg.bin_path,
        full_num_rows,
        shape,
        cfg.memmap_dtype,
        eval_transform,
        cfg.normalize_mode,
        cfg.percentile_low,
        cfg.percentile_high,
    )

    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )
    if cfg.num_workers > 0:
        # Keep multiprocessing, but do not keep worker processes alive across epochs/evaluations.
        # This avoids retaining native/host resources between sequential experiment runs.
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2

    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # CRITICAL STABILITY FIX:
    # Reuse the one ViT object created in main(). Constructing another timm ViT after
    # a completed DataLoader/training run reproducibly segfaults in timm trunc_normal_
    # on the single-GPU Determined tasks. Restoring this CPU snapshot is equivalent to
    # starting from the same seeded random initialization, without re-entering
    # timm.create_model().
    restore_initial_model_state(model, initial_model_state)
    model.to(device)
    cfg.embedding_dim = int(model.embedding_dim)

    load_info: Optional[dict[str, Any]] = None
    if mode in {"frozen_jepa", "finetune_jepa"}:
        load_info = load_jepa_backbone(model, cfg.checkpoint, device)

    if mode == "frozen_jepa":
        set_backbone_trainable(model, False)
        stage = "head_warmup"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, cfg, mode, stage, cfg.epochs)
    elif mode == "finetune_jepa":
        set_backbone_trainable(model, False)
        stage = "head_warmup"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(
            optimizer,
            cfg,
            mode,
            stage,
            max(1, min(cfg.head_warmup_epochs, cfg.epochs)),
        )
    elif mode == "random":
        set_backbone_trainable(model, True)
        stage = "finetune"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, cfg, mode, stage, cfg.epochs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    labels_np = train_used["target_collapsed"].to_numpy(dtype=np.int64)
    class_weights = (
        compute_class_weights(labels_np, len(CLASS_NAMES)).to(device)
        if cfg.use_class_weights
        else None
    )

    run_config = {
        "mode": mode,
        "budget": budget_text,
        "budget_resolved": budget_name(budget_text),
        "seed": int(run_seed),
        "seed_index": int(seed_index),
        "subset_strategy": cfg.subset_strategy,
        "subset_meta": {k: v for k, v in subset_meta.items() if k != "df"},
        "train_used_rows": int(len(train_used)),
        "train_used_class_counts": {
            CLASS_NAMES[i]: int((labels_np == i).sum()) for i in range(3)
        },
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "class_weights": (
            None
            if class_weights is None
            else {
                CLASS_NAMES[i]: float(class_weights.detach().cpu()[i])
                for i in range(3)
            }
        ),
        "config": asdict(cfg),
        "load_info": load_info,
    }
    save_json(run_config, out_dir / "config.json")

    if cfg.save_run_plots:
        plot_class_distribution(
            train_used,
            val_df,
            test_df,
            out_dir / "class_distribution.png",
        )

    history: dict[str, list[Any]] = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "val_min_class_recall": [],
        "val_recall_routine": [],
        "val_recall_follow_up": [],
        "val_recall_biopsy": [],
        "val_pred_fraction_routine": [],
        "val_pred_fraction_follow_up": [],
        "val_pred_fraction_biopsy": [],
        "lr_backbone": [],
        "lr_head": [],
        "epoch_time_sec": [],
    }

    best_score = -1.0
    best_val_ba = -1.0
    best_macro_f1 = -1.0
    best_epoch = 0
    best_path = out_dir / f"best_by_val_{cfg.selection_metric}.pt"
    no_improve = 0
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    random_min_stop_epoch = int(math.ceil(cfg.epochs * cfg.random_min_epoch_fraction))

    print(f"\n=== Run: {run_name} | seed={run_seed} ===", flush=True)
    print(
        "Model lifecycle: reused shared ViT; reset to initial seeded state; "
        f"JEPA backbone load={'yes' if mode in {'frozen_jepa', 'finetune_jepa'} else 'no'}.",
        flush=True,
    )
    print(
        f"Train rows: {len(train_used):,} | counts: {run_config['train_used_class_counts']} | "
        f"balance raw={subset_meta.get('raw_balance_degree', float('nan')):.4f} "
        f"effective={subset_meta.get('effective_balance_degree', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"Model: backbone={cfg.backbone} image_size={cfg.image_size} "
        f"backbone_num_classes={cfg.backbone_num_classes} embedding_dim={cfg.embedding_dim} "
        f"aug_mode={cfg.supervised_aug_mode}",
        flush=True,
    )
    if mode == "random":
        print(
            f"Random optimization: lr={cfg.random_learning_rate:.2e}, "
            f"warmup={cfg.random_warmup_epochs} epochs, "
            f"earliest early-stop epoch={random_min_stop_epoch}",
            flush=True,
        )
    if load_info is not None:
        print(
            f"Loaded JEPA backbone keys: {load_info['loaded_backbone_keys']} | "
            f"ignored projector: {load_info['ignored_projector_keys']}",
            flush=True,
        )
        if load_info["missing_keys"] or load_info["unexpected_keys"]:
            print(
                f"Backbone load missing keys: {len(load_info['missing_keys'])}, "
                f"unexpected: {len(load_info['unexpected_keys'])}",
                flush=True,
            )

    for epoch in range(1, cfg.epochs + 1):
        if mode == "finetune_jepa" and epoch == cfg.head_warmup_epochs + 1:
            print(
                "Switching finetune_jepa from head warm-up to end-to-end fine-tuning. "
                "Resetting early-stop patience and using the lower finetune head LR.",
                flush=True,
            )
            set_backbone_trainable(model, True)
            stage = "finetune"
            optimizer = make_optimizer(model, cfg, mode, stage)
            scheduler = make_scheduler(
                optimizer,
                cfg,
                mode,
                stage,
                max(1, cfg.epochs - cfg.head_warmup_epochs),
            )
            no_improve = 0

        model.train()
        if mode in {"frozen_jepa", "finetune_jepa"} and stage == "head_warmup":
            model.backbone.eval()

        # Record the learning rates actually used for this epoch.
        lr_backbone, lr_head = get_current_lrs(optimizer, mode, stage)

        epoch_start = time.perf_counter()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"{run_name} seed={run_seed} epoch {epoch}/{cfg.epochs}",
            leave=False,
        )
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(cfg.amp and use_cuda),
            ):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float(),
                    y,
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

        val_metrics = evaluate(model, val_loader, device, cfg.amp)
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
        history["val_min_class_recall"].append(float(val_metrics["min_class_recall"]))
        history["val_recall_routine"].append(float(val_metrics["per_class"]["routine"]["recall"]))
        history["val_recall_follow_up"].append(float(val_metrics["per_class"]["follow_up"]["recall"]))
        history["val_recall_biopsy"].append(float(val_metrics["per_class"]["biopsy"]["recall"]))
        history["val_pred_fraction_routine"].append(float(val_metrics["prediction_fractions"]["routine"]))
        history["val_pred_fraction_follow_up"].append(float(val_metrics["prediction_fractions"]["follow_up"]))
        history["val_pred_fraction_biopsy"].append(float(val_metrics["prediction_fractions"]["biopsy"]))
        history["lr_backbone"].append(float(lr_backbone))
        history["lr_head"].append(float(lr_head))
        history["epoch_time_sec"].append(float(elapsed))

        save_json(history, out_dir / "training_history.json")

        score = selection_value(val_metrics, cfg.selection_metric)
        # BA remains the primary default. Macro-F1 only breaks effectively exact ties.
        improved = (
            score > best_score + 1e-8
            or (
                abs(score - best_score) <= 1e-8
                and float(val_metrics["macro_f1"]) > best_macro_f1 + 1e-8
            )
        )
        if improved:
            best_score = float(score)
            best_val_ba = float(val_metrics["balanced_accuracy"])
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": int(epoch),
                    "mode": mode,
                    "budget": budget_text,
                    "seed": int(run_seed),
                    "selection_metric": cfg.selection_metric,
                    "selection_score": float(best_score),
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
            f"Epoch {epoch:03d} | stage={stage} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"min_recall={val_metrics['min_class_recall']:.4f} | "
            f"best_{cfg.selection_metric}={best_score:.4f}@{best_epoch} | "
            f"lr_backbone={lr_backbone:.2e} lr_head={lr_head:.2e} | "
            f"time={elapsed/60:.2f} min",
            flush=True,
        )

        if scheduler is not None:
            scheduler.step()

        early_stop_allowed = True
        if mode == "random":
            early_stop_allowed = epoch >= random_min_stop_epoch

        if early_stop_allowed and no_improve >= cfg.patience:
            print(
                f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}. "
                f"Selection metric: {cfg.selection_metric}={best_score:.4f}",
                flush=True,
            )
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
        "seed": int(run_seed),
        "seed_index": int(seed_index),
        "subset_strategy": cfg.subset_strategy,
        "raw_balance_degree": float(subset_meta.get("raw_balance_degree", 0.0)),
        "effective_balance_degree": float(subset_meta.get("effective_balance_degree", 0.0)),
        "train_used_rows": int(len(train_used)),
        "train_used_class_counts": run_config["train_used_class_counts"],
        "selection_metric": cfg.selection_metric,
        "best_selection_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy_during_training": float(best_val_ba),
        "val_best": val_best_metrics,
        "test": test_metrics,
        "class_names": CLASS_NAMES,
        "output_dir": str(out_dir),
    }
    save_json(result, out_dir / "metrics.json")

    if cfg.save_run_plots:
        plot_history(history, out_dir)
        plot_confusion(
            val_best_metrics["confusion_matrix"],
            f"Val confusion matrix: {run_name}",
            out_dir / "confusion_matrix_val_best.png",
        )
        plot_confusion(
            test_metrics["confusion_matrix"],
            f"Test confusion matrix: {run_name}",
            out_dir / "confusion_matrix_test.png",
        )

    print(
        "Test summary:",
        json.dumps(
            {
                "accuracy": test_metrics["accuracy"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "min_class_recall": test_metrics["min_class_recall"],
                "missing_predicted_classes": test_metrics["missing_predicted_classes"],
                "prediction_fractions": test_metrics["prediction_fractions"],
            },
            indent=2,
        ),
        flush=True,
    )

    # Explicitly release PER-RUN resources. The ViT itself is intentionally kept alive
    # and reused by the next experiment, which avoids the native timm initialization crash.
    del payload
    del optimizer
    if scheduler is not None:
        del scheduler
    del train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds

    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# -----------------------------------------------------------------------------
# Aggregate outputs / summary plots
# -----------------------------------------------------------------------------

def _budget_numeric_for_plot(budget: str, train_full_n: int) -> int:
    return train_full_n if budget == "full" else int(budget)


def _summary_rows(results: list[dict[str, Any]], train_full_n: int) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "seed": r["seed"],
            "seed_index": r["seed_index"],
            "budget": r["budget_resolved"],
            "budget_numeric": int(r["train_used_rows"]),
            "mode": r["mode"],
            "subset_strategy": r["subset_strategy"],
            "raw_balance_degree": r["raw_balance_degree"],
            "effective_balance_degree": r["effective_balance_degree"],
            "train_used_rows": r["train_used_rows"],
            "train_routine": r["train_used_class_counts"]["routine"],
            "train_follow_up": r["train_used_class_counts"]["follow_up"],
            "train_biopsy": r["train_used_class_counts"]["biopsy"],
            "best_epoch": r["best_epoch"],
            "val_balanced_accuracy": r["val_best"]["balanced_accuracy"],
            "val_macro_f1": r["val_best"]["macro_f1"],
            "val_accuracy": r["val_best"]["accuracy"],
            "val_min_class_recall": r["val_best"]["min_class_recall"],
            "test_balanced_accuracy": r["test"]["balanced_accuracy"],
            "test_macro_f1": r["test"]["macro_f1"],
            "test_accuracy": r["test"]["accuracy"],
            "test_weighted_f1": r["test"]["weighted_f1"],
            "test_min_class_recall": r["test"]["min_class_recall"],
            "routine_recall": r["test"]["per_class"]["routine"]["recall"],
            "follow_up_recall": r["test"]["per_class"]["follow_up"]["recall"],
            "biopsy_recall": r["test"]["per_class"]["biopsy"]["recall"],
            "pred_routine_fraction": r["test"]["prediction_fractions"]["routine"],
            "pred_follow_up_fraction": r["test"]["prediction_fractions"]["follow_up"],
            "pred_biopsy_fraction": r["test"]["prediction_fractions"]["biopsy"],
            "missing_predicted_classes": ",".join(r["test"]["missing_predicted_classes"]),
            "output_dir": r["output_dir"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "budget_numeric",
        "raw_balance_degree",
        "effective_balance_degree",
        "train_used_rows",
        "train_routine",
        "train_follow_up",
        "train_biopsy",
        "best_epoch",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_accuracy",
        "val_min_class_recall",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_accuracy",
        "test_weighted_f1",
        "test_min_class_recall",
        "routine_recall",
        "follow_up_recall",
        "biopsy_recall",
        "pred_routine_fraction",
        "pred_follow_up_fraction",
        "pred_biopsy_fraction",
    ]
    agg = df.groupby(["budget", "mode"], as_index=False)[numeric_cols].mean()
    return agg


def _plot_metric_curve(mean_df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for mode, sub in mean_df.groupby("mode"):
        sub = sub.sort_values("budget_numeric")
        plt.plot(sub["budget_numeric"], sub[metric], marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("Labeled train samples")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_summary_outputs(df: pd.DataFrame, train_full_n: int, out_dir: Path) -> None:
    if df.empty:
        return
    mean_df = _mean_summary(df)

    _plot_metric_curve(
        mean_df,
        "test_balanced_accuracy",
        "Test balanced accuracy",
        "Label-efficiency comparison",
        out_dir / "summary_test_balanced_accuracy.png",
    )
    _plot_metric_curve(
        mean_df,
        "test_macro_f1",
        "Test macro F1",
        "Label-efficiency macro-F1 comparison",
        out_dir / "summary_test_macro_f1.png",
    )
    _plot_metric_curve(
        mean_df,
        "test_min_class_recall",
        "Minimum test class recall",
        "Worst-class recall across label budgets",
        out_dir / "summary_test_min_class_recall.png",
    )

    # Per-class recall summary.
    plt.figure(figsize=(12, 4))
    for i, class_name in enumerate(CLASS_NAMES):
        ax = plt.subplot(1, 3, i + 1)
        metric = f"{class_name}_recall"
        for mode, sub in mean_df.groupby("mode"):
            sub = sub.sort_values("budget_numeric")
            ax.plot(sub["budget_numeric"], sub[metric], marker="o", label=mode)
        ax.set_xscale("log")
        ax.set_title(f"{class_name} recall")
        ax.set_xlabel("Labeled samples")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Test recall")
        if i == 2:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_test_per_class_recall.png", dpi=200)
    plt.close()

    # Best epoch by mode.
    plt.figure(figsize=(8, 5))
    for mode, sub in mean_df.groupby("mode"):
        sub = sub.sort_values("budget_numeric")
        plt.plot(sub["budget_numeric"], sub["best_epoch"], marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("Labeled train samples")
    plt.ylabel("Best epoch")
    plt.title("Best checkpoint epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_best_epoch.png", dpi=200)
    plt.close()

    # Subset schedule is identical across modes for a seed; average across duplicates/modes.
    schedule = (
        df.groupby(["budget", "budget_numeric"], as_index=False)[
            [
                "effective_balance_degree",
                "train_routine",
                "train_follow_up",
                "train_biopsy",
            ]
        ]
        .mean()
        .sort_values("budget_numeric")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        schedule["budget_numeric"],
        schedule["effective_balance_degree"],
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("Labeled train samples")
    plt.ylabel("Effective balance degree")
    plt.ylim(-0.03, 1.03)
    plt.title("Progressive balance schedule")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_balance_degree.png", dpi=200)
    plt.close()

    # Normalized class composition across budgets.
    total = (
        schedule["train_routine"]
        + schedule["train_follow_up"]
        + schedule["train_biopsy"]
    ).to_numpy()
    x = np.arange(len(schedule))
    routine = schedule["train_routine"].to_numpy() / total
    follow = schedule["train_follow_up"].to_numpy() / total
    biopsy = schedule["train_biopsy"].to_numpy() / total
    plt.figure(figsize=(9, 5))
    plt.bar(x, routine, label="routine")
    plt.bar(x, follow, bottom=routine, label="follow_up")
    plt.bar(x, biopsy, bottom=routine + follow, label="biopsy")
    plt.xticks(x, schedule["budget"].astype(str).tolist())
    plt.xlabel("Label budget")
    plt.ylabel("Fraction of training subset")
    plt.title("Training-subset class composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_subset_class_composition.png", dpi=200)
    plt.close()

    # Predicted class fractions at the selected checkpoints. This exposes class collapse.
    plt.figure(figsize=(12, 4))
    for i, mode in enumerate(sorted(mean_df["mode"].unique())):
        ax = plt.subplot(1, len(mean_df["mode"].unique()), i + 1)
        sub = mean_df[mean_df["mode"] == mode].sort_values("budget_numeric")
        for class_name in CLASS_NAMES:
            ax.plot(
                sub["budget_numeric"],
                sub[f"pred_{class_name}_fraction"],
                marker="o",
                label=class_name,
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(mode)
        ax.set_xlabel("Labeled samples")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Predicted test fraction")
        if i == len(mean_df["mode"].unique()) - 1:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_predicted_class_fractions.png", dpi=200)
    plt.close()


def write_aggregate_outputs(
    results: list[dict[str, Any]],
    cfg: ExperimentConfig,
    train_full_n: int,
) -> None:
    out_dir = Path(cfg.output_dir)
    df = _summary_rows(results, train_full_n)
    df.to_csv(out_dir / "summary_table.csv", index=False)
    _mean_summary(df).to_csv(out_dir / "summary_mean.csv", index=False)
    save_json(results, out_dir / "summary_results.json")
    try:
        plot_summary_outputs(df, train_full_n, out_dir)
    except Exception as exc:
        print(f"WARNING: failed to plot one or more aggregate summaries: {exc}", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MedJEPA supervised transfer / label-efficiency experiment V2."
    )
    p.add_argument("--checkpoint", required=True, type=str)
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
    p.add_argument("--backbone-num-classes", default=None, type=int)
    p.add_argument(
        "--auto-model-from-checkpoint",
        dest="auto_model_from_checkpoint",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-auto-model-from-checkpoint",
        dest="auto_model_from_checkpoint",
        action="store_false",
    )
    p.add_argument("--drop-path-rate", default=0.1, type=float)
    p.add_argument("--image-height", default=512, type=int)
    p.add_argument("--image-width", default=512, type=int)
    p.add_argument("--memmap-dtype", default="uint16", type=str)
    p.add_argument(
        "--normalize-mode",
        default="uint16",
        choices=["uint16", "per_image_percentile"],
    )
    p.add_argument("--percentile-low", default=1.0, type=float)
    p.add_argument("--percentile-high", default=99.0, type=float)

    p.add_argument(
        "--subset-sizes",
        nargs="+",
        required=True,
        help="Label budgets, e.g. 2000 5000 10k 50k full.",
    )
    p.add_argument(
        "--subset-strategy",
        default="progressive",
        choices=["progressive", "balanced", "natural"],
        help=(
            "progressive: V2 nested schedule from balanced small subsets to natural full data; "
            "balanced: as balanced as uniquely feasible; natural: natural class proportions."
        ),
    )
    p.add_argument(
        "--balance-curve",
        default="log",
        choices=["log", "linear"],
        help="How progressive balance degree falls from 1 to 0 with growing budget.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["random", "frozen_jepa", "finetune_jepa"],
        choices=["random", "frozen_jepa", "finetune_jepa"],
    )

    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--patience", default=15, type=int)
    p.add_argument("--head-warmup-epochs", default=5, type=int)
    p.add_argument(
        "--random-min-epoch-fraction",
        default=0.5,
        type=float,
        help="Random mode cannot early-stop before ceil(epochs * this fraction).",
    )
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--eval-batch-size", default=256, type=int)
    p.add_argument("--num-workers", default=4, type=int)

    p.add_argument("--random-learning-rate", default=3e-4, type=float)
    p.add_argument("--random-warmup-epochs", default=5, type=int)
    p.add_argument("--random-warmup-start-factor", default=0.1, type=float)
    p.add_argument("--head-learning-rate", default=3e-4, type=float)
    p.add_argument("--finetune-head-learning-rate", default=1e-4, type=float)
    p.add_argument("--backbone-learning-rate", default=3e-5, type=float)
    p.add_argument("--min-learning-rate", default=1e-6, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--grad-clip-norm", default=0.0, type=float)
    p.add_argument("--label-smoothing", default=0.0, type=float)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument(
        "--selection-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "macro_f1"],
        help="Validation metric used to select/checkpoint the best epoch.",
    )

    p.add_argument(
        "--supervised-aug-mode",
        default="mild",
        choices=["config", "mild", "none"],
    )
    p.add_argument("--hflip-p", default=0.5, type=float)
    p.add_argument("--max-rotation-deg", default=5.0, type=float)

    p.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Base seed. With --num-seeds N, V2 uses seed, seed+1, ..., seed+N-1.",
    )
    p.add_argument(
        "--num-seeds",
        default=1,
        type=int,
        help="Number of repeated seeds. Default 1 keeps V2 runtime comparable to V1.",
    )
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-save-train-used-csv", action="store_true")
    p.add_argument("--no-run-plots", action="store_true")
    return p.parse_args()


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if cfg.patience <= 0:
        raise ValueError("--patience must be > 0")
    if not (0.0 <= cfg.random_min_epoch_fraction <= 1.0):
        raise ValueError("--random-min-epoch-fraction must be in [0, 1]")
    if cfg.num_seeds <= 0:
        raise ValueError("--num-seeds must be >= 1")
    if cfg.head_warmup_epochs < 0:
        raise ValueError("--head-warmup-epochs must be >= 0")
    if cfg.random_warmup_epochs < 0:
        raise ValueError("--random-warmup-epochs must be >= 0")


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
        balance_curve=args.balance_curve,
        modes=tuple(args.modes),
        epochs=args.epochs,
        patience=args.patience,
        head_warmup_epochs=args.head_warmup_epochs,
        random_min_epoch_fraction=args.random_min_epoch_fraction,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        random_learning_rate=args.random_learning_rate,
        random_warmup_epochs=args.random_warmup_epochs,
        random_warmup_start_factor=args.random_warmup_start_factor,
        head_learning_rate=args.head_learning_rate,
        finetune_head_learning_rate=args.finetune_head_learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        use_class_weights=not bool(args.no_class_weights),
        selection_metric=args.selection_metric,
        supervised_aug_mode=args.supervised_aug_mode,
        hflip_p=args.hflip_p,
        max_rotation_deg=args.max_rotation_deg,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
        amp=not bool(args.no_amp),
        save_train_used_csv=not bool(args.no_save_train_used_csv),
        save_run_plots=not bool(args.no_run_plots),
    )
    validate_config(cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hints: dict[str, Any] = {}
    if cfg.auto_model_from_checkpoint:
        checkpoint_hints = maybe_update_config_from_checkpoint(cfg)
        print(
            "Auto model config from checkpoint:",
            json.dumps(checkpoint_hints, indent=2, default=str),
            flush=True,
        )

    save_json(asdict(cfg), out_dir / "experiment_config.json")
    if checkpoint_hints:
        save_json(checkpoint_hints, out_dir / "checkpoint_model_hints.json")

    set_seed(cfg.seed)
    device = torch.device(
        cfg.device
        if torch.cuda.is_available() and cfg.device.startswith("cuda")
        else "cpu"
    )
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device), flush=True)

    with open(cfg.aug_config, "r", encoding="utf-8") as f:
        aug_cfg = json.load(f)
    save_json(aug_cfg, out_dir / "augmentation_config_used.json")

    full_df, train_df, val_df, test_df = load_splits(cfg)
    leakage = verify_no_patient_leakage(train_df, val_df, test_df)

    expected_bytes = (
        len(full_df)
        * cfg.image_height
        * cfg.image_width
        * np.dtype(cfg.memmap_dtype).itemsize
    )
    actual_bytes = Path(cfg.bin_path).stat().st_size
    print(
        f"BIN check: expected={expected_bytes/1024**3:.3f} GiB "
        f"actual={actual_bytes/1024**3:.3f} GiB "
        f"match={expected_bytes == actual_bytes}",
        flush=True,
    )
    if expected_bytes != actual_bytes:
        raise RuntimeError("Full CSV and BIN do not match. Refusing to train.")

    split_summary = {
        "patient_leakage": leakage,
        "train": {
            "rows": len(train_df),
            "classes": train_df["collapsed_birads"].value_counts().to_dict(),
        },
        "val": {
            "rows": len(val_df),
            "classes": val_df["collapsed_birads"].value_counts().to_dict(),
        },
        "test": {
            "rows": len(test_df),
            "classes": test_df["collapsed_birads"].value_counts().to_dict(),
        },
    }
    save_json(split_summary, out_dir / "split_summary.json")
    print(json.dumps(split_summary, indent=2), flush=True)

    # ------------------------------------------------------------------
    # Create the timm ViT ONCE for the whole process.
    #
    # On the current single-GPU Determined tasks, constructing a second ViT after a
    # completed training/DataLoader run can crash natively inside timm's trunc_normal_
    # initialization. Every experiment below therefore reuses this model object.
    #
    # The snapshot is taken from the same cfg.seed initialization that the original
    # script used for every run when --num-seeds=1, so the experiment semantics are
    # preserved for the current Hologic/GE runs.
    # ------------------------------------------------------------------
    set_seed(cfg.seed)
    print("Creating shared ViT model once for all transfer runs...", flush=True)
    shared_model = ViTBackboneClassifier(
        cfg.backbone,
        cfg.image_size,
        cfg.embedding_dim,
        cfg.drop_path_rate,
        len(CLASS_NAMES),
        cfg.backbone_num_classes,
    ).to(device)
    cfg.embedding_dim = int(shared_model.embedding_dim)
    initial_model_state = clone_state_dict_to_cpu(shared_model)
    print(
        f"Shared model ready: backbone={cfg.backbone}, "
        f"embedding_dim={cfg.embedding_dim}. "
        "No further timm.create_model() calls will be made.",
        flush=True,
    )

    if cfg.num_seeds > 1:
        print(
            "WARNING: model-reuse stability mode uses the same initial model weights "
            "for all seed repetitions. Dataset subset/order randomness still follows each "
            "run seed. For fully independent model-initialization seeds, run separate "
            "processes/tasks with --num-seeds 1 and different --seed values.",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    start = time.perf_counter()

    for seed_index in range(cfg.num_seeds):
        run_seed = cfg.seed + seed_index
        print(f"\n######## Seed repetition {seed_index + 1}/{cfg.num_seeds}: seed={run_seed} ########", flush=True)

        subset_plan = build_subset_plan(
            train_df,
            cfg.subset_sizes,
            cfg.subset_strategy,
            cfg.balance_curve,
            run_seed,
        )
        subset_plan_serializable = {
            k: {kk: vv for kk, vv in v.items() if kk != "df"}
            for k, v in subset_plan.items()
        }
        save_json(
            subset_plan_serializable,
            out_dir / f"subset_plan_seed_{run_seed}.json",
        )
        print(
            "Subset schedule:",
            json.dumps(
                {
                    k: {
                        "rows": v["rows"],
                        "raw_balance_degree": v["raw_balance_degree"],
                        "effective_balance_degree": v["effective_balance_degree"],
                        "actual_counts": v["actual_counts"],
                    }
                    for k, v in subset_plan.items()
                },
                indent=2,
            ),
            flush=True,
        )

        for budget in cfg.subset_sizes:
            meta = subset_plan[budget_name(budget)]
            train_used = meta["df"]
            for mode in cfg.modes:
                result = train_one_run(
                    cfg,
                    mode,
                    budget,
                    train_used,
                    meta,
                    val_df,
                    test_df,
                    len(full_df),
                    aug_cfg,
                    device,
                    run_seed,
                    seed_index,
                    shared_model,
                    initial_model_state,
                )
                results.append(result)
                write_aggregate_outputs(results, cfg, len(train_df))

    elapsed = time.perf_counter() - start
    save_json(
        {
            "wall_time_sec": elapsed,
            "num_runs": len(results),
            "num_seeds": cfg.num_seeds,
        },
        out_dir / "timing.json",
    )
    write_aggregate_outputs(results, cfg, len(train_df))
    print("Finished all V2 transfer experiments.", flush=True)
    print(
        pd.read_csv(out_dir / "summary_table.csv").to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
