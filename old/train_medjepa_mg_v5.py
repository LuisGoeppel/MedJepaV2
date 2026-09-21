#!/usr/bin/env python3
"""
MedJEPA LeJEPA training script v5.

Purpose:
- Train LeJEPA/SIGReg on pre-created patient-disjoint split CSVs.
- Read raw uint16 512x512 image memmap by original_index.
- Load augmentation policy from a JSON config.
- Save final trained encoder/checkpoint and training history.
- Run exactly one diagnostic linear probe on collapsed BI-RADS after training.

No split creation and no online probe. Adds DDP training, gradient accumulation, FP32 SSL loss outside autocast, reduced per-batch diagnostics, periodic checkpoints, and optional top-corner watermark masking after foreground crop. Final probe/PCA are optional because larger analysis is usually done by separate scripts.

Example:
    python train_medjepa_mg_focused.py \
      --full-csv /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg/mg-only-all.csv \
      --bin /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg/mg-only-all.bin \
      --train-csv /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg/splits/mg_train.csv \
      --val-csv /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg/splits/mg_val.csv \
      --test-csv /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg/splits/mg_test.csv \
      --aug-config /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg/mg_lejepa_aug_smoke_v1.json \
      --output-dir /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/runs/mg_lejepa_focused \
      --epochs 50 --batch-size 128 --num-workers 8 --probe-epochs 50
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler, DistributedSampler, Sampler
from torchvision.transforms import InterpolationMode, RandomResizedCrop
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("Missing dependency: timm. Install with: pip install timm") from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
except ImportError as exc:
    raise ImportError("Missing dependency: scikit-learn. Install with: pip install scikit-learn") from exc


@dataclass
class TrainConfig:
    # Runtime
    seed: int = 42
    num_workers: int = 8
    output_dir: str = "outputs_medjepa_focused"

    # Data
    full_csv_path: str = ""
    train_csv_path: str = ""
    val_csv_path: str = ""
    test_csv_path: str = ""
    bin_path: str = ""
    aug_config_path: str = ""
    image_height: int = 512
    image_width: int = 512
    memmap_dtype: str = "uint16"
    image_size: int = 0  # 0 means: read image.output_size from augmentation config
    normalize_mode: str = "uint16"  # uint16 or per_image_percentile
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    # SSL views
    num_views: int = 4

    # Model
    backbone_name: str = "vit_small_patch8_224"
    backbone_output_dim: int = 512
    projection_dim: int = 16
    projector_hidden_dim: int = 2048
    drop_path_rate: float = 0.1

    # LeJEPA/SIGReg loss
    lambda_sigreg: float = 0.02
    sigreg_knots: int = 17
    sigreg_num_projections: int = 256
    # Projection normalization used for the invariance term only.
    # Choices: none, unit, sqrt_dim. sqrt_dim uses F.normalize(proj) * sqrt(D),
    # making invariance scale-invariant while leaving SIGReg on raw projections
    # so it can still penalize raw projector scale/variance collapse.
    projection_normalization: str = "sqrt_dim"

    # Optimization
    epochs: int = 50
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 5e-2
    eta_min: float = 1e-5
    warmup_epochs: int = 1
    use_weighted_sampler: bool = False
    # Optional batch construction. "random" keeps the old behavior.
    # "balanced_collapsed" samples each physical global batch with balanced collapsed BI-RADS labels.
    batch_construction: str = "random"
    grad_clip_norm: float = 0.0
    grad_accum_steps: int = 1

    # Checkpointing and training diagnostics
    checkpoint_every_epochs: int = 50  # 0 disables periodic checkpoints; final checkpoint is always saved
    diagnostic_every_batches: int = 50  # compute proj/emb diagnostics every N batches; 1 means every batch

    # Optional built-in final analysis. For large runs we normally use separate scripts.
    run_final_analysis: bool = False
    resume_checkpoint_path: str = ""
    resume_weights_only: bool = False
    use_sync_batchnorm: bool = False

    # Linear probe
    probe_epochs: int = 50
    probe_learning_rate: float = 1e-3
    probe_weight_decay: float = 1e-7
    probe_train_max_samples: int = 60000
    probe_use_class_weights: bool = True

    # Evaluation / PCA
    pca_max_samples: int = 3000

    # Multi-GPU. "auto" uses DataParallel if multiple CUDA devices are visible.
    multi_gpu: str = "auto"

    # Timing. CUDA synchronization gives more accurate GPU timing but adds overhead.
    timing_cuda_synchronize: bool = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Focused LeJEPA training on MedJEPA split CSVs.")
    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--bin", required=True, type=str)
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--val-csv", required=True, type=str)
    p.add_argument("--test-csv", required=True, type=str)
    p.add_argument("--aug-config", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--probe-epochs", type=int, default=50)
    p.add_argument("--probe-train-max-samples", type=int, default=60000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=0, help="0 means read image.output_size from augmentation config.")
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--backbone", type=str, default="vit_small_patch8_224")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Default lowered from earlier 2e-3 for stability on MG.")
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--projection-dim", type=int, default=16)
    p.add_argument("--projector-hidden-dim", type=int, default=2048)
    p.add_argument("--lambda-sigreg", type=float, default=0.02)
    p.add_argument("--projection-normalization", type=str, default="sqrt_dim", choices=["none", "unit", "sqrt_dim"],
                   help="Normalize projector outputs before the invariance term only. SIGReg is computed on raw projections. Default sqrt_dim uses F.normalize(proj)*sqrt(D).")
    p.add_argument("--normalize-mode", type=str, default="uint16", choices=["uint16", "per_image_percentile"])
    p.add_argument("--batch-weighting", action="store_true", help="Use collapsed-BI-RADS weighted sampler for SSL batches. Non-DDP legacy option.")
    p.add_argument("--batch-construction", type=str, default="random", choices=["random", "balanced_collapsed"],
                   help="Optional balanced physical-batch construction. Default random keeps old sampling behavior.")
    p.add_argument("--grad-clip-norm", type=float, default=0.0, help="0 disables gradient clipping.")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over this many physical batches before optimizer step. Effective global batch = batch_size * grad_accum_steps.")
    p.add_argument("--checkpoint-every-epochs", type=int, default=50,
                   help="Save a resumable checkpoint every N epochs. Use 0 to disable periodic checkpoints.")
    p.add_argument("--diagnostic-every-batches", type=int, default=50,
                   help="Compute expensive proj/embedding diagnostics every N batches instead of every batch. Use 1 for every batch.")
    p.add_argument("--run-final-analysis", action="store_true",
                   help="Run built-in final collapsed-BI-RADS linear probe and PCA after training. Usually disabled for large runs.")
    p.add_argument("--resume-checkpoint", type=str, default="",
                   help="Resume from a checkpoint saved by this script.")
    p.add_argument("--resume-weights-only", action="store_true",
                   help="Load only model weights from --resume-checkpoint and start a fresh optimizer/scheduler/history.")
    p.add_argument("--sync-batchnorm", action="store_true",
                   help="Convert BatchNorm layers to SyncBatchNorm before DDP wrapping. Off by default.")
    p.add_argument("--no-probe-class-weights", action="store_true")
    p.add_argument("--pca-max-samples", type=int, default=3000)
    p.add_argument("--multi-gpu", type=str, default="auto", choices=["auto", "none", "data_parallel", "ddp"],
                   help="auto uses DDP when launched with torchrun, otherwise DataParallel if >1 CUDA GPU is visible. Batch size is global.")
    p.add_argument("--no-cuda-timing-sync", action="store_true",
                   help="Disable torch.cuda.synchronize() around timed regions. Faster, but GPU timing becomes approximate.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        full_csv_path=args.full_csv,
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        test_csv_path=args.test_csv,
        bin_path=args.bin,
        aug_config_path=args.aug_config,
        image_size=args.image_size,
        normalize_mode=args.normalize_mode,
        num_views=args.num_views,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        warmup_epochs=args.warmup_epochs,
        projection_dim=args.projection_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        lambda_sigreg=args.lambda_sigreg,
        projection_normalization=args.projection_normalization,
        use_weighted_sampler=args.batch_weighting,
        batch_construction=args.batch_construction,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=max(1, args.grad_accum_steps),
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        diagnostic_every_batches=max(1, args.diagnostic_every_batches),
        run_final_analysis=args.run_final_analysis,
        resume_checkpoint_path=args.resume_checkpoint,
        resume_weights_only=args.resume_weights_only,
        use_sync_batchnorm=args.sync_batchnorm,
        probe_epochs=args.probe_epochs,
        probe_train_max_samples=args.probe_train_max_samples,
        probe_use_class_weights=not args.no_probe_class_weights,
        pca_max_samples=args.pca_max_samples,
        multi_gpu=args.multi_gpu,
        timing_cuda_synchronize=not args.no_cuda_timing_sync,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def create_dirs(output_dir: str) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "models": root / "models",
        "plots": root / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# -----------------------------
# Labels, split verification
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
    df["target_collapsed"] = df["collapsed_birads"].map({"routine": 0, "follow_up": 1, "biopsy": 2}).astype(int)
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


def verify_no_patient_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if "patient" not in train_df.columns:
        rank0_print("WARNING: no patient column found; cannot verify patient leakage.")
        return
    train_p = set(train_df["patient"].astype(str))
    val_p = set(val_df["patient"].astype(str))
    test_p = set(test_df["patient"].astype(str))
    overlaps = {"train_val": len(train_p & val_p), "train_test": len(train_p & test_p), "val_test": len(val_p & test_p)}
    rank0_print("Patient leakage check:", overlaps)
    if any(v != 0 for v in overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")


def load_splits(cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_raw = read_csv_clean(cfg.full_csv_path)
    full_df = prepare_labels(full_raw)
    if "original_index" not in full_df.columns:
        full_df = full_df.reset_index(drop=False).rename(columns={"index": "original_index"})

    train_df = prepare_labels(ensure_original_index(read_csv_clean(cfg.train_csv_path), full_raw, "train"))
    val_df = prepare_labels(ensure_original_index(read_csv_clean(cfg.val_csv_path), full_raw, "val"))
    test_df = prepare_labels(ensure_original_index(read_csv_clean(cfg.test_csv_path), full_raw, "test"))

    verify_no_patient_leakage(train_df, val_df, test_df)
    rank0_print("Split summary:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_pat = df["patient"].nunique() if "patient" in df.columns else "?"
        rank0_print(f"  {name}: rows={len(df):,}, patients={n_pat}, collapsed={df['collapsed_birads'].value_counts().to_dict()}")
    return full_df, train_df, val_df, test_df


# -----------------------------
# Augmentation from JSON config
# -----------------------------

def deep_get(dct: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class ConfigurableMGAugmentation(nn.Module):
    """Config-driven mammography augmentation.

    Supports the v2 JSON keys used in mg_lejepa_aug_v2.json. Operations that
    require OpenCV, such as CLAHE, are skipped with a one-time warning if cv2 is
    not installed. All transforms operate on one-channel float tensors in [0, 1].
    """

    _warned_no_cv2 = False

    def __init__(self, aug_cfg: dict[str, Any], image_size: int, train: bool):
        super().__init__()
        self.cfg = aug_cfg
        self.image_size = image_size
        self.train = train

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,H,W] in [0,1]
        x = self._foreground_crop(x)
        x = self._mask_top_corner(x)

        if self.train:
            # v2 config can resize to a larger intermediate canvas, then crop to output size.
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

        # Evaluation transform: only deterministic foreground crop + final resize.
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
        """Mask likely watermark / metadata text in the top corner after foreground crop.

        Enabled by this augmentation config block:
        {
          "preprocessing": {
            "top_right_corner_mask": {
              "enabled": true,
              "frac_x": 0.30,
              "frac_y": 0.12,
              "value": 0.0,
              "foreground_threshold": 0.0001,
              "min_component_area_frac": 0.0002,
              "skip_if_single_component": true
            }
          }
        }

        The side is chosen dynamically: the mask is applied to the top side with
        less foreground tissue, so left/right mammograms are handled consistently.
        """
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
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask_np,
                    connectivity=8,
                )

                min_area = max(1, int(round(h * w * min_component_area_frac)))

                relevant_components = 0
                for label_idx in range(1, num_labels):
                    area = int(stats[label_idx, cv2.CC_STAT_AREA])
                    if area >= min_area:
                        relevant_components += 1

                if relevant_components <= 1:
                    return x

            except Exception:
                # If OpenCV is not available or connected components fail, still apply
                # the deterministic top-corner side heuristic below.
                pass

        left_half = foreground[:, : w // 2]
        right_half = foreground[:, w // 2 :]

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
            clip_limit = float(c.get("clip_limit", 2.0))
            tile_grid_size = tuple(c.get("tile_grid_size", [8, 8]))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
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
            bits = int(c.get("bits", 6))
            bits = max(1, min(8, bits))
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
                x[:, i:i+erase_h, j:j+erase_w] = value
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
        x[:, i:i+ch, j:j+cw] = value
        return x


# -----------------------------
# Dataset
# -----------------------------

class MedJEPADataset(Dataset):
    def __init__(self, df: pd.DataFrame, bin_path: str | Path, full_num_rows: int,
                 image_shape: tuple[int, int], dtype: str, transform: nn.Module,
                 num_views: int, normalize_mode: str, percentile_low: float, percentile_high: float):
        self.df = df.reset_index(drop=True)
        self.bin_path = Path(bin_path)
        self.full_num_rows = full_num_rows
        self.image_shape = image_shape
        self.dtype = np.dtype(dtype)
        self.transform = transform
        self.num_views = num_views
        self.normalize_mode = normalize_mode
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
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
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_tensor(int(row["original_index"]))
        views = torch.stack([self.transform(x.clone()) for _ in range(self.num_views)])
        label = int(row["target_collapsed"])
        return views, label


def collate_batch(batch):
    views = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return views, labels


def make_weighted_sampler(df: pd.DataFrame, label_col: str = "target_collapsed") -> WeightedRandomSampler:
    counts = df[label_col].value_counts().to_dict()
    weights = df[label_col].map(lambda y: 1.0 / counts[int(y)]).astype(float).values
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


class DistributedBalancedLabelBatchSampler(Sampler[list[int]]):
    """Balanced physical-batch sampler with DDP slicing.

    This is optional and disabled by default. It samples with replacement to make
    each global physical batch approximately balanced over the given label column.
    Each rank receives a disjoint slice of that global batch.
    """
    def __init__(self, labels: np.ndarray, per_process_batch_size: int, world_size: int = 1, rank: int = 0,
                 seed: int = 42, drop_last: bool = True):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.per_process_batch_size = int(per_process_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.global_batch_size = self.per_process_batch_size * self.world_size
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        self.classes = np.array(sorted(np.unique(self.labels).tolist()), dtype=np.int64)
        self.class_to_indices = {int(c): np.where(self.labels == c)[0] for c in self.classes}
        for c, idx in self.class_to_indices.items():
            if len(idx) == 0:
                raise ValueError(f"No indices for class {c}")
        self.num_batches = len(self.labels) // self.global_batch_size if self.drop_last else math.ceil(len(self.labels) / self.global_batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n_cls = len(self.classes)
        base = self.global_batch_size // n_cls
        rem = self.global_batch_size % n_cls
        for _ in range(self.num_batches):
            batch = []
            class_order = self.classes.copy()
            rng.shuffle(class_order)
            for i, c in enumerate(class_order):
                n = base + (1 if i < rem else 0)
                idx_pool = self.class_to_indices[int(c)]
                chosen = rng.choice(idx_pool, size=n, replace=True)
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            start = self.rank * self.per_process_batch_size
            end = start + self.per_process_batch_size
            yield batch[start:end]


def make_balanced_train_loader(dataset: MedJEPADataset, cfg: TrainConfig, per_process_batch_size: int,
                               world_size: int, rank: int) -> DataLoader:
    labels = dataset.df["target_collapsed"].to_numpy(dtype=np.int64)
    batch_sampler = DistributedBalancedLabelBatchSampler(
        labels=labels,
        per_process_batch_size=per_process_batch_size,
        world_size=world_size,
        rank=rank,
        seed=cfg.seed,
        drop_last=True,
    )
    kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def make_loader(dataset: Dataset, cfg: TrainConfig, batch_size: int, shuffle: bool, drop_last: bool,
                sampler: Optional[WeightedRandomSampler] = None) -> DataLoader:
    if sampler is not None:
        shuffle = False
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                  drop_last=drop_last, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                  collate_fn=collate_batch)
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# -----------------------------
# Model and loss
# -----------------------------

class ViTEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig):
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
        # Batch-first projection shape is required for correct torch.nn.DataParallel gathering.
        # Shape: [B, V, D]. Older versions returned [V, B, D], which breaks when
        # DataParallel sees unequal per-GPU batch sizes and also gathers along the wrong axis.
        proj = self.proj(emb).reshape(b, v, -1)
        return emb, proj


def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    # proj: [B, V, D]. Make all views of the same image agree.
    return (proj.mean(dim=1, keepdim=True) - proj).square().mean()


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_projections: int = 256):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_projections = num_projections
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: [B, V, D]. SIGReg is applied to the pooled set of all projected views.
        z = proj.reshape(-1, proj.size(-1))  # [B*V, D]
        dim = z.size(-1)
        n = z.size(0)
        A = torch.randn(dim, self.num_projections, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        x_t = (z @ A).unsqueeze(-1) * self.t.to(z.device, dtype=z.dtype)  # [N, P, K]
        err = (x_t.cos().mean(dim=0) - self.phi.to(z.device, dtype=z.dtype)).square()
        err = err + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights.to(z.device, dtype=z.dtype)) * n
        return statistic.mean()


def normalize_projection_for_loss(proj: torch.Tensor, mode: str = "sqrt_dim") -> torch.Tensor:
    """Normalize projector output before the SSL loss.

    - none: old behavior.
    - unit: unit L2 norm per projected view.
    - sqrt_dim: unit L2 norm scaled by sqrt(D), matching a standard-normal
      per-vector norm scale more closely for SIGReg's target distribution.
    """
    if mode == "none":
        return proj
    z = F.normalize(proj, dim=-1, eps=1e-6)
    if mode == "sqrt_dim":
        z = z * math.sqrt(float(proj.size(-1)))
    elif mode != "unit":
        raise ValueError(f"Unknown projection_normalization mode: {mode}")
    return z


def _effective_rank(x: torch.Tensor) -> torch.Tensor:
    """Entropy effective rank of the centered feature matrix. Cheap enough for sampled diagnostics."""
    x = x.float()
    if x.ndim != 2 or x.size(0) < 2:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on [N, D] is stable for our diagnostic batches; no gradients are used.
    s = torch.linalg.svdvals(x)
    eig = s.square()
    total = eig.sum()
    if total <= 0:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    p = eig / total.clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return torch.exp(entropy)


def representation_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().float().reshape(-1, x.size(-1))
    std = x.std(dim=0).mean()
    norm = x.norm(dim=1).mean()
    erank = _effective_rank(x)
    return std, norm, erank


class LeJEPALoss(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.lambda_sigreg = cfg.lambda_sigreg
        self.projection_normalization = cfg.projection_normalization
        self.sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_num_projections)

    def forward(self, proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Important v5_modified behavior:
        # - Invariance is computed on normalized projections, so shrinking the raw
        #   projector output cannot trivially reduce the invariance loss.
        # - SIGReg is computed on the raw projections, so it can still see and
        #   counteract raw projector variance / scale collapse.
        proj_inv = normalize_projection_for_loss(proj, self.projection_normalization)
        inv = invariance_loss(proj_inv)
        sig = self.sigreg(proj)
        total = sig * self.lambda_sigreg + inv * (1.0 - self.lambda_sigreg)
        return total, inv, sig

    def normalize_for_diagnostics(self, proj: torch.Tensor) -> torch.Tensor:
        # This diagnostic represents the projection actually used for invariance.
        return normalize_projection_for_loss(proj, self.projection_normalization)


def build_optimizer_scheduler(net: nn.Module, loader: DataLoader, cfg: TrainConfig):
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer_steps_per_epoch = max(1, math.ceil(len(loader) / max(1, cfg.grad_accum_steps)))
    warmup_steps = max(0, optimizer_steps_per_epoch * max(0, cfg.warmup_epochs))
    total_steps = max(1, optimizer_steps_per_epoch * cfg.epochs)
    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.eta_min),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.eta_min)
    return optimizer, scheduler


# -----------------------------
# Training and probe
# -----------------------------

def maybe_cuda_synchronize(device: torch.device, cfg: TrainConfig) -> None:
    if device.type == "cuda" and cfg.timing_cuda_synchronize:
        torch.cuda.synchronize()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def rank0_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed(cfg: TrainConfig) -> tuple[bool, int, int, int, torch.device]:
    """Initialize DDP when launched with torchrun.

    Batch size in cfg remains a global batch size. In DDP, each process uses
    cfg.batch_size // world_size samples per physical step.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if cfg.multi_gpu == "ddp" and world_size <= 1:
        raise RuntimeError("--multi-gpu ddp requires launching with torchrun so WORLD_SIZE > 1.")

    use_ddp = world_size > 1 and cfg.multi_gpu in {"auto", "ddp"}
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        return True, rank, world_size, local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def reduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def save_training_checkpoint(net: nn.Module, optimizer, scheduler, epoch: int, cfg: TrainConfig,
                             aug_cfg: dict[str, Any], history: dict[str, list[float]],
                             dirs: dict[str, Path], final: bool = False) -> None:
    if not is_main_process():
        return
    model_to_save = unwrap_model(net)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(cfg),
        "augmentation_config": aug_cfg,
        "history": history,
    }
    if final:
        ckpt_path = dirs["models"] / "final_lejepa_checkpoint.pt"
        state_path = dirs["models"] / "final_lejepa_encoder_state_dict.pt"
    else:
        ckpt_path = dirs["models"] / f"checkpoint_epoch_{epoch:04d}.pt"
        state_path = dirs["models"] / f"encoder_state_dict_epoch_{epoch:04d}.pt"
    torch.save(payload, ckpt_path)
    torch.save(model_to_save.state_dict(), state_path)


def train_lejepa(net: nn.Module, loss_fn: nn.Module, loader: DataLoader, optimizer, scheduler,
                 cfg: TrainConfig, device: torch.device, dirs: dict[str, Path],
                 aug_cfg: dict[str, Any], start_epoch: int = 0,
                 existing_history: Optional[dict[str, list[float]]] = None) -> dict[str, list[float]]:
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    accum_steps = max(1, int(cfg.grad_accum_steps))
    diagnostic_every = max(1, int(cfg.diagnostic_every_batches))

    default_history = {
        "lejepa": [], "invariance": [], "sigreg": [],
        # raw projector diagnostics keep the old proj_std/proj_norm aliases for compatibility.
        "proj_std": [], "proj_norm": [],
        "raw_proj_std": [], "raw_proj_norm": [], "raw_proj_effective_rank": [],
        "loss_proj_std": [], "loss_proj_norm": [], "loss_proj_effective_rank": [],
        "emb_std": [], "emb_norm": [], "emb_effective_rank": [],
        "lr": [],
        "epoch_time_sec": [],
        "data_wait_and_augmentation_time_sec": [],
        "h2d_transfer_time_sec": [],
        "forward_loss_time_sec": [],
        "backward_optimizer_time_sec": [],
        "metrics_bookkeeping_time_sec": [],
        "data_wait_and_augmentation_time_per_batch_sec": [],
        "h2d_transfer_time_per_batch_sec": [],
        "forward_loss_time_per_batch_sec": [],
        "backward_optimizer_time_per_batch_sec": [],
        "metrics_bookkeeping_time_per_batch_sec": [],
        "data_wait_and_augmentation_fraction": [],
        "h2d_transfer_fraction": [],
        "forward_loss_fraction": [],
        "backward_optimizer_fraction": [],
        "metrics_bookkeeping_fraction": [],
        "num_batches": [],
        "num_optimizer_steps": [],
        "grad_accum_steps": [],
    }
    history = existing_history if existing_history is not None else default_history
    for key, value in default_history.items():
        history.setdefault(key, value)

    if start_epoch >= cfg.epochs:
        rank0_print(f"Resume checkpoint already reached epoch {start_epoch}; cfg.epochs={cfg.epochs}. Nothing to train.")
        return history

    for epoch in range(start_epoch, cfg.epochs):
        # DistributedSampler uses loader.sampler; custom balanced batch construction uses loader.batch_sampler.
        if hasattr(getattr(loader, "batch_sampler", None), "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        elif hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_start_time = time.perf_counter()
        net.train()

        metric_sums = torch.zeros(12, device=device, dtype=torch.float64)
        # [lejepa, invariance, sigreg,
        #  raw_proj_std, raw_proj_norm, raw_proj_erank,
        #  loss_proj_std, loss_proj_norm, loss_proj_erank,
        #  emb_std, emb_norm, emb_erank]
        count_tensor = torch.zeros(3, device=device, dtype=torch.float64)
        # [batch_count, diagnostic_count, optimizer_step_count]

        timing_sums = {
            "data_wait_and_augmentation_time_sec": 0.0,
            "h2d_transfer_time_sec": 0.0,
            "forward_loss_time_sec": 0.0,
            "backward_optimizer_time_sec": 0.0,
            "metrics_bookkeeping_time_sec": 0.0,
        }

        optimizer.zero_grad(set_to_none=True)
        loader_iter = iter(loader)
        fetch_start = time.perf_counter()
        pbar = tqdm(range(len(loader)), desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False,
                    disable=not is_main_process())

        for batch_idx in pbar:
            try:
                views, _ = next(loader_iter)
            except StopIteration:
                break

            fetch_end = time.perf_counter()
            timing_sums["data_wait_and_augmentation_time_sec"] += fetch_end - fetch_start

            t0 = time.perf_counter()
            views = views.to(device, non_blocking=True)
            maybe_cuda_synchronize(device, cfg)
            timing_sums["h2d_transfer_time_sec"] += time.perf_counter() - t0

            sync_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader))
            no_sync_ctx = net.no_sync() if (isinstance(net, DDP) and not sync_step) else contextlib.nullcontext()

            t0 = time.perf_counter()
            with no_sync_ctx:
                # Backbone/projector forward uses BF16 autocast for speed on A100.
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                    emb, proj = net(views)

                # Keep the SSL objective in FP32. SIGReg uses random projections and
                # sin/cos statistics, which are safer outside BF16 autocast.
                loss, inv, sig = loss_fn(proj.float())
                (loss / accum_steps).backward()
            maybe_cuda_synchronize(device, cfg)
            timing_sums["forward_loss_time_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if sync_step:
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                count_tensor[2] += 1.0
            maybe_cuda_synchronize(device, cfg)
            timing_sums["backward_optimizer_time_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.no_grad():
                metric_sums[0] += loss.detach().double()
                metric_sums[1] += inv.detach().double()
                metric_sums[2] += sig.detach().double()
                count_tensor[0] += 1.0

                # Expensive diagnostics are intentionally sampled. This avoids
                # repeated full-vector reductions and repeated CUDA scalar syncs.
                if (batch_idx % diagnostic_every == 0) or (batch_idx + 1 == len(loader)):
                    raw_std, raw_norm, raw_erank = representation_stats(proj)
                    if hasattr(loss_fn, "normalize_for_diagnostics"):
                        proj_loss_diag = loss_fn.normalize_for_diagnostics(proj.detach().float())
                    else:
                        proj_loss_diag = proj.detach().float()
                    loss_std, loss_norm, loss_erank = representation_stats(proj_loss_diag)
                    emb_std, emb_norm, emb_erank = representation_stats(emb)

                    metric_sums[3] += raw_std.double()
                    metric_sums[4] += raw_norm.double()
                    metric_sums[5] += raw_erank.double()
                    metric_sums[6] += loss_std.double()
                    metric_sums[7] += loss_norm.double()
                    metric_sums[8] += loss_erank.double()
                    metric_sums[9] += emb_std.double()
                    metric_sums[10] += emb_norm.double()
                    metric_sums[11] += emb_erank.double()
                    count_tensor[1] += 1.0
            timing_sums["metrics_bookkeeping_time_sec"] += time.perf_counter() - t0

            fetch_start = time.perf_counter()

        reduce_sum_tensor(metric_sums)
        reduce_sum_tensor(count_tensor)

        global_nb = max(1.0, float(count_tensor[0].item()))
        global_diag_nb = max(1.0, float(count_tensor[1].item()))
        global_opt_steps = int(count_tensor[2].item())

        history["lejepa"].append(float(metric_sums[0].item() / global_nb))
        history["invariance"].append(float(metric_sums[1].item() / global_nb))
        history["sigreg"].append(float(metric_sums[2].item() / global_nb))
        raw_proj_std = float(metric_sums[3].item() / global_diag_nb)
        raw_proj_norm = float(metric_sums[4].item() / global_diag_nb)
        raw_proj_erank = float(metric_sums[5].item() / global_diag_nb)
        loss_proj_std = float(metric_sums[6].item() / global_diag_nb)
        loss_proj_norm = float(metric_sums[7].item() / global_diag_nb)
        loss_proj_erank = float(metric_sums[8].item() / global_diag_nb)
        emb_std = float(metric_sums[9].item() / global_diag_nb)
        emb_norm = float(metric_sums[10].item() / global_diag_nb)
        emb_erank = float(metric_sums[11].item() / global_diag_nb)

        # Compatibility aliases: proj_* refers to the raw projector output, as in v4.
        history["proj_std"].append(raw_proj_std)
        history["proj_norm"].append(raw_proj_norm)
        history["raw_proj_std"].append(raw_proj_std)
        history["raw_proj_norm"].append(raw_proj_norm)
        history["raw_proj_effective_rank"].append(raw_proj_erank)
        history["loss_proj_std"].append(loss_proj_std)
        history["loss_proj_norm"].append(loss_proj_norm)
        history["loss_proj_effective_rank"].append(loss_proj_erank)
        history["emb_std"].append(emb_std)
        history["emb_norm"].append(emb_norm)
        history["emb_effective_rank"].append(emb_erank)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        epoch_time = float(time.perf_counter() - epoch_start_time)
        local_batches = max(1, int(len(loader)))
        history["epoch_time_sec"].append(epoch_time)
        history["num_batches"].append(int(global_nb))
        history["num_optimizer_steps"].append(global_opt_steps)
        history["grad_accum_steps"].append(accum_steps)

        for k, v in timing_sums.items():
            v = float(v)
            history[k].append(v)
            history[k.replace("_time_sec", "_time_per_batch_sec")].append(v / max(1, local_batches))

        denom = max(epoch_time, 1e-12)
        history["data_wait_and_augmentation_fraction"].append(float(timing_sums["data_wait_and_augmentation_time_sec"] / denom))
        history["h2d_transfer_fraction"].append(float(timing_sums["h2d_transfer_time_sec"] / denom))
        history["forward_loss_fraction"].append(float(timing_sums["forward_loss_time_sec"] / denom))
        history["backward_optimizer_fraction"].append(float(timing_sums["backward_optimizer_time_sec"] / denom))
        history["metrics_bookkeeping_fraction"].append(float(timing_sums["metrics_bookkeeping_time_sec"] / denom))

        if is_main_process():
            print(
                f"Epoch {epoch + 1:03d} | LeJEPA {history['lejepa'][-1]:.4f} | "
                f"Inv {history['invariance'][-1]:.4f} | SIGReg {history['sigreg'][-1]:.4f} | "
                f"RawProjStd {history['raw_proj_std'][-1]:.4f} | LossProjStd {history['loss_proj_std'][-1]:.4f} | "
                f"EmbNorm {history['emb_norm'][-1]:.2f} | "
                f"LR {history['lr'][-1]:.2e} | OptSteps {global_opt_steps} | Time {epoch_time/60:.2f} min | "
                f"Data/Aug {timing_sums['data_wait_and_augmentation_time_sec']/60:.2f} min | "
                f"Fwd+Loss+Bwd {timing_sums['forward_loss_time_sec']/60:.2f} min | "
                f"Opt {timing_sums['backward_optimizer_time_sec']/60:.2f} min",
                flush=True,
            )
            save_json(history, dirs["metrics"] / "training_history.json")

        epoch_num = epoch + 1
        if cfg.checkpoint_every_epochs and cfg.checkpoint_every_epochs > 0 and epoch_num % cfg.checkpoint_every_epochs == 0:
            save_training_checkpoint(net, optimizer, scheduler, epoch_num, cfg, aug_cfg, history, dirs, final=False)
            if is_main_process():
                print(f"Saved periodic checkpoint at epoch {epoch_num}.", flush=True)

    return history

def plot_training_history(history: dict[str, list[float]], dirs: dict[str, Path]) -> None:
    epochs = np.arange(1, len(history["lejepa"]) + 1)
    plt.figure(figsize=(16, 4))
    for i, key in enumerate(["lejepa", "invariance", "sigreg", "lr"]):
        plt.subplot(1, 4, i + 1)
        plt.plot(epochs, history[key])
        plt.title(key)
        plt.xlabel("Epoch")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "training_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(15, 8))
    diag_groups = [
        ("standard deviation", ["raw_proj_std", "loss_proj_std", "emb_std"]),
        ("mean vector norm", ["raw_proj_norm", "loss_proj_norm", "emb_norm"]),
        ("effective rank", ["raw_proj_effective_rank", "loss_proj_effective_rank", "emb_effective_rank"]),
    ]
    for row, (title, keys) in enumerate(diag_groups):
        ax = plt.subplot(3, 1, row + 1)
        for key in keys:
            if key in history and len(history[key]) == len(epochs):
                ax.plot(epochs, history[key], label=key)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "training_diagnostics.png", dpi=200)
    plt.savefig(dirs["plots"] / "projection_embedding_diagnostics.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 8))
    for i, key in enumerate(["lejepa", "invariance", "sigreg", "raw_proj_std", "loss_proj_std", "emb_std"]):
        ax = plt.subplot(2, 3, i + 1)
        if key in history:
            ax.plot(epochs, history[key])
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "collapse_diagnostics.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.get("epoch_time_sec", []))
    plt.title("epoch_time_sec")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "epoch_times.png", dpi=200)
    plt.close()

    timing_keys = [
        "data_wait_and_augmentation_time_sec",
        "h2d_transfer_time_sec",
        "forward_loss_time_sec",
        "backward_optimizer_time_sec",
        "metrics_bookkeeping_time_sec",
    ]
    plt.figure(figsize=(10, 5))
    for key in timing_keys:
        if key in history:
            plt.plot(epochs, history[key], label=key.replace("_time_sec", ""))
    plt.title("Epoch timing components")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "epoch_timing_components.png", dpi=200)
    plt.close()

    fraction_keys = [
        "data_wait_and_augmentation_fraction",
        "h2d_transfer_fraction",
        "forward_loss_fraction",
        "backward_optimizer_fraction",
        "metrics_bookkeeping_fraction",
    ]
    plt.figure(figsize=(10, 5))
    for key in fraction_keys:
        if key in history:
            plt.plot(epochs, history[key], label=key.replace("_fraction", ""))
    plt.title("Epoch timing fractions")
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of epoch wall time")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "epoch_timing_fractions.png", dpi=200)
    plt.close()


@torch.inference_mode()
def extract_features(loader: DataLoader, net: nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    feats, labels = [], []
    net.eval()
    for views, y in tqdm(loader, desc="Extract features"):
        views = views.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            emb, _ = net(views)
        feats.append(emb.float().cpu())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
    def forward(self, x):
        return self.net(x)


def balanced_subset_indices(labels: torch.Tensor, max_total: int, seed: int) -> torch.Tensor:
    if max_total <= 0 or len(labels) <= max_total:
        return torch.arange(len(labels))
    g = torch.Generator().manual_seed(seed)
    classes = sorted(labels.unique().tolist())
    per_class = max(1, max_total // len(classes))
    chunks = []
    for c in classes:
        idx = torch.where(labels == int(c))[0]
        idx = idx[torch.randperm(len(idx), generator=g)[: min(per_class, len(idx))]]
        chunks.append(idx)
    out = torch.cat(chunks)
    return out[torch.randperm(len(out), generator=g)]


def compute_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-12)


def eval_probe(probe: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, Any]:
    probe.eval()
    with torch.inference_mode():
        pred = probe(x.to(device)).argmax(dim=1).cpu().numpy()
    true = y.cpu().numpy()
    names = ["routine", "follow_up", "biopsy"]
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(true, pred, labels=[0, 1, 2], target_names=names, zero_division=0, output_dict=True),
    }


def train_collapsed_probe(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor,
                          test_x: torch.Tensor, test_y: torch.Tensor, cfg: TrainConfig, device: torch.device,
                          dirs: dict[str, Path]) -> dict[str, Any]:
    idx = balanced_subset_indices(train_y, cfg.probe_train_max_samples, cfg.seed)
    train_x, train_y = train_x[idx], train_y[idx]
    probe = LinearProbe(cfg.backbone_output_dim, 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_learning_rate, weight_decay=cfg.probe_weight_decay)
    weights = compute_class_weights(train_y, 3).to(device) if cfg.probe_use_class_weights else None
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.eval_batch_size, shuffle=True, drop_last=False)
    hist = {"train_loss": [], "train_acc": [], "val_balanced_accuracy": [], "val_macro_f1": []}

    for epoch in range(cfg.probe_epochs):
        probe.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += xb.size(0)
        val_metrics = eval_probe(probe, val_x, val_y, device)
        hist["train_loss"].append(total_loss / max(1, total))
        hist["train_acc"].append(correct / max(1, total))
        hist["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        hist["val_macro_f1"].append(val_metrics["macro_f1"])
        print(
            f"Probe epoch {epoch + 1:03d} | loss {hist['train_loss'][-1]:.4f} | "
            f"train_acc {hist['train_acc'][-1]:.4f} | val_bal_acc {hist['val_balanced_accuracy'][-1]:.4f}",
            flush=True,
        )

    val_metrics = eval_probe(probe, val_x, val_y, device)
    test_metrics = eval_probe(probe, test_x, test_y, device)
    result = {"history": hist, "val": val_metrics, "test": test_metrics, "class_names": ["routine", "follow_up", "biopsy"]}
    save_json(result, dirs["metrics"] / "linear_probe_collapsed_birads.json")
    plot_probe_history(hist, dirs)
    print("Collapsed BI-RADS linear probe test summary:")
    print(json.dumps({k: test_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2))
    return result


def plot_probe_history(hist: dict[str, list[float]], dirs: dict[str, Path]) -> None:
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(14, 4))
    for i, key in enumerate(["train_loss", "train_acc", "val_balanced_accuracy", "val_macro_f1"]):
        plt.subplot(1, 4, i + 1)
        plt.plot(epochs, hist[key])
        plt.title(key)
        plt.xlabel("Epoch")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "linear_probe_collapsed_birads.png", dpi=200)
    plt.close()



# -----------------------------
# PCA and multi-GPU helpers
# -----------------------------

def unwrap_model(net: nn.Module) -> nn.Module:
    return net.module if isinstance(net, (nn.DataParallel, DDP)) else net


def maybe_wrap_model(net: nn.Module, cfg: TrainConfig, device: torch.device, distributed: bool) -> nn.Module:
    if distributed:
        if cfg.use_sync_batchnorm:
            rank0_print("Converting projector/backbone BatchNorm layers to SyncBatchNorm.", flush=True)
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        rank0_print(f"Using torch.nn.parallel.DistributedDataParallel across {get_world_size()} GPUs. Global batch size={cfg.batch_size}.", flush=True)
        return DDP(net, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    if device.type != "cuda":
        return net
    n_gpu = torch.cuda.device_count()
    use_dp = cfg.multi_gpu == "data_parallel" or (cfg.multi_gpu == "auto" and n_gpu > 1)
    if use_dp and n_gpu > 1:
        print(f"Using torch.nn.DataParallel across {n_gpu} GPUs. Global batch size={cfg.batch_size}.", flush=True)
        return nn.DataParallel(net)
    print(f"Using single GPU. Visible CUDA devices: {n_gpu}.", flush=True)
    return net


def make_balanced_pca_subset(features: torch.Tensor, labels: torch.Tensor, max_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = balanced_subset_indices(labels, max_samples, seed)
    return features[idx].cpu(), labels[idx].cpu()


def plot_pca_latent_space(features: torch.Tensor, labels: torch.Tensor, dirs: dict[str, Path], max_samples: int, seed: int) -> dict[str, Any]:
    class_names = ["routine", "follow_up", "biopsy"]
    x, y = make_balanced_pca_subset(features, labels, max_samples, seed)
    x_np = x.numpy()
    y_np = y.numpy()

    pca2 = PCA(n_components=2, random_state=seed)
    z2 = pca2.fit_transform(x_np)
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(z2[:, 0], z2[:, 1], c=y_np, s=10, alpha=0.7, cmap="tab10")
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(class_names)
    plt.title(f"Backbone embeddings PCA 2D by collapsed BI-RADS\nExplained variance: {pca2.explained_variance_ratio_[0]*100:.2f}%, {pca2.explained_variance_ratio_[1]*100:.2f}%")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "latent_pca_2d_collapsed.png", dpi=200)
    plt.close()

    pca3 = PCA(n_components=3, random_state=seed)
    z3 = pca3.fit_transform(x_np)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(z3[:, 0], z3[:, 1], z3[:, 2], c=y_np, s=10, alpha=0.7, cmap="tab10")
    cbar = fig.colorbar(sc, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(class_names)
    ax.set_title("Backbone embeddings PCA 3D by collapsed BI-RADS\nExplained variance: " + ", ".join(f"{v*100:.2f}%" for v in pca3.explained_variance_ratio_))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "latent_pca_3d_collapsed.png", dpi=200)
    plt.close()

    info = {
        "num_samples": int(len(x)),
        "class_counts": {class_names[int(c)]: int((y == int(c)).sum().item()) for c in y.unique()},
        "pca2_explained_variance_ratio": [float(v) for v in pca2.explained_variance_ratio_],
        "pca3_explained_variance_ratio": [float(v) for v in pca3.explained_variance_ratio_],
    }
    save_json(info, dirs["metrics"] / "pca_summary.json")
    return info

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    script_start_time = time.perf_counter()
    args = parse_args()
    cfg = build_config(args)
    distributed, rank, world_size, local_rank, device = setup_distributed(cfg)
    set_seed(cfg.seed + rank)
    dirs = create_dirs(cfg.output_dir)

    with open(cfg.aug_config_path, "r", encoding="utf-8") as f:
        aug_cfg = json.load(f)

    # If not explicitly provided, let the augmentation config determine the final model/input size.
    if cfg.image_size <= 0:
        cfg.image_size = int(deep_get(aug_cfg, ["image", "output_size"], 224))
        rank0_print(f"Using image_size from augmentation config: {cfg.image_size}", flush=True)
    else:
        rank0_print(f"Using image_size from CLI: {cfg.image_size}", flush=True)

    if is_main_process():
        save_json(asdict(cfg), dirs["root"] / "config.json")
        save_json(aug_cfg, dirs["root"] / "augmentation_config_used.json")

    timing: dict[str, Any] = {}
    rank0_print("Using device:", device)
    if device.type == "cuda":
        rank0_print("Primary GPU:", torch.cuda.get_device_name(device.index or 0))
        rank0_print("Visible CUDA devices:", torch.cuda.device_count())
        rank0_print("Distributed:", distributed, "rank/world/local_rank:", rank, world_size, local_rank)
        free, total = torch.cuda.mem_get_info(device)
        rank0_print(f"CUDA memory free/total GiB: {free/1024**3:.2f}/{total/1024**3:.2f}")

    t0 = time.perf_counter()
    full_df, train_df, val_df, test_df = load_splits(cfg)
    expected_bytes = len(full_df) * cfg.image_height * cfg.image_width * np.dtype(cfg.memmap_dtype).itemsize
    actual_bytes = Path(cfg.bin_path).stat().st_size
    rank0_print(f"BIN check: expected={expected_bytes/1024**3:.3f} GiB actual={actual_bytes/1024**3:.3f} GiB match={expected_bytes == actual_bytes}")
    if expected_bytes != actual_bytes:
        raise RuntimeError("Full CSV and BIN do not match. Refusing to train.")
    timing["csv_split_loading_and_verification_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    train_transform = ConfigurableMGAugmentation(aug_cfg, cfg.image_size, train=True)
    eval_transform = ConfigurableMGAugmentation(aug_cfg, cfg.image_size, train=False)
    shape = (cfg.image_height, cfg.image_width)
    n_full = len(full_df)

    train_ds = MedJEPADataset(train_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, train_transform,
                              cfg.num_views, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)
    train_eval_ds = MedJEPADataset(train_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, eval_transform,
                                   1, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)
    val_ds = MedJEPADataset(val_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, eval_transform,
                            1, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)
    test_ds = MedJEPADataset(test_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, eval_transform,
                             1, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)

    if cfg.use_weighted_sampler and cfg.batch_construction != "random":
        raise RuntimeError("Use either --batch-weighting or --batch-construction, not both.")

    if distributed:
        if cfg.batch_size % world_size != 0:
            raise ValueError(f"Global --batch-size {cfg.batch_size} must be divisible by world_size {world_size} for DDP.")
        per_process_batch_size = cfg.batch_size // world_size
    else:
        per_process_batch_size = cfg.batch_size

    if cfg.batch_construction == "balanced_collapsed":
        train_loader = make_balanced_train_loader(train_ds, cfg, per_process_batch_size, world_size, rank)
        train_sampler = None
        rank0_print("Using balanced collapsed-BI-RADS physical batch construction.", flush=True)
    elif cfg.batch_construction == "random":
        if distributed and cfg.use_weighted_sampler:
            raise RuntimeError("--batch-weighting is not implemented for DDP. Use --batch-construction balanced_collapsed instead.")
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else (make_weighted_sampler(train_df) if cfg.use_weighted_sampler else None)
        train_loader = make_loader(train_ds, cfg, per_process_batch_size, shuffle=(not distributed), drop_last=True, sampler=train_sampler)
    else:
        raise ValueError(f"Unknown batch_construction: {cfg.batch_construction}")
    train_eval_loader = make_loader(train_eval_ds, cfg, cfg.eval_batch_size, shuffle=False, drop_last=False)
    val_loader = make_loader(val_ds, cfg, cfg.eval_batch_size, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, cfg, cfg.eval_batch_size, shuffle=False, drop_last=False)
    timing["dataset_and_dataloader_setup_sec"] = time.perf_counter() - t0

    rank0_print("Train batches per process:", len(train_loader))
    rank0_print("Global batch size:", cfg.batch_size)
    rank0_print("Per-process batch size:", per_process_batch_size)
    rank0_print("Gradient accumulation steps:", cfg.grad_accum_steps)
    rank0_print("Effective optimizer batch size:", cfg.batch_size * cfg.grad_accum_steps)
    rank0_print("Views per sample:", cfg.num_views)
    rank0_print("Effective augmented views per physical step:", cfg.batch_size * cfg.num_views)
    rank0_print("Effective augmented views per optimizer step:", cfg.batch_size * cfg.num_views * cfg.grad_accum_steps)

    t0 = time.perf_counter()
    net = ViTEncoder(cfg).to(device)
    net = maybe_wrap_model(net, cfg, device, distributed)
    loss_fn = LeJEPALoss(cfg).to(device)
    optimizer, scheduler = build_optimizer_scheduler(net, train_loader, cfg)

    start_epoch = 0
    resume_history: Optional[dict[str, list[float]]] = None
    if cfg.resume_checkpoint_path:
        ckpt_path = Path(cfg.resume_checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        unwrap_model(net).load_state_dict(payload["model_state_dict"])
        if cfg.resume_weights_only:
            start_epoch = 0
            resume_history = None
            rank0_print(
                f"Loaded model weights only from {ckpt_path}; starting fresh optimizer/scheduler/history for {cfg.epochs} epochs.",
                flush=True,
            )
        else:
            if "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            if "scheduler_state_dict" in payload:
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload.get("epoch", 0))
            resume_history = payload.get("history", None)
            rank0_print(f"Resumed checkpoint {ckpt_path} at completed epoch {start_epoch}.", flush=True)

    timing["model_optimizer_setup_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    history = train_lejepa(net, loss_fn, train_loader, optimizer, scheduler, cfg, device, dirs, aug_cfg,
                           start_epoch=start_epoch, existing_history=resume_history)
    timing["training_total_sec"] = time.perf_counter() - t0
    timing["training_epoch_time_mean_sec"] = float(np.mean(history.get("epoch_time_sec", [0.0])))
    timing["training_epoch_time_median_sec"] = float(np.median(history.get("epoch_time_sec", [0.0])))
    timing["training_epoch_time_min_sec"] = float(np.min(history.get("epoch_time_sec", [0.0])))
    timing["training_epoch_time_max_sec"] = float(np.max(history.get("epoch_time_sec", [0.0])))
    timing["timing_cuda_synchronize"] = bool(cfg.timing_cuda_synchronize)
    for key in [
        "data_wait_and_augmentation_time_sec",
        "h2d_transfer_time_sec",
        "forward_loss_time_sec",
        "backward_optimizer_time_sec",
        "metrics_bookkeeping_time_sec",
    ]:
        vals = history.get(key, [])
        if vals:
            timing[f"training_{key}_total"] = float(np.sum(vals))
            timing[f"training_{key}_mean_per_epoch"] = float(np.mean(vals))
            timing[f"training_{key}_mean_per_batch"] = float(np.sum(vals) / max(1, np.sum(history.get("num_batches", [1]))))
    if is_main_process():
        plot_training_history(history, dirs)

    # Save final resumable LeJEPA checkpoint.
    t0 = time.perf_counter()
    save_training_checkpoint(net, optimizer, scheduler, cfg.epochs, cfg, aug_cfg, history, dirs, final=True)
    timing["model_saving_sec"] = time.perf_counter() - t0

    probe_result: Optional[dict[str, Any]] = None
    pca_info: Optional[dict[str, Any]] = None

    if cfg.run_final_analysis:
        if distributed:
            rank0_print("WARNING: --run-final-analysis is skipped in DDP mode. Use the separate analysis scripts on the saved checkpoints.", flush=True)
        else:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()

            t0 = time.perf_counter()
            train_x, train_y = extract_features(train_eval_loader, net, device)
            val_x, val_y = extract_features(val_loader, net, device)
            test_x, test_y = extract_features(test_loader, net, device)
            timing["feature_extraction_total_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            probe_result = train_collapsed_probe(train_x, train_y, val_x, val_y, test_x, test_y, cfg, device, dirs)
            timing["linear_probe_total_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            pca_info = plot_pca_latent_space(test_x, test_y, dirs, cfg.pca_max_samples, cfg.seed)
            timing["pca_total_sec"] = time.perf_counter() - t0

    timing["total_script_wall_time_sec"] = time.perf_counter() - script_start_time

    summary = {
        "final_lejepa_loss": history["lejepa"][-1],
        "final_invariance_loss": history["invariance"][-1],
        "final_sigreg_loss": history["sigreg"][-1],
        "final_proj_std": history["proj_std"][-1],
        "final_raw_proj_std": history.get("raw_proj_std", history["proj_std"])[-1],
        "final_loss_proj_std": history.get("loss_proj_std", history["proj_std"])[-1],
        "final_emb_std": history.get("emb_std", [None])[-1],
        "linear_probe_collapsed_val_accuracy": None if probe_result is None else probe_result["val"]["accuracy"],
        "linear_probe_collapsed_val_balanced_accuracy": None if probe_result is None else probe_result["val"]["balanced_accuracy"],
        "linear_probe_collapsed_val_macro_f1": None if probe_result is None else probe_result["val"]["macro_f1"],
        "linear_probe_collapsed_test_accuracy": None if probe_result is None else probe_result["test"]["accuracy"],
        "linear_probe_collapsed_test_balanced_accuracy": None if probe_result is None else probe_result["test"]["balanced_accuracy"],
        "linear_probe_collapsed_test_macro_f1": None if probe_result is None else probe_result["test"]["macro_f1"],
        "pca": pca_info,
        "timing": timing,
        "multi_gpu": {
            "requested": cfg.multi_gpu,
            "visible_cuda_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "distributed": bool(distributed),
            "world_size": int(world_size),
            "used_data_parallel": isinstance(net, nn.DataParallel),
            "used_ddp": isinstance(net, DDP),
        },
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
        "num_test_rows": int(len(test_df)),
        "output_dir": str(dirs["root"]),
    }
    if is_main_process():
        save_json(summary, dirs["metrics"] / "summary.json")
        print("Run finished.")
        print(json.dumps(summary, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
