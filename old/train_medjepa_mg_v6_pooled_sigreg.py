#!/usr/bin/env python3
"""
MedJEPA LeJEPA training script v6.

Purpose:
- Train LeJEPA/SIGReg on pre-created patient-disjoint split CSVs.
- Read raw uint16 512x512 image memmap by original_index.
- Load augmentation policy from a JSON config.
- Save final trained encoder/checkpoint and training history.
- Run exactly one diagnostic linear probe on collapsed BI-RADS after training.

No split creation and no online probe. Adds DDP training, gradient accumulation, FP32 SSL loss outside autocast, 
reduced per-batch diagnostics, periodic checkpoints, and optional top-corner watermark masking after foreground crop. 
Final probe/PCA are optional because larger analysis is usually done by separate scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import platform
import random
import re
import socket
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

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
    """Internal flattened config used by the training code.

    v6 reads a compact JSON run config and maps it to this dataclass. The
    compact JSON intentionally exposes only experiment-level knobs; implementation
    defaults stay here.
    """
    # Run metadata
    schema_version: str = "medjepa_v6_config_0.1"
    run_name: str = "mg_v6_run"
    run_description: str = ""
    run_config_path: str = ""
    save_resolved_config: bool = True

    # Runtime
    seed: int = 42
    num_workers: int = 8
    output_dir: str = "outputs_medjepa_v6"

    # Data / paths
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
    normalize_mode: str = "uint16"
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

    # LeJEPA/SIGReg loss. v6 keeps the successful v5_modified behavior fixed:
    # invariance(normalized projection) and SIGReg(raw projection).
    lambda_sigreg: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_projections: int = 256
    projection_normalization: str = "sqrt_dim"

    # Optimization
    epochs: int = 300
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 5e-2
    eta_min: float = 1e-5
    warmup_epochs: int = 10
    grad_clip_norm: float = 0.0
    grad_accum_steps: int = 1

    # Batch construction / sampler. Disabled reproduces the successful v5 run.
    batch_construction_enabled: bool = False
    batch_construction_mode: str = "weighted_infinite"
    batch_iterations_per_epoch: int = 1650
    batch_balance_keys: tuple[str, ...] = ("collapsed_birads", "dataset", "machine_family", "view")
    batch_balance_strength: float = 0.5
    batch_dry_run_iterations: int = 10000

    # Positive real-image pairs. When enabled, some SSL views come from a paired
    # image, e.g. same patient/exam/laterality but different view.
    positive_pairs_enabled: bool = False
    positive_pairs_probability: float = 0.0
    positive_require_same_patient: bool = True
    positive_require_same_exam: bool = True
    positive_require_same_laterality: bool = True
    positive_require_different_view: bool = True

    # Optional supervised contrastive signal on embeddings. Disabled by default.
    supcon_enabled: bool = False
    supcon_target: str = "collapsed_birads"
    supcon_mode: str = "alternating"  # auxiliary or alternating
    supcon_weight: float = 1.0
    supcon_temperature: float = 0.1
    supcon_lejepa_epochs: int = 20
    supcon_contrastive_epochs: int = 1

    # Checkpointing and training diagnostics
    checkpoint_every_epochs: int = 50
    diagnostic_every_batches: int = 50
    resume_checkpoint_path: str = ""
    resume_weights_only: bool = False

    # Built-in final analysis stays off by default.
    run_final_analysis: bool = False
    probe_epochs: int = 50
    probe_learning_rate: float = 1e-3
    probe_weight_decay: float = 1e-7
    probe_train_max_samples: int = 60000
    probe_use_class_weights: bool = True
    pca_max_samples: int = 3000

    # Multi-GPU / timing
    multi_gpu: str = "ddp"
    use_sync_batchnorm: bool = False
    timing_cuda_synchronize: bool = True


def _cfg_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedJEPA LeJEPA v6 training from compact JSON run config.")
    p.add_argument("--run-config", required=True, type=str, help="Path to compact v6 JSON run configuration.")
    return p.parse_args()


def load_run_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Run config JSON must contain an object at the top level.")
    return cfg


def build_config_from_run_config(run_cfg: dict[str, Any], run_config_path: str) -> TrainConfig:
    cfg = TrainConfig()
    cfg.schema_version = str(run_cfg.get("schema_version", cfg.schema_version))
    cfg.run_config_path = str(run_config_path)

    run = run_cfg.get("run", {})
    paths = run_cfg.get("paths", {})
    model = run_cfg.get("model", {})
    training = run_cfg.get("training", {})
    optimizer = run_cfg.get("optimizer", {})
    loss = run_cfg.get("loss", {})
    batch_construction = run_cfg.get("batch_construction", {})
    positive_pairs = run_cfg.get("positive_pairs", {})
    supcon = run_cfg.get("supervised_contrastive", {})
    distributed = run_cfg.get("distributed", {})
    checkpointing = run_cfg.get("checkpointing", {})
    diagnostics = run_cfg.get("diagnostics", {})

    cfg.run_name = str(run.get("name", cfg.run_name))
    cfg.run_description = str(run.get("description", cfg.run_description))
    cfg.output_dir = str(run.get("output_dir", cfg.output_dir))
    cfg.seed = int(run.get("seed", cfg.seed))
    cfg.save_resolved_config = bool(run.get("save_resolved_config", cfg.save_resolved_config))

    cfg.full_csv_path = str(paths.get("full_csv", cfg.full_csv_path))
    cfg.bin_path = str(paths.get("bin", cfg.bin_path))
    cfg.train_csv_path = str(paths.get("train_csv", cfg.train_csv_path))
    cfg.val_csv_path = str(paths.get("val_csv", cfg.val_csv_path))
    cfg.test_csv_path = str(paths.get("test_csv", cfg.test_csv_path))
    cfg.aug_config_path = str(paths.get("augmentation_config", cfg.aug_config_path))

    cfg.backbone_name = str(model.get("backbone_name", cfg.backbone_name))
    cfg.image_size = int(model.get("image_size", cfg.image_size))
    cfg.backbone_output_dim = int(model.get("backbone_output_dim", cfg.backbone_output_dim))
    cfg.projection_dim = int(model.get("projection_dim", cfg.projection_dim))
    cfg.projector_hidden_dim = int(model.get("projector_hidden_dim", cfg.projector_hidden_dim))
    cfg.drop_path_rate = float(model.get("drop_path_rate", cfg.drop_path_rate))

    cfg.epochs = int(training.get("epochs", cfg.epochs))
    cfg.batch_size = int(training.get("batch_size", cfg.batch_size))
    cfg.num_views = int(training.get("num_views", cfg.num_views))
    cfg.num_workers = int(training.get("num_workers", cfg.num_workers))
    cfg.grad_accum_steps = max(1, int(training.get("grad_accum_steps", cfg.grad_accum_steps)))

    cfg.learning_rate = float(optimizer.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(optimizer.get("weight_decay", cfg.weight_decay))
    cfg.warmup_epochs = int(optimizer.get("warmup_epochs", cfg.warmup_epochs))

    cfg.lambda_sigreg = float(loss.get("lambda_sigreg", cfg.lambda_sigreg))
    cfg.sigreg_knots = int(loss.get("sigreg_knots", cfg.sigreg_knots))
    cfg.sigreg_num_projections = int(loss.get("sigreg_num_projections", cfg.sigreg_num_projections))
    cfg.projection_normalization = str(loss.get("projection_normalization", cfg.projection_normalization))
    if cfg.projection_normalization not in {"none", "unit", "sqrt_dim"}:
        raise ValueError(f"Unknown projection_normalization: {cfg.projection_normalization}")

    cfg.batch_construction_enabled = bool(batch_construction.get("enabled", cfg.batch_construction_enabled))
    cfg.batch_construction_mode = str(batch_construction.get("mode", cfg.batch_construction_mode))
    cfg.batch_iterations_per_epoch = int(batch_construction.get("iterations_per_epoch", cfg.batch_iterations_per_epoch))
    cfg.batch_balance_keys = tuple(str(x) for x in batch_construction.get("balance_keys", list(cfg.batch_balance_keys)))
    cfg.batch_balance_strength = float(batch_construction.get("balance_strength", cfg.batch_balance_strength))
    cfg.batch_dry_run_iterations = int(batch_construction.get("dry_run_iterations", cfg.batch_dry_run_iterations))

    cfg.positive_pairs_enabled = bool(positive_pairs.get("enabled", cfg.positive_pairs_enabled))
    cfg.positive_pairs_probability = float(positive_pairs.get("probability", cfg.positive_pairs_probability))
    cfg.positive_require_same_patient = bool(positive_pairs.get("require_same_patient", cfg.positive_require_same_patient))
    cfg.positive_require_same_exam = bool(positive_pairs.get("require_same_exam", cfg.positive_require_same_exam))
    cfg.positive_require_same_laterality = bool(positive_pairs.get("require_same_laterality", cfg.positive_require_same_laterality))
    cfg.positive_require_different_view = bool(positive_pairs.get("require_different_view", cfg.positive_require_different_view))

    cfg.supcon_enabled = bool(supcon.get("enabled", cfg.supcon_enabled))
    cfg.supcon_target = str(supcon.get("target", cfg.supcon_target))
    cfg.supcon_mode = str(supcon.get("mode", cfg.supcon_mode))
    cfg.supcon_weight = float(supcon.get("weight", cfg.supcon_weight))
    cfg.supcon_temperature = float(supcon.get("temperature", cfg.supcon_temperature))
    cfg.supcon_lejepa_epochs = int(supcon.get("lejepa_epochs", cfg.supcon_lejepa_epochs))
    cfg.supcon_contrastive_epochs = int(supcon.get("contrastive_epochs", cfg.supcon_contrastive_epochs))
    if cfg.supcon_mode not in {"auxiliary", "alternating"}:
        raise ValueError(f"Unknown supervised_contrastive.mode: {cfg.supcon_mode}")

    cfg.multi_gpu = str(distributed.get("multi_gpu", cfg.multi_gpu))
    cfg.use_sync_batchnorm = bool(distributed.get("sync_batchnorm", cfg.use_sync_batchnorm))
    cfg.timing_cuda_synchronize = not bool(distributed.get("no_cuda_timing_sync", not cfg.timing_cuda_synchronize))

    cfg.checkpoint_every_epochs = int(checkpointing.get("checkpoint_every_epochs", cfg.checkpoint_every_epochs))
    cfg.resume_checkpoint_path = str(checkpointing.get("resume_checkpoint", cfg.resume_checkpoint_path) or "")
    cfg.resume_weights_only = bool(checkpointing.get("resume_weights_only", cfg.resume_weights_only))

    cfg.diagnostic_every_batches = max(1, int(diagnostics.get("diagnostic_every_batches", cfg.diagnostic_every_batches)))

    required_paths = {
        "paths.full_csv": cfg.full_csv_path,
        "paths.bin": cfg.bin_path,
        "paths.train_csv": cfg.train_csv_path,
        "paths.val_csv": cfg.val_csv_path,
        "paths.test_csv": cfg.test_csv_path,
        "paths.augmentation_config": cfg.aug_config_path,
        "run.output_dir": cfg.output_dir,
    }
    missing = [k for k, v in required_paths.items() if not v]
    if missing:
        raise ValueError(f"Run config is missing required fields: {missing}")

    return cfg

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



def _short_obj(obj: Any, max_len: int = 500) -> str:
    s = str(obj)
    return s if len(s) <= max_len else s[:max_len] + "..."


def collect_runtime_metadata(cfg: TrainConfig, run_cfg: dict[str, Any], device: torch.device,
                             distributed: bool, rank: int, world_size: int, local_rank: int) -> dict[str, Any]:
    """Collect explicit experiment metadata for reproducibility."""
    env_keys = [
        "DET_MASTER", "DET_WORKSPACE", "SLURM_JOB_ID", "WORLD_SIZE", "RANK", "LOCAL_RANK",
        "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    ]
    meta = {
        "created_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "schema_version": cfg.schema_version,
        "run_name": cfg.run_name,
        "run_description": cfg.run_description,
        "run_config_path": cfg.run_config_path,
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": str(device),
        "primary_gpu": torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else None,
        "distributed": {
            "enabled": bool(distributed),
            "rank": int(rank),
            "world_size": int(world_size),
            "local_rank": int(local_rank),
            "multi_gpu": cfg.multi_gpu,
            "sync_batchnorm": bool(cfg.use_sync_batchnorm),
        },
        "environment": {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None},
        "compact_run_config": run_cfg,
        "resolved_train_config": asdict(cfg),
        "v6_feature_switches": {
            "batch_construction_enabled": bool(cfg.batch_construction_enabled),
            "positive_pairs_enabled": bool(cfg.positive_pairs_enabled),
            "supervised_contrastive_enabled": bool(cfg.supcon_enabled),
        },
    }
    return meta


def update_experiment_metadata(dirs: dict[str, Path], updates: dict[str, Any]) -> None:
    if not is_main_process():
        return
    path = dirs["root"] / "experiment_metadata.json"
    base: dict[str, Any] = {}
    if path.exists():
        try:
            base = json.loads(path.read_text())
        except Exception:
            base = {}
    base.update(updates)
    save_json(base, path)


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


def _metadata_deep_get(dct: dict[str, Any], keys: list[str], default: Any = "unknown") -> Any:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _normalize_view(x: Any) -> str:
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "unknown", "missing"}:
        return "unknown"
    if s in {"mlo", "mediolateral oblique", "medio-lateral oblique"} or "mediolateral" in s or "oblique" in s:
        return "MLO"
    if s in {"cc", "cranial caudal", "craniocaudal", "cranio-caudal"} or "cranial" in s or "caudal" in s:
        return "CC"
    return s


def _normalize_laterality(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"left", "l"}:
        return "left"
    if s in {"right", "r"}:
        return "right"
    if not s or s in {"nan", "none", "unknown", "missing"}:
        return "unknown"
    return s


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
    if "howtek" in lo or "lumysis" in lo or "dba" in lo:
        return "Howtek/Lumysis"
    if "fuji" in lo or "fujifilm" in lo:
        return "Fujifilm"
    if "siemens" in lo:
        return "Siemens"
    return "Other"


def add_derived_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable metadata columns used by v6 batch construction and pairing."""
    df = df.copy()
    if "context" in df.columns:
        parsed = df["context"].apply(_safe_json_loads)
        df["view"] = parsed.apply(lambda d: _normalize_view(_metadata_deep_get(d, ["exam", "view"], "unknown")))
        df["laterality"] = parsed.apply(lambda d: _normalize_laterality(_metadata_deep_get(d, ["exam", "laterality"], "unknown")))
    else:
        df["view"] = "unknown"
        df["laterality"] = "unknown"

    if "machine" in df.columns:
        df["machine_family"] = df["machine"].apply(_machine_family)
    else:
        df["machine_family"] = "unknown"

    # If an exam column is not present, use available alternatives. This is only
    # used for positive-pair grouping; if everything is missing, pairs simply do not form.
    if "exam" not in df.columns:
        if "source_id" in df.columns:
            df["exam"] = df["source_id"]
        elif "id" in df.columns:
            df["exam"] = df["id"]
        else:
            df["exam"] = "unknown"
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
    full_df = add_derived_metadata(prepare_labels(full_raw))
    if "original_index" not in full_df.columns:
        full_df = full_df.reset_index(drop=False).rename(columns={"index": "original_index"})

    train_df = add_derived_metadata(prepare_labels(ensure_original_index(read_csv_clean(cfg.train_csv_path), full_raw, "train")))
    val_df = add_derived_metadata(prepare_labels(ensure_original_index(read_csv_clean(cfg.val_csv_path), full_raw, "val")))
    test_df = add_derived_metadata(prepare_labels(ensure_original_index(read_csv_clean(cfg.test_csv_path), full_raw, "test")))

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
                 num_views: int, normalize_mode: str, percentile_low: float, percentile_high: float,
                 pair_candidates: Optional[list[list[int]]] = None,
                 positive_pairs_probability: float = 0.0):
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
        self.pair_candidates = pair_candidates
        self.positive_pairs_probability = float(positive_pairs_probability)
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

    def _choose_pair_index(self, idx: int) -> Optional[int]:
        if self.pair_candidates is None or self.positive_pairs_probability <= 0:
            return None
        if np.random.random() >= self.positive_pairs_probability:
            return None
        candidates = self.pair_candidates[idx]
        if not candidates:
            return None
        return int(candidates[np.random.randint(0, len(candidates))])

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pair_idx = self._choose_pair_index(idx)
        source_indices: list[int]
        if pair_idx is not None:
            # Alternate original and paired image. For num_views=4 this gives
            # two augmentations of the anchor and two augmentations of the real positive.
            source_indices = [idx if (v % 2 == 0) else pair_idx for v in range(self.num_views)]
        else:
            source_indices = [idx for _ in range(self.num_views)]

        views = []
        for src_idx in source_indices:
            src_row = self.df.iloc[int(src_idx)]
            x = self._load_tensor(int(src_row["original_index"]))
            views.append(self.transform(x.clone()))
        views_t = torch.stack(views)
        label = int(row["target_collapsed"])
        return views_t, label


def collate_batch(batch):
    views = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return views, labels


def make_weighted_sampler(df: pd.DataFrame, label_col: str = "target_collapsed") -> WeightedRandomSampler:
    counts = df[label_col].value_counts().to_dict()
    weights = df[label_col].map(lambda y: 1.0 / counts[int(y)]).astype(float).values
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)



def build_positive_pair_candidates(df: pd.DataFrame, cfg: TrainConfig) -> tuple[list[list[int]], dict[str, Any]]:
    """Build row-index candidates for same-exam real positives.

    The compact config defines the constraints. If metadata is missing or sparse,
    candidates are simply empty for those samples.
    """
    candidates: list[list[int]] = [[] for _ in range(len(df))]
    if not cfg.positive_pairs_enabled:
        return candidates, {"enabled": False, "eligible_anchor_rows": 0, "total_candidate_links": 0}

    required = []
    if cfg.positive_require_same_patient:
        required.append("patient")
    if cfg.positive_require_same_exam:
        required.append("exam")
    if cfg.positive_require_same_laterality:
        required.append("laterality")
    if cfg.positive_require_different_view:
        required.append("view")
    missing = [c for c in required if c not in df.columns]
    if missing:
        return candidates, {"enabled": True, "error": f"missing required columns: {missing}"}

    group_keys = []
    if cfg.positive_require_same_patient:
        group_keys.append("patient")
    if cfg.positive_require_same_exam:
        group_keys.append("exam")
    if cfg.positive_require_same_laterality:
        group_keys.append("laterality")
    if not group_keys:
        group_keys = ["patient"] if "patient" in df.columns else []

    if not group_keys:
        return candidates, {"enabled": True, "eligible_anchor_rows": 0, "total_candidate_links": 0, "reason": "no grouping keys"}

    for _, group in df.reset_index().groupby(group_keys, dropna=False):
        row_indices = group["index"].astype(int).tolist()
        if len(row_indices) < 2:
            continue
        for anchor_idx in row_indices:
            anchor_view = str(df.iloc[anchor_idx].get("view", "unknown"))
            out = []
            for other_idx in row_indices:
                if other_idx == anchor_idx:
                    continue
                if cfg.positive_require_different_view:
                    other_view = str(df.iloc[other_idx].get("view", "unknown"))
                    if other_view == anchor_view or "unknown" in {other_view, anchor_view}:
                        continue
                out.append(int(other_idx))
            candidates[anchor_idx] = out

    eligible = sum(1 for x in candidates if x)
    links = sum(len(x) for x in candidates)
    views = df["view"].value_counts().to_dict() if "view" in df.columns else {}
    stats = {
        "enabled": True,
        "probability": float(cfg.positive_pairs_probability),
        "constraints": {
            "same_patient": bool(cfg.positive_require_same_patient),
            "same_exam": bool(cfg.positive_require_same_exam),
            "same_laterality": bool(cfg.positive_require_same_laterality),
            "different_view": bool(cfg.positive_require_different_view),
        },
        "eligible_anchor_rows": int(eligible),
        "eligible_anchor_fraction": float(eligible / max(1, len(df))),
        "total_candidate_links": int(links),
        "mean_candidates_per_eligible_anchor": float(links / max(1, eligible)),
        "train_view_counts": {str(k): int(v) for k, v in views.items()},
    }
    return candidates, stats


def compute_metadata_balance_weights(df: pd.DataFrame, keys: Sequence[str], strength: float,
                                     max_weight: float = 8.0, mix_uniform: float = 0.15) -> np.ndarray:
    if not keys:
        return np.ones(len(df), dtype=np.float64)
    weights = np.ones(len(df), dtype=np.float64)
    for key in keys:
        if key not in df.columns:
            rank0_print(f"WARNING: batch_construction balance key '{key}' missing; ignoring it.", flush=True)
            continue
        values = df[key].astype(str).fillna("unknown")
        counts = values.value_counts().to_dict()
        key_weights = values.map(lambda x: 1.0 / (float(counts.get(str(x), 1)) ** strength)).to_numpy(dtype=np.float64)
        key_weights = key_weights / max(float(key_weights.mean()), 1e-12)
        weights *= key_weights
    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.minimum(weights, max_weight)
    weights = (1.0 - mix_uniform) * weights + mix_uniform
    weights = weights / max(float(weights.mean()), 1e-12)
    return np.minimum(weights, max_weight)


class DistributedWeightedInfiniteBatchSampler(Sampler[list[int]]):
    """Metadata-weighted fixed-iteration sampler for DDP or single GPU.

    Each rank samples its local batch with replacement from the same weighted
    distribution. This intentionally defines an epoch by iterations_per_epoch,
    not by a full pass over the natural dataset.
    """
    def __init__(self, weights: np.ndarray, per_process_batch_size: int, iterations_per_epoch: int,
                 rank: int = 0, seed: int = 42):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("weights must be a non-empty 1D array")
        self.weights = np.maximum(self.weights, 0.0)
        if self.weights.sum() <= 0:
            raise ValueError("weights sum to zero")
        self.prob = self.weights / self.weights.sum()
        self.per_process_batch_size = int(per_process_batch_size)
        self.iterations_per_epoch = int(iterations_per_epoch)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.iterations_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch + 9176 * self.rank)
        for _ in range(self.iterations_per_epoch):
            yield rng.choice(len(self.prob), size=self.per_process_batch_size, replace=True, p=self.prob).tolist()


def make_weighted_infinite_train_loader(dataset: MedJEPADataset, cfg: TrainConfig,
                                        per_process_batch_size: int, rank: int) -> tuple[DataLoader, np.ndarray]:
    weights = compute_metadata_balance_weights(dataset.df, cfg.batch_balance_keys, cfg.batch_balance_strength)
    batch_sampler = DistributedWeightedInfiniteBatchSampler(
        weights=weights,
        per_process_batch_size=per_process_batch_size,
        iterations_per_epoch=cfg.batch_iterations_per_epoch,
        rank=rank,
        seed=cfg.seed,
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
    return DataLoader(**kwargs), weights


def preview_sampler_distribution(df: pd.DataFrame, cfg: TrainConfig, weights: Optional[np.ndarray] = None) -> dict[str, Any]:
    keys = list(cfg.batch_balance_keys)
    out: dict[str, Any] = {
        "enabled": bool(cfg.batch_construction_enabled),
        "mode": cfg.batch_construction_mode,
        "iterations": int(cfg.batch_dry_run_iterations),
        "balance_keys": keys,
        "balance_strength": float(cfg.batch_balance_strength),
        "natural_distribution": {},
        "sampled_distribution": {},
    }
    for key in keys:
        if key in df.columns:
            out["natural_distribution"][key] = {str(k): int(v) for k, v in df[key].astype(str).value_counts().to_dict().items()}
    if not cfg.batch_construction_enabled:
        return out
    if weights is None:
        weights = compute_metadata_balance_weights(df, cfg.batch_balance_keys, cfg.batch_balance_strength)
    prob = weights / weights.sum()
    rng = np.random.default_rng(cfg.seed + 12345)
    n = int(cfg.batch_dry_run_iterations) * int(cfg.batch_size)
    sampled = rng.choice(len(df), size=n, replace=True, p=prob)
    sdf = df.iloc[sampled]
    for key in keys:
        if key in sdf.columns:
            out["sampled_distribution"][key] = {str(k): int(v) for k, v in sdf[key].astype(str).value_counts().to_dict().items()}
    out["weight_summary"] = {
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
    }
    return out


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

    # Batch-first projection shape is required for correct torch.nn.DataParallel gathering.
    # Shape: [B, V, D]. Older versions returned [V, B, D], which breaks when
    # DataParallel sees unequal per-GPU batch sizes and also gathers along the wrong axis.
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, v = x.shape[:2]
        flat = x.flatten(0, 1)
        emb = self.backbone(flat)
        proj = self.proj(emb).reshape(b, v, -1)
        return emb, proj


def invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    # proj: [B, V, D]. Make all views of the same image agree.
    return (proj.mean(dim=1, keepdim=True) - proj).square().mean()



def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over all views of a batch.

    features: [B, V, D] or [N, D]. Labels are per original sample [B].
    All views with the same label are positives, excluding self.
    """
    if features.ndim == 3:
        b, v, d = features.shape
        z = features.reshape(b * v, d)
        y = labels.reshape(-1).repeat_interleave(v)
    elif features.ndim == 2:
        z = features
        y = labels.reshape(-1)
    else:
        raise ValueError("features must be [B,V,D] or [N,D]")
    if z.size(0) <= 1:
        return z.new_tensor(0.0)
    z = F.normalize(z.float(), dim=-1, eps=1e-6)
    logits = (z @ z.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    pos_mask = (y[:, None] == y[None, :]) & (~self_mask)
    logits_masked = logits.masked_fill(self_mask, -1e9)
    log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not bool(valid.any()):
        return z.new_tensor(0.0)
    loss = -(log_prob * pos_mask.float()).sum(dim=1)[valid] / pos_counts[valid].float()
    return loss.mean()


def contrastive_phase_for_epoch(epoch_zero_based: int, cfg: TrainConfig) -> str:
    if not cfg.supcon_enabled:
        return "lejepa"
    if cfg.supcon_mode == "auxiliary":
        return "lejepa_auxiliary_supcon"
    cycle = max(1, int(cfg.supcon_lejepa_epochs) + int(cfg.supcon_contrastive_epochs))
    pos = epoch_zero_based % cycle
    return "contrastive" if pos >= int(cfg.supcon_lejepa_epochs) else "lejepa"



class SIGReg(nn.Module):
    """Stable pooled-view SIGReg.

    Input convention in this script:
        proj: [B, V, D]

    This version intentionally pools all projected views into [B*V, D] before
    applying SIGReg. This is the empirically stable v5/v6 behavior used before
    the per-view SIGReg experiment.
    """
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
        if proj.ndim != 3:
            raise ValueError(f"Expected proj with shape [B, V, D], got {tuple(proj.shape)}")

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
        "optimization_loss": [], "lejepa": [], "invariance": [], "sigreg": [], "supcon": [], "phase": [],
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

        metric_sums = torch.zeros(14, device=device, dtype=torch.float64)
        # [optimization_loss, lejepa, invariance, sigreg, supcon,
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
                views, labels = next(loader_iter)
            except StopIteration:
                break

            fetch_end = time.perf_counter()
            timing_sums["data_wait_and_augmentation_time_sec"] += fetch_end - fetch_start

            t0 = time.perf_counter()
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
                lejepa_loss, inv, sig = loss_fn(proj.float())
                supcon = proj.new_tensor(0.0, dtype=torch.float32)
                phase = contrastive_phase_for_epoch(epoch, cfg)
                if cfg.supcon_enabled:
                    emb_views = emb.reshape(views.size(0), views.size(1), -1).float()
                    supcon = supervised_contrastive_loss(emb_views, labels, cfg.supcon_temperature)
                    if cfg.supcon_mode == "auxiliary":
                        loss = lejepa_loss + cfg.supcon_weight * supcon
                    elif phase == "contrastive":
                        loss = cfg.supcon_weight * supcon
                    else:
                        loss = lejepa_loss
                else:
                    loss = lejepa_loss
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
                metric_sums[1] += lejepa_loss.detach().double()
                metric_sums[2] += inv.detach().double()
                metric_sums[3] += sig.detach().double()
                metric_sums[4] += supcon.detach().double()
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

                    metric_sums[5] += raw_std.double()
                    metric_sums[6] += raw_norm.double()
                    metric_sums[7] += raw_erank.double()
                    metric_sums[8] += loss_std.double()
                    metric_sums[9] += loss_norm.double()
                    metric_sums[10] += loss_erank.double()
                    metric_sums[11] += emb_std.double()
                    metric_sums[12] += emb_norm.double()
                    metric_sums[13] += emb_erank.double()
                    count_tensor[1] += 1.0
            timing_sums["metrics_bookkeeping_time_sec"] += time.perf_counter() - t0

            fetch_start = time.perf_counter()

        reduce_sum_tensor(metric_sums)
        reduce_sum_tensor(count_tensor)

        global_nb = max(1.0, float(count_tensor[0].item()))
        global_diag_nb = max(1.0, float(count_tensor[1].item()))
        global_opt_steps = int(count_tensor[2].item())

        history["optimization_loss"].append(float(metric_sums[0].item() / global_nb))
        history["lejepa"].append(float(metric_sums[1].item() / global_nb))
        history["invariance"].append(float(metric_sums[2].item() / global_nb))
        history["sigreg"].append(float(metric_sums[3].item() / global_nb))
        history["supcon"].append(float(metric_sums[4].item() / global_nb))
        history["phase"].append(contrastive_phase_for_epoch(epoch, cfg))
        raw_proj_std = float(metric_sums[5].item() / global_diag_nb)
        raw_proj_norm = float(metric_sums[6].item() / global_diag_nb)
        raw_proj_erank = float(metric_sums[7].item() / global_diag_nb)
        loss_proj_std = float(metric_sums[8].item() / global_diag_nb)
        loss_proj_norm = float(metric_sums[9].item() / global_diag_nb)
        loss_proj_erank = float(metric_sums[10].item() / global_diag_nb)
        emb_std = float(metric_sums[11].item() / global_diag_nb)
        emb_norm = float(metric_sums[12].item() / global_diag_nb)
        emb_erank = float(metric_sums[13].item() / global_diag_nb)

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
                f"Epoch {epoch + 1:03d} | Phase {history['phase'][-1]} | OptLoss {history['optimization_loss'][-1]:.4f} | "
                f"LeJEPA {history['lejepa'][-1]:.4f} | Inv {history['invariance'][-1]:.4f} | "
                f"SIGReg {history['sigreg'][-1]:.4f} | SupCon {history['supcon'][-1]:.4f} | "
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
        if (
            cfg.checkpoint_every_epochs
            and cfg.checkpoint_every_epochs > 0
            and epoch_num % cfg.checkpoint_every_epochs == 0
            and epoch_num != cfg.epochs
        ):
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
    run_cfg = load_run_config(args.run_config)
    cfg = build_config_from_run_config(run_cfg, args.run_config)
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
        if cfg.save_resolved_config:
            save_json(asdict(cfg), dirs["root"] / "resolved_config.json")
            save_json(run_cfg, dirs["root"] / "run_config_original.json")
        save_json(aug_cfg, dirs["root"] / "augmentation_config_used.json")
        save_json(
            collect_runtime_metadata(cfg, run_cfg, device, distributed, rank, world_size, local_rank),
            dirs["root"] / "experiment_metadata.json",
        )

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
    if is_main_process():
        split_metadata = {}
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_metadata[name] = {
                "rows": int(len(df)),
                "patients": int(df["patient"].nunique()) if "patient" in df.columns else None,
                "collapsed_birads": {str(k): int(v) for k, v in df["collapsed_birads"].value_counts().to_dict().items()},
                "dataset": {str(k): int(v) for k, v in df["dataset"].value_counts().to_dict().items()} if "dataset" in df.columns else {},
                "machine_family": {str(k): int(v) for k, v in df["machine_family"].value_counts().to_dict().items()} if "machine_family" in df.columns else {},
                "view": {str(k): int(v) for k, v in df["view"].value_counts().to_dict().items()} if "view" in df.columns else {},
            }
        save_json(split_metadata, dirs["metrics"] / "split_metadata.json")
        update_experiment_metadata(dirs, {"split_metadata": split_metadata})
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

    pair_candidates, pair_stats = build_positive_pair_candidates(train_df, cfg)
    if is_main_process():
        save_json(pair_stats, dirs["metrics"] / "positive_pair_stats.json")
        update_experiment_metadata(dirs, {"positive_pair_stats": pair_stats})

    train_ds = MedJEPADataset(
        train_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, train_transform,
        cfg.num_views, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high,
        pair_candidates=pair_candidates if cfg.positive_pairs_enabled else None,
        positive_pairs_probability=cfg.positive_pairs_probability if cfg.positive_pairs_enabled else 0.0,
    )
    train_eval_ds = MedJEPADataset(train_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, eval_transform,
                                   1, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)
    val_ds = MedJEPADataset(val_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, eval_transform,
                            1, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)
    test_ds = MedJEPADataset(test_df, cfg.bin_path, n_full, shape, cfg.memmap_dtype, eval_transform,
                             1, cfg.normalize_mode, cfg.percentile_low, cfg.percentile_high)

    if distributed:
        if cfg.batch_size % world_size != 0:
            raise ValueError(f"Global batch_size {cfg.batch_size} must be divisible by world_size {world_size} for DDP.")
        per_process_batch_size = cfg.batch_size // world_size
    else:
        per_process_batch_size = cfg.batch_size

    sampler_weights: Optional[np.ndarray] = None
    if cfg.batch_construction_enabled:
        if cfg.batch_construction_mode != "weighted_infinite":
            raise ValueError(f"Unsupported batch_construction.mode: {cfg.batch_construction_mode}")
        train_loader, sampler_weights = make_weighted_infinite_train_loader(train_ds, cfg, per_process_batch_size, rank)
        rank0_print(
            "Using v6 weighted_infinite batch construction with fixed iterations_per_epoch="
            f"{cfg.batch_iterations_per_epoch}, keys={list(cfg.batch_balance_keys)}, strength={cfg.batch_balance_strength}.",
            flush=True,
        )
    else:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
        train_loader = make_loader(train_ds, cfg, per_process_batch_size, shuffle=(not distributed), drop_last=True, sampler=train_sampler)
        rank0_print("Using natural random batch construction.", flush=True)

    sampler_preview = preview_sampler_distribution(train_df, cfg, sampler_weights)
    if is_main_process():
        save_json(sampler_preview, dirs["metrics"] / "batch_construction_preview.json")
        update_experiment_metadata(dirs, {"batch_construction_preview": sampler_preview})
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
        update_experiment_metadata(dirs, {"final_summary": summary})
        print("Run finished.")
        print(json.dumps(summary, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
