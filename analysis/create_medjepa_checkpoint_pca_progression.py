#!/usr/bin/env python3
"""
Create a multi-page PDF visualizing MedJEPA / LeJEPA embedding-space progression across checkpoints.

The script scans a models/ directory for checkpoint_epoch_*.pt files, extracts deterministic
backbone embeddings for the same sampled MG images at every checkpoint, fits one shared PCA
coordinate system across all checkpoint embeddings, and plots the checkpoints side-by-side.

PDF pages included by default:
- Shared/global PCA pages, comparable across checkpoints:
  - 2D PCA progression colored by collapsed BI-RADS
  - 2D PCA progression colored by machine family
  - 2D PCA progression colored by view
  - 3D PCA progression colored by collapsed BI-RADS
- Local PCA pages, PCA fitted independently per checkpoint:
  - 2D PCA colored by collapsed BI-RADS
  - 2D PCA colored by machine family
  - 2D PCA colored by view
  - 3D PCA colored by collapsed BI-RADS

This script is compatible with the v5/v5_modified training checkpoints that use:
    timm vit_small_patch8_224, num_classes=512, in_chans=1
"""

from __future__ import annotations
from core.config import deep_get
from core.models import ViTEncoder, model_hints
from core.data import (
    read_csv_clean,
    prepare_labels,
    EvalDataset as MedJEPAPCADataset,
    collate_batch,
    validate_bin,
    ensure_original_index as validate_original_index,
)
from core.transforms import ConfigurableMGAugmentation
from core.features import extract_embedding_batches
from core.pca import make_label_series


import argparse
import hashlib
import json
import math
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# -----------------------------
# Config / model compatibility
# -----------------------------


@dataclass
class ModelConfig:
    image_size: int = 384
    backbone_name: str = "vit_small_patch8_224"
    backbone_output_dim: int = 512
    projection_dim: int = 16
    projector_hidden_dim: int = 2048
    drop_path_rate: float = 0.1
    backbone_num_classes: int = 512


def strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Common wrappers: DDP saves module.*, some helper scripts may save encoder.*.
    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "encoder.")
    return state_dict


def load_checkpoint_payload(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict_obj = None
        for key in ["model_state_dict", "state_dict", "encoder_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict_obj = checkpoint[key]
                break
        if state_dict_obj is None and all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict_obj = checkpoint
        if state_dict_obj is None:
            raise ValueError(f"Could not find a state dict in checkpoint: {path}")
        cfg_dict = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
        aug_cfg = (
            checkpoint.get("augmentation_config", {})
            if isinstance(checkpoint.get("augmentation_config", {}), dict)
            else {}
        )
        return normalize_state_dict_keys(state_dict_obj), cfg_dict, aug_cfg
    raise ValueError(f"Unsupported checkpoint format: {path}")


# -----------------------------
# CSV / labels / metadata
# -----------------------------


def _canonical_key_value(x: Any) -> str:
    """String-normalize a CSV cell for robust row matching."""
    if pd.isna(x):
        return "<NA>"
    # Keep JSON/text fields as strings, but avoid differences from surrounding whitespace.
    return str(x).strip()


def _make_row_keys(df: pd.DataFrame, key_cols: list[str]) -> list[tuple[str, ...]]:
    if not key_cols:
        raise ValueError("No key columns available for row matching.")
    arr = df[key_cols].copy()
    for c in key_cols:
        arr[c] = arr[c].map(_canonical_key_value)
    return list(map(tuple, arr.to_numpy(dtype=object)))


def _choose_matching_columns(split_df: pd.DataFrame, full_df_raw: pd.DataFrame) -> list[str]:
    """Choose columns likely to identify the original full-CSV row.

    The MG CSV can contain duplicated `id` values, so matching only by id is unsafe.
    Split CSVs created without `original_index` are usually literal subsets of the
    full CSV, so a composite key over the original metadata columns can recover the
    memmap row index.
    """
    excluded = {
        "original_index",
        "orig_index",
        "row_index",
        "memmap_index",
        "index",
        "Unnamed: 0",
        "split",
        "target",
        "target_collapsed",
        "machine_family",
        "view",
        "view_laterality",
        "laterality",  # derived in this script for some datasets
    }
    common = [c for c in split_df.columns if c in full_df_raw.columns and c not in excluded]

    # Prefer stable identifying/original columns first, then append remaining common columns.
    preferred = [
        "id",
        "patient",
        "dataset",
        "modality",
        "machine",
        "exam",
        "birads",
        "original_birads",
        "birads_numeric",
        "collapsed_birads",
        "race",
        "segmentation",
        "context",
        "findings",
    ]
    ordered: list[str] = []
    for c in preferred:
        if c in common and c not in ordered:
            ordered.append(c)
    for c in common:
        if c not in ordered:
            ordered.append(c)
    return ordered


def _map_split_rows_by_composite_key(split_df: pd.DataFrame, full_df_raw: pd.DataFrame, split_name: str) -> pd.Series:
    key_cols = _choose_matching_columns(split_df, full_df_raw)
    if not key_cols:
        raise ValueError(
            f"{split_name} split has no usable columns in common with full CSV for memmap lookup. "
            "Please add original_index to the split CSV."
        )

    full_keys = _make_row_keys(full_df_raw, key_cols)
    split_keys = _make_row_keys(split_df, key_cols)

    # First try a direct unique composite mapping.
    full_key_series = pd.Series(full_keys)
    if not full_key_series.duplicated().any():
        key_to_idx = {k: i for i, k in enumerate(full_keys)}
        mapped = pd.Series([key_to_idx.get(k, np.nan) for k in split_keys], index=split_df.index)
    else:
        # Fall back to occurrence-aware matching. Remaining duplicates here are rows that are
        # indistinguishable from the CSV alone. Popping in full-CSV order is the safest available
        # deterministic behavior, and for truly identical duplicate rows no CSV-only method can
        # recover a more precise index.
        from collections import defaultdict, deque

        buckets: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
        for i, k in enumerate(full_keys):
            buckets[k].append(i)
        out: list[float] = []
        ambiguous_keys = 0
        for k in split_keys:
            q = buckets.get(k)
            if not q:
                out.append(np.nan)
                continue
            if len(q) > 1:
                ambiguous_keys += 1
            out.append(float(q.popleft()))
        mapped = pd.Series(out, index=split_df.index)
        if ambiguous_keys:
            print(
                f"[warning] {split_name}: {ambiguous_keys} rows matched a composite key that appears "
                "multiple times in the full CSV. Assigned duplicate matches in full-CSV order. "
                "For exact reproducibility, consider adding original_index to the split CSVs."
            )

    missing = int(mapped.isna().sum())
    if missing:
        example_cols = ", ".join(key_cols[:8]) + (", ..." if len(key_cols) > 8 else "")
        raise ValueError(
            f"Could not map {missing} rows in {split_name} split back to full CSV. "
            f"Tried composite key columns: {example_cols}. "
            "Please add original_index to the split CSV or provide split files generated from the same full CSV."
        )

    print(
        f"Mapped {split_name} split to memmap rows using composite key: {', '.join(key_cols[:10])}"
        f"{'...' if len(key_cols) > 10 else ''}"
    )
    return mapped.astype(int)


def ensure_original_index(split_df: pd.DataFrame, full_df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_df = split_df.copy()
    if "original_index" in split_df.columns:
        return validate_original_index(split_df, full_df_raw, split_name)
    for candidate in ["orig_index", "row_index", "memmap_index", "Unnamed: 0", "index"]:
        if candidate in split_df.columns:
            vals = pd.to_numeric(split_df[candidate], errors="coerce")
            if vals.notna().all() and vals.min() >= 0 and vals.max() < len(full_df_raw):
                split_df["original_index"] = vals
                return validate_original_index(split_df, full_df_raw, split_name)

    if (
        "id" in split_df.columns
        and "id" in full_df_raw.columns
        and not full_df_raw["id"].astype(str).duplicated().any()
    ):
        id_to_idx = pd.Series(np.arange(len(full_df_raw)), index=full_df_raw["id"].astype(str)).to_dict()
        split_df["original_index"] = split_df["id"].astype(str).map(id_to_idx)
        missing = int(split_df["original_index"].isna().sum())
        if missing:
            raise ValueError(f"Could not map {missing} rows in {split_name} split by id.")
        return validate_original_index(split_df, full_df_raw, split_name)

    # Full CSV has duplicated IDs or ID is unavailable: recover indices from a richer composite key.
    split_df["original_index"] = _map_split_rows_by_composite_key(split_df, full_df_raw, split_name)
    return validate_original_index(split_df, full_df_raw, split_name)


def build_eval_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    full_raw = read_csv_clean(args.full_csv)

    split_paths = {"train": args.train_csv, "val": args.val_csv, "test": args.test_csv}
    selected = ["train", "val", "test"] if args.split == "all" else [args.split]
    frames: list[pd.DataFrame] = []
    for split_name in selected:
        path = split_paths.get(split_name, "")
        if not path:
            raise ValueError(f"--split {args.split!r} requires --{split_name}-csv")
        split_df = ensure_original_index(read_csv_clean(path), full_raw, split_name)
        split_df = prepare_labels(split_df)
        split_df["split"] = split_name
        frames.append(split_df)
    out = pd.concat(frames, ignore_index=True)
    return out, len(full_raw)


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


def _normalize_view(x: Any) -> str:
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "unknown", "missing", "null"}:
        return "unknown"
    if s in {"mlo", "mediolateral oblique", "medio-lateral oblique"} or "mediolateral" in s or "oblique" in s:
        return "MLO"
    if s in {"cc", "cranial caudal", "craniocaudal", "cranio-caudal"} or "cranial" in s or "caudal" in s:
        return "CC"
    su = s.upper()
    # Common compact mammography view strings.
    for pat, name in [
        (r"\bMLO\b", "MLO"),
        (r"\bCC\b", "CC"),
        (r"\bLMO\b", "LMO"),
        (r"\bLM\b", "LM"),
        (r"\bML\b", "ML"),
        (r"\bXCCL\b", "XCCL"),
        (r"\bXCCM\b", "XCCM"),
    ]:
        if re.search(pat, su):
            return name
    return s


def _extract_context_value(context_obj: dict[str, Any], keys: list[list[str]], default: str = "unknown") -> Any:
    for key_path in keys:
        val = deep_get(context_obj, key_path, None)
        if val not in [None, "", "unknown", "missing"]:
            return val
    return default


def _machine_family(x: Any) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    lo = s.lower()
    if not s or lo in {"nan", "none", "unknown", "missing", "null"}:
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
    if "philips" in lo:
        return "Philips"
    if "planmed" in lo:
        return "Planmed"
    return "Other"


def add_derived_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # view: prefer a real column; otherwise parse the JSON context if present; otherwise fallback to text fields.
    view_col = next((c for c in ["view", "image_view", "View", "mammography_view"] if c in df.columns), None)
    if view_col is not None:
        df["view"] = df[view_col].apply(_normalize_view)
    elif "context" in df.columns:
        parsed = df["context"].apply(_safe_json_loads)
        df["view"] = parsed.apply(lambda d: _normalize_view(_extract_context_value(d, [["exam", "view"], ["view"]])))
    else:
        df["view"] = "unknown"

    # If many views are still unknown, try a conservative regex fallback from other text columns.
    unknown_mask = df["view"].eq("unknown")
    if unknown_mask.any():
        text_cols = [c for c in ["exam", "findings", "id", "path", "filename"] if c in df.columns]
        if text_cols:
            joined = df.loc[unknown_mask, text_cols].fillna("").astype(str).agg(" ".join, axis=1)
            df.loc[unknown_mask, "view"] = joined.apply(_normalize_view)

    if "machine_family" not in df.columns:
        if "machine" in df.columns:
            df["machine_family"] = df["machine"].apply(_machine_family)
        else:
            df["machine_family"] = "unknown"
    else:
        df["machine_family"] = df["machine_family"].fillna("unknown").astype(str)

    if "birads_numeric" in df.columns:
        df["birads_numeric"] = pd.to_numeric(df["birads_numeric"], errors="coerce").fillna(-1).astype(int).astype(str)
    return df


# -----------------------------
# Deterministic eval transform / dataset
# -----------------------------


class EvalMammographyTransform(ConfigurableMGAugmentation):
    """Historical progression masking policy; crop/resize are shared."""

    def __init__(self, aug_cfg, image_size):
        super().__init__(aug_cfg, image_size, train=False)

    def _mask_top_corner(self, x: torch.Tensor) -> torch.Tensor:
        # Mirrors the optional watermark/top-corner masking used by v4/v5 configs when enabled.
        c = deep_get(self.cfg, ["preprocessing", "top_corner_mask"], {})
        if not c.get("enabled", False):
            c = deep_get(self.cfg, ["preprocessing", "mask_top_corner"], {})
        if not c.get("enabled", False):
            return x
        frac_h = float(c.get("height_frac", c.get("h_frac", 0.12)))
        frac_w = float(c.get("width_frac", c.get("w_frac", 0.25)))
        value = float(c.get("value", 0.0))
        _, h, w = x.shape
        mh = max(1, int(round(h * frac_h)))
        mw = max(1, int(round(w * frac_w)))
        side = str(c.get("side", "both")).lower()
        x = x.clone()
        if side in {"left", "both"}:
            x[:, :mh, :mw] = value
        if side in {"right", "both"}:
            x[:, :mh, w - mw :] = value
        return x


# -----------------------------
# Checkpoint discovery / sampling / feature extraction
# -----------------------------


def checkpoint_epoch(path: Path) -> Optional[int]:
    m = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else None


def find_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(models_dir)
    ckpts = sorted(
        models_dir.glob(args.checkpoint_pattern),
        key=lambda p: checkpoint_epoch(p) if checkpoint_epoch(p) is not None else 10**9,
    )
    out: list[tuple[str, Path]] = []
    for p in ckpts:
        ep = checkpoint_epoch(p)
        label = f"epoch {ep:04d}" if ep is not None else p.stem
        out.append((label, p))
    if args.include_final:
        final_path = models_dir / "final_lejepa_checkpoint.pt"
        if final_path.exists():
            # Avoid duplicate if final and last epoch checkpoint likely represent the same epoch.
            if args.include_final_duplicate or not any(checkpoint_epoch(p) == args.final_epoch for _, p in out):
                out.append(("final", final_path))
    if args.max_checkpoints > 0 and len(out) > args.max_checkpoints:
        # Keep an approximately uniform subset, always including first and last.
        idx = np.linspace(0, len(out) - 1, args.max_checkpoints).round().astype(int)
        seen: set[int] = set()
        out = [out[i] for i in idx if not (i in seen or seen.add(int(i)))]
    if not out:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_pattern!r} in {models_dir}")
    return out


def sample_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if args.max_samples <= 0 or len(df) <= args.max_samples:
        return df

    rng = np.random.default_rng(args.seed)
    if args.sampling == "random":
        idx = rng.choice(len(df), size=args.max_samples, replace=False)
        return df.iloc[np.sort(idx)].reset_index(drop=True)

    if args.sampling == "balanced_collapsed":
        classes = ["routine", "follow_up", "biopsy"]
        per_class = max(1, args.max_samples // len(classes))
        selected: list[int] = []
        for cls in classes:
            pool = np.flatnonzero(df["collapsed_birads"].to_numpy() == cls)
            if len(pool) == 0:
                continue
            n = min(per_class, len(pool))
            selected.extend(rng.choice(pool, size=n, replace=False).tolist())
        if len(selected) < args.max_samples:
            remaining = np.array([i for i in range(len(df)) if i not in set(selected)], dtype=int)
            if len(remaining) > 0:
                n_extra = min(args.max_samples - len(selected), len(remaining))
                selected.extend(rng.choice(remaining, size=n_extra, replace=False).tolist())
        selected = sorted(selected)
        return df.iloc[selected].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    raise ValueError(f"Unknown sampling mode: {args.sampling}")


@torch.inference_mode()
def extract_features_for_checkpoint(
    label: str,
    checkpoint_path: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
    aug_cfg: dict[str, Any],
    device: torch.device,
    full_num_rows: int,
    model_cfg_hint: Optional[ModelConfig] = None,
) -> tuple[np.ndarray, ModelConfig]:
    state_dict, cfg_dict, ckpt_aug_cfg = load_checkpoint_payload(checkpoint_path)
    if not aug_cfg and ckpt_aug_cfg:
        aug_cfg = ckpt_aug_cfg
    image_size = int(cfg_dict.get("image_size") or deep_get(aug_cfg, ["image", "output_size"], 384))
    model_cfg = ModelConfig(
        image_size=image_size,
        backbone_name=str(
            cfg_dict.get(
                "backbone_name",
                cfg_dict.get("backbone", model_cfg_hint.backbone_name if model_cfg_hint else "vit_small_patch8_224"),
            )
        ),
        backbone_output_dim=int(
            cfg_dict.get("backbone_output_dim", model_cfg_hint.backbone_output_dim if model_cfg_hint else 512)
        ),
        projection_dim=int(cfg_dict.get("projection_dim", model_cfg_hint.projection_dim if model_cfg_hint else 16)),
        projector_hidden_dim=int(
            cfg_dict.get("projector_hidden_dim", model_cfg_hint.projector_hidden_dim if model_cfg_hint else 2048)
        ),
        drop_path_rate=float(cfg_dict.get("drop_path_rate", model_cfg_hint.drop_path_rate if model_cfg_hint else 0.1)),
    )

    hints = model_hints({"model_state_dict": state_dict, "config": cfg_dict})
    model_cfg.backbone_num_classes = int(hints.get("backbone_num_classes", model_cfg.backbone_output_dim))
    model_cfg.backbone_output_dim = int(hints.get("backbone_output_dim", model_cfg.backbone_output_dim))
    model = ViTEncoder(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    validate_bin(args.bin, full_num_rows, args.image_height, args.image_width, args.memmap_dtype)

    transform = EvalMammographyTransform(aug_cfg, image_size=image_size)
    ds = MedJEPAPCADataset(
        df=df,
        bin_path=args.bin,
        full_num_rows=full_num_rows,
        image_shape=(args.image_height, args.image_width),
        dtype=args.memmap_dtype,
        transform=transform,
        normalize_mode=args.normalize_mode,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_batch,
    )

    features, _, _ = extract_embedding_batches(model, loader, device, True, f"Extracting {label}")
    features = features.numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, model_cfg


def cache_path_for(args: argparse.Namespace, checkpoint_path: Path, n_rows: int) -> Path:
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output_pdf).with_suffix(".cache")
    safe_name = checkpoint_path.stem.replace("/", "_")
    return cache_dir / f"{safe_name}_n{n_rows}_seed{args.seed}.npz"


def extract_or_load_features(
    label: str,
    checkpoint_path: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
    aug_cfg: dict[str, Any],
    device: torch.device,
    full_num_rows: int,
    model_cfg_hint: Optional[ModelConfig],
) -> tuple[np.ndarray, ModelConfig]:
    cp = cache_path_for(args, checkpoint_path, len(df))
    signature = None
    if args.use_cache:
        files = []
        for path in (checkpoint_path, Path(args.bin)):
            stat = path.stat()
            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        identity = {
            "version": "shared_core_v1",
            "files": files,
            "rows": df["original_index"].tolist(),
            "full_num_rows": full_num_rows,
            "augmentation": aug_cfg,
            "hint": asdict(model_cfg_hint) if model_cfg_hint else None,
            "preprocessing": {
                key: getattr(args, key)
                for key in (
                    "image_height",
                    "image_width",
                    "memmap_dtype",
                    "normalize_mode",
                    "percentile_low",
                    "percentile_high",
                )
            },
        }
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.use_cache and cp.exists():
        with np.load(cp, allow_pickle=False) as data:
            if "signature" in data and str(data["signature"]) == signature:
                print(f"Loaded cached features for {label}: {cp}")
                return data["features"].astype(np.float32), ModelConfig(**json.loads(str(data["model_config"])))
        print(f"Recomputing incompatible feature cache: {cp}")
    features, cfg = extract_features_for_checkpoint(
        label, checkpoint_path, df, args, aug_cfg, device, full_num_rows, model_cfg_hint
    )
    if args.use_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp, features=features.astype(np.float32), signature=signature, model_config=json.dumps(asdict(cfg))
        )
        print(f"Cached features for {label}: {cp}")
    return features, cfg


# -----------------------------
# Plotting
# -----------------------------

BLUE_PURPLE_RED = LinearSegmentedColormap.from_list("blue_purple_red", ["#2166AC", "#7B3294", "#B2182B"])

ORDINAL_COLOR_ORDERS: dict[str, list[str]] = {
    "collapsed_birads": ["routine", "follow_up", "biopsy"],
    "birads_numeric": ["1", "2", "3", "4", "5"],
}


def colors_for_labels(col: str, labels: pd.Series) -> tuple[list[str], dict[str, Any], bool]:
    labels = labels.astype(str)
    vals = set(labels.unique())
    if col in ORDINAL_COLOR_ORDERS:
        cats = [v for v in ORDINAL_COLOR_ORDERS[col] if v in vals]
        cats += sorted(vals - set(cats))
        scale_cats = [c for c in cats if c not in {"unknown", "Other", "missing"}]
        color_map: dict[str, Any] = {}
        if len(scale_cats) == 1:
            color_map[scale_cats[0]] = BLUE_PURPLE_RED(0.5)
        else:
            for i, cat in enumerate(scale_cats):
                color_map[cat] = BLUE_PURPLE_RED(i / max(1, len(scale_cats) - 1))
        for cat in cats:
            if cat not in color_map:
                color_map[cat] = "#8C8C8C"
        return cats, color_map, True
    cats = list(labels.value_counts().index)
    cmap = plt.get_cmap("tab20", max(1, len(cats)))
    color_map = {cat: cmap(i) for i, cat in enumerate(cats)}
    return cats, color_map, False


def panel_grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def add_text_page(pdf: PdfPages, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def set_same_2d_limits(axes: list[Any], coords: dict[str, np.ndarray], pad_frac: float = 0.05) -> None:
    all_z = np.concatenate([z[:, :2] for z in coords.values()], axis=0)
    xmin, ymin = all_z.min(axis=0)
    xmax, ymax = all_z.max(axis=0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    for ax in axes:
        ax.set_xlim(xmin - dx * pad_frac, xmax + dx * pad_frac)
        ax.set_ylim(ymin - dy * pad_frac, ymax + dy * pad_frac)


def set_same_3d_limits(axes: list[Any], coords: dict[str, np.ndarray], pad_frac: float = 0.05) -> None:
    all_z = np.concatenate([z[:, :3] for z in coords.values()], axis=0)
    mins = all_z.min(axis=0)
    maxs = all_z.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    for ax in axes:
        ax.set_xlim(mins[0] - spans[0] * pad_frac, maxs[0] + spans[0] * pad_frac)
        ax.set_ylim(mins[1] - spans[1] * pad_frac, maxs[1] + spans[1] * pad_frac)
        ax.set_zlim(mins[2] - spans[2] * pad_frac, maxs[2] + spans[2] * pad_frac)


def plot_2d_progression_page(
    pdf: PdfPages,
    coords: dict[str, np.ndarray],
    labels: pd.Series,
    col: str,
    title: str,
    evr: np.ndarray,
    args: argparse.Namespace,
) -> None:
    ckpt_labels = list(coords.keys())
    rows, cols = panel_grid(len(ckpt_labels))
    fig, axes_arr = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), squeeze=False)
    axes = [ax for row in axes_arr for ax in row]
    for ax in axes:
        ax.axis("off")

    cats, color_map, is_ordinal = colors_for_labels(col, labels)
    labels_np = labels.astype(str).to_numpy()
    used_axes: list[Any] = []
    for ax, ckpt_label in zip(axes, ckpt_labels):
        ax.axis("on")
        z = coords[ckpt_label]
        for cat in cats:
            mask = labels_np == str(cat)
            if mask.any():
                ax.scatter(
                    z[mask, 0],
                    z[mask, 1],
                    s=args.point_size_2d,
                    alpha=args.alpha_2d,
                    color=color_map[cat],
                    label=str(cat),
                    linewidths=0,
                    rasterized=True,
                )
        ax.set_title(ckpt_label, fontsize=11)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
        ax.grid(alpha=0.2)
        used_axes.append(ax)

    if args.same_axes:
        set_same_2d_limits(used_axes, coords)

    handles, legend_labels = used_axes[0].get_legend_handles_labels()
    if len(legend_labels) <= args.max_categories:
        legend_title = "ordered scale" if is_ordinal else col
        fig.legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="lower center",
            ncol=min(len(legend_labels), 6),
            fontsize=9,
            title_fontsize=9,
            frameon=True,
        )
        bottom = 0.10
    else:
        bottom = 0.03
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, bottom, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_3d_progression_page(
    pdf: PdfPages,
    coords: dict[str, np.ndarray],
    labels: pd.Series,
    col: str,
    title: str,
    evr: np.ndarray,
    args: argparse.Namespace,
) -> None:
    ckpt_labels = list(coords.keys())
    rows, cols = panel_grid(len(ckpt_labels))
    fig = plt.figure(figsize=(5.2 * cols, 4.7 * rows))
    axes: list[Any] = []
    cats, color_map, is_ordinal = colors_for_labels(col, labels)
    labels_np = labels.astype(str).to_numpy()

    for i, ckpt_label in enumerate(ckpt_labels, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        axes.append(ax)
        z = coords[ckpt_label]
        for cat in cats:
            mask = labels_np == str(cat)
            if mask.any():
                ax.scatter(
                    z[mask, 0],
                    z[mask, 1],
                    z[mask, 2],
                    s=args.point_size_3d,
                    alpha=args.alpha_3d,
                    color=color_map[cat],
                    label=str(cat),
                    linewidths=0,
                )
        ax.set_title(ckpt_label, fontsize=11)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
        ax.set_zlabel(f"PC3 ({evr[2] * 100:.1f}%)")
        ax.view_init(elev=args.view_elev, azim=args.view_azim)

    if args.same_axes:
        set_same_3d_limits(axes, coords)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if len(legend_labels) <= args.max_categories:
        legend_title = "ordered scale" if is_ordinal else col
        fig.legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="lower center",
            ncol=min(len(legend_labels), 6),
            fontsize=9,
            title_fontsize=9,
            frameon=True,
        )
        bottom = 0.10
    else:
        bottom = 0.03
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, bottom, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def compute_local_pca_coordinates(
    features_by_label: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit one independent PCA per checkpoint.

    This is a *local representation view*: each checkpoint is centered and rotated into its
    own best PCA basis. It is useful for inspecting the internal structure of each checkpoint
    without the global scale contraction dominating the visualization.
    """
    coords: dict[str, np.ndarray] = {}
    evr_by_label: dict[str, np.ndarray] = {}
    for label, features in features_by_label.items():
        x = features
        if args.standardize_features:
            x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=3, random_state=args.seed)
        z = pca.fit_transform(x)
        coords[label] = z
        evr_by_label[label] = pca.explained_variance_ratio_
    return coords, evr_by_label


def plot_2d_local_pca_page(
    pdf: PdfPages,
    coords: dict[str, np.ndarray],
    evr_by_label: dict[str, np.ndarray],
    labels: pd.Series,
    col: str,
    title: str,
    args: argparse.Namespace,
) -> None:
    ckpt_labels = list(coords.keys())
    rows, cols = panel_grid(len(ckpt_labels))
    fig, axes_arr = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.7 * rows), squeeze=False)
    axes = [ax for row in axes_arr for ax in row]
    for ax in axes:
        ax.axis("off")

    cats, color_map, is_ordinal = colors_for_labels(col, labels)
    labels_np = labels.astype(str).to_numpy()
    used_axes: list[Any] = []
    for ax, ckpt_label in zip(axes, ckpt_labels):
        ax.axis("on")
        z = coords[ckpt_label]
        evr = evr_by_label[ckpt_label]
        for cat in cats:
            mask = labels_np == str(cat)
            if mask.any():
                ax.scatter(
                    z[mask, 0],
                    z[mask, 1],
                    s=args.point_size_2d,
                    alpha=args.alpha_2d,
                    color=color_map[cat],
                    label=str(cat),
                    linewidths=0,
                    rasterized=True,
                )
        ax.set_title(f"{ckpt_label}\nPC1={evr[0] * 100:.1f}%, PC2={evr[1] * 100:.1f}%", fontsize=10)
        ax.set_xlabel(f"local PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"local PC2 ({evr[1] * 100:.1f}%)")
        ax.grid(alpha=0.2)
        used_axes.append(ax)

    if args.local_same_axes:
        set_same_2d_limits(used_axes, coords)

    handles, legend_labels = used_axes[0].get_legend_handles_labels()
    if len(legend_labels) <= args.max_categories:
        legend_title = "ordered scale" if is_ordinal else col
        fig.legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="lower center",
            ncol=min(len(legend_labels), 6),
            fontsize=9,
            title_fontsize=9,
            frameon=True,
        )
        bottom = 0.11
    else:
        bottom = 0.03
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, bottom, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_3d_local_pca_page(
    pdf: PdfPages,
    coords: dict[str, np.ndarray],
    evr_by_label: dict[str, np.ndarray],
    labels: pd.Series,
    col: str,
    title: str,
    args: argparse.Namespace,
) -> None:
    ckpt_labels = list(coords.keys())
    rows, cols = panel_grid(len(ckpt_labels))
    fig = plt.figure(figsize=(5.2 * cols, 4.9 * rows))
    axes: list[Any] = []
    cats, color_map, is_ordinal = colors_for_labels(col, labels)
    labels_np = labels.astype(str).to_numpy()

    for i, ckpt_label in enumerate(ckpt_labels, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        axes.append(ax)
        z = coords[ckpt_label]
        evr = evr_by_label[ckpt_label]
        for cat in cats:
            mask = labels_np == str(cat)
            if mask.any():
                ax.scatter(
                    z[mask, 0],
                    z[mask, 1],
                    z[mask, 2],
                    s=args.point_size_3d,
                    alpha=args.alpha_3d,
                    color=color_map[cat],
                    label=str(cat),
                    linewidths=0,
                )
        ax.set_title(
            f"{ckpt_label}\nPC1={evr[0] * 100:.1f}%, PC2={evr[1] * 100:.1f}%, PC3={evr[2] * 100:.1f}%", fontsize=9
        )
        ax.set_xlabel("local PC1")
        ax.set_ylabel("local PC2")
        ax.set_zlabel("local PC3")
        ax.view_init(elev=args.view_elev, azim=args.view_azim)

    if args.local_same_axes:
        set_same_3d_limits(axes, coords)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if len(legend_labels) <= args.max_categories:
        legend_title = "ordered scale" if is_ordinal else col
        fig.legend(
            handles,
            legend_labels,
            title=legend_title,
            loc="lower center",
            ncol=min(len(legend_labels), 6),
            fontsize=9,
            title_fontsize=9,
            frameon=True,
        )
        bottom = 0.11
    else:
        bottom = 0.03
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, bottom, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_progression_pdf(
    df: pd.DataFrame,
    features_by_label: dict[str, np.ndarray],
    args: argparse.Namespace,
    output_pdf: Path,
    checkpoint_paths: list[tuple[str, Path]],
) -> None:
    labels_for_concat = list(features_by_label.keys())
    concat_features = np.concatenate([features_by_label[k] for k in labels_for_concat], axis=0)

    scaler: Optional[StandardScaler] = None
    pca_input = concat_features
    if args.standardize_features:
        scaler = StandardScaler().fit(concat_features)
        pca_input = scaler.transform(concat_features)

    pca = PCA(n_components=3, random_state=args.seed)
    concat_z = pca.fit_transform(pca_input)
    evr = pca.explained_variance_ratio_

    coords: dict[str, np.ndarray] = {}
    offset = 0
    n = len(df)
    for label in labels_for_concat:
        z = concat_z[offset : offset + n]
        coords[label] = z
        offset += n

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        lines = [
            "MedJEPA checkpoint PCA progression report",
            "==========================================",
            "",
            f"Models dir: {args.models_dir}",
            f"Checkpoints: {len(checkpoint_paths)}",
            *[f"  - {label}: {path.name}" for label, path in checkpoint_paths],
            "",
            f"Split: {args.split}",
            f"Sampling: {args.sampling}",
            f"Rows per checkpoint: {len(df):,}",
            f"Feature dimension: {next(iter(features_by_label.values())).shape[1]}",
            "PCA fit: shared/global across all checkpoints and sampled images",
            f"Feature standardization before PCA: {args.standardize_features}",
            f"PCA explained variance: PC1={evr[0] * 100:.2f}%, PC2={evr[1] * 100:.2f}%, PC3={evr[2] * 100:.2f}%",
            f"Cumulative PC1-PC3: {evr[:3].sum() * 100:.2f}%",
            "",
            "Collapsed BI-RADS counts:",
            str(df["collapsed_birads"].value_counts().to_dict()),
            "",
            "Machine family counts:",
            str(df["machine_family"].value_counts().to_dict()),
            "",
            "View counts:",
            str(df["view"].value_counts().to_dict()),
            "",
            "Color convention:",
            "collapsed_birads uses blue -> purple -> red: routine -> follow_up -> biopsy.",
            "machine_family and view use categorical colors.",
        ]
        add_text_page(pdf, lines)

        plot_2d_progression_page(
            pdf,
            coords,
            make_label_series(df, "collapsed_birads", args.max_categories),
            "collapsed_birads",
            "2D PCA embedding progression colored by collapsed BI-RADS",
            evr,
            args,
        )
        plot_2d_progression_page(
            pdf,
            coords,
            make_label_series(df, "machine_family", args.max_categories),
            "machine_family",
            "2D PCA embedding progression colored by machine family",
            evr,
            args,
        )
        plot_2d_progression_page(
            pdf,
            coords,
            make_label_series(df, "view", args.max_categories),
            "view",
            "2D PCA embedding progression colored by view",
            evr,
            args,
        )
        plot_3d_progression_page(
            pdf,
            coords,
            make_label_series(df, "collapsed_birads", args.max_categories),
            "collapsed_birads",
            "3D PCA embedding progression colored by collapsed BI-RADS",
            evr,
            args,
        )

        if args.include_local_pca_pages:
            local_coords, local_evr_by_label = compute_local_pca_coordinates(features_by_label, args)
            add_text_page(
                pdf,
                [
                    "Local PCA / local representation view",
                    "====================================",
                    "",
                    "These pages fit PCA separately for each checkpoint.",
                    "Each panel is centered and rotated into that checkpoint's own best PCA basis.",
                    "This removes the global scale-contraction effect and shows the internal structure",
                    "of each checkpoint in its local coordinate system.",
                    "",
                    f"Local panels use same axes across checkpoints: {args.local_same_axes}",
                    f"Feature standardization before each local PCA: {args.standardize_features}",
                    "",
                    "Important: local PCA axes are not directly comparable across epochs;",
                    "they are intended to match the style of a final-only PCA report.",
                    "The epoch-0300 local panel should therefore align with the detailed final PCA",
                    "when the same checkpoint, sample, preprocessing, and random seed are used.",
                ],
            )
            plot_2d_local_pca_page(
                pdf,
                local_coords,
                local_evr_by_label,
                make_label_series(df, "collapsed_birads", args.max_categories),
                "collapsed_birads",
                "2D local PCA per checkpoint colored by collapsed BI-RADS",
                args,
            )
            plot_2d_local_pca_page(
                pdf,
                local_coords,
                local_evr_by_label,
                make_label_series(df, "machine_family", args.max_categories),
                "machine_family",
                "2D local PCA per checkpoint colored by machine family",
                args,
            )
            plot_2d_local_pca_page(
                pdf,
                local_coords,
                local_evr_by_label,
                make_label_series(df, "view", args.max_categories),
                "view",
                "2D local PCA per checkpoint colored by view",
                args,
            )
            plot_3d_local_pca_page(
                pdf,
                local_coords,
                local_evr_by_label,
                make_label_series(df, "collapsed_birads", args.max_categories),
                "collapsed_birads",
                "3D local PCA per checkpoint colored by collapsed BI-RADS",
                args,
            )

    print(f"Saved PCA progression PDF to: {output_pdf}")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create side-by-side PCA progression PDF across MedJEPA checkpoints.")
    p.add_argument(
        "--models-dir", required=True, type=str, help="Run models/ directory containing checkpoint_epoch_*.pt files."
    )
    p.add_argument("--checkpoint-pattern", type=str, default="checkpoint_epoch_*.pt")
    p.add_argument("--include-final", action="store_true", help="Also include final_lejepa_checkpoint.pt if present.")
    p.add_argument(
        "--include-final-duplicate",
        action="store_true",
        help="Include final even if checkpoint_epoch_0300.pt is present.",
    )
    p.add_argument(
        "--final-epoch", type=int, default=300, help="Epoch number used to detect final/checkpoint duplicate."
    )
    p.add_argument(
        "--max-checkpoints", type=int, default=0, help="0 = all; otherwise use an approximately uniform subset."
    )

    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--bin", required=True, type=str)
    p.add_argument("--train-csv", type=str, default="")
    p.add_argument("--val-csv", type=str, default="")
    p.add_argument("--test-csv", type=str, default="")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument(
        "--aug-config",
        type=str,
        default="",
        help="Optional. If omitted, uses checkpoint augmentation_config if available.",
    )
    p.add_argument("--output-pdf", required=True, type=str)

    p.add_argument("--max-samples", type=int, default=1998)
    p.add_argument("--sampling", choices=["random", "balanced_collapsed"], default="balanced_collapsed")
    p.add_argument("--max-categories", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-cache", action="store_true", help="Cache extracted features as compressed npz files.")
    p.add_argument("--cache-dir", type=str, default="", help="Optional cache directory. Default: <output_pdf>.cache")
    p.add_argument(
        "--standardize-features", action="store_true", help="Z-score features across all checkpoints before global PCA."
    )

    p.add_argument("--image-height", type=int, default=512)
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--memmap-dtype", type=str, default="uint16")
    p.add_argument("--normalize-mode", type=str, default="uint16", choices=["uint16", "per_image_percentile"])
    p.add_argument("--percentile-low", type=float, default=1.0)
    p.add_argument("--percentile-high", type=float, default=99.0)

    p.add_argument(
        "--same-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use same axis limits across checkpoint panels for the shared/global PCA pages.",
    )
    p.add_argument(
        "--include-local-pca-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also add local PCA pages where PCA is fitted separately per checkpoint.",
    )
    p.add_argument(
        "--local-same-axes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use same axis limits across checkpoint panels on the local PCA pages. Default false to show each checkpoint in its own final-report-like view.",
    )
    p.add_argument("--point-size-2d", type=float, default=8.0)
    p.add_argument("--point-size-3d", type=float, default=5.0)
    p.add_argument("--alpha-2d", type=float, default=0.60)
    p.add_argument("--alpha-3d", type=float, default=0.45)
    p.add_argument("--view-elev", type=float, default=22.0)
    p.add_argument("--view-azim", type=float, default=-60.0)
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    checkpoints = find_checkpoints(args)
    print("Checkpoints selected:")
    for label, path in checkpoints:
        print(f"  {label}: {path}")

    print("Loading augmentation config...")
    aug_cfg: dict[str, Any] = {}
    if args.aug_config:
        with open(args.aug_config, "r", encoding="utf-8") as f:
            aug_cfg = json.load(f)

    print("Loading split metadata...")
    df, full_num_rows = build_eval_dataframe(args)
    df = sample_dataframe(df, args)
    df = add_derived_metadata(df)
    print(f"Rows selected: {len(df):,}")
    print("Collapsed counts:", df["collapsed_birads"].value_counts().to_dict())
    print("Machine family counts:", df["machine_family"].value_counts().to_dict())
    print("View counts:", df["view"].value_counts().to_dict())

    if not aug_cfg:
        # Try first checkpoint metadata as fallback.
        _, _, aug_cfg = load_checkpoint_payload(checkpoints[0][1])
    if not aug_cfg:
        raise ValueError(
            "No augmentation config found. Pass --aug-config or use checkpoints containing augmentation_config."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    features_by_label: dict[str, np.ndarray] = {}
    model_cfg_hint: Optional[ModelConfig] = None
    for label, path in checkpoints:
        features, cfg = extract_or_load_features(label, path, df, args, aug_cfg, device, full_num_rows, model_cfg_hint)
        features_by_label[label] = features.astype(np.float32)
        model_cfg_hint = cfg
        print(f"{label} features: {features.shape}")

    print("Creating progression PDF...")
    create_progression_pdf(df, features_by_label, args, Path(args.output_pdf), checkpoints)


if __name__ == "__main__":
    main()
