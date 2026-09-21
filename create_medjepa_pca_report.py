#!/usr/bin/env python3
"""
Create a single multi-page PCA PDF report for a saved MedJEPA / LeJEPA checkpoint.

The script:
- loads a final_lejepa_checkpoint.pt saved by train_medjepa_mg_v3.py,
- rebuilds the ViT encoder,
- extracts deterministic backbone embeddings from a chosen split,
- computes ONE PCA coordinate system,
- creates multiple PCA plots using the same coordinates, colored by different metadata columns.

This is meant for checking whether the learned latent space is organized by clinical labels
(collapsed BI-RADS / numeric BI-RADS) or by nuisance variables such as source dataset or machine.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.decomposition import PCA
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("Missing dependency: timm. Install with: pip install timm") from exc


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


class ViTEncoder(nn.Module):
    """Compatible with train_medjepa_mg_v3.py checkpoints."""

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
        # x: [B,V,1,H,W]
        b, v = x.shape[:2]
        flat = x.flatten(0, 1)
        emb = self.backbone(flat)
        proj = self.proj(emb).reshape(b, v, -1)  # [B,V,D]
        return emb, proj


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def deep_get(dct: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -----------------------------
# Labels / CSV handling
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


def build_eval_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    full_raw = read_csv_clean(args.full_csv)
    full_df = prepare_labels(full_raw)
    if "original_index" not in full_df.columns:
        full_df = full_df.reset_index(drop=False).rename(columns={"index": "original_index"})

    split_frames: list[pd.DataFrame] = []
    paths = {
        "train": args.train_csv,
        "val": args.val_csv,
        "test": args.test_csv,
    }
    selected = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split_name in selected:
        path = paths.get(split_name)
        if not path:
            raise ValueError(f"--split {args.split!r} requires --{split_name}-csv")
        df = ensure_original_index(read_csv_clean(path), full_raw, split_name)
        df = prepare_labels(df)
        df["split"] = split_name
        split_frames.append(df)

    out = pd.concat(split_frames, ignore_index=True)
    return out, len(full_df)


# -----------------------------
# Derived metadata for better PCA colorings
# -----------------------------

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


def _extract_age_group(x: Any) -> str:
    # Expected examples: "40-49", "50-59", or numeric-like values.
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "unknown", "missing"}:
        return "unknown"
    m = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"\d{2,3}", s)
    if m:
        age = int(m.group(0))
        lo = (age // 10) * 10
        return f"{lo}-{lo+9}"
    return s


def _age_group_sort_key(label: Any) -> int:
    s = str(label)
    m = re.search(r"\d{2,3}", s)
    if m:
        return int(m.group(0))
    return 10_000


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


def _has_segmentation(x: Any) -> str:
    if pd.isna(x):
        return "no_segmentation"
    s = str(x).strip().lower()
    if not s or s in {"nan", "none", "null", "missing", "[]"}:
        return "no_segmentation"
    return "has_segmentation"


def add_derived_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable metadata columns for interpretable PCA plots."""
    df = df.copy()

    if "context" in df.columns:
        parsed = df["context"].apply(_safe_json_loads)
        df["age_group"] = parsed.apply(lambda d: _extract_age_group(deep_get(d, ["patient", "age"], "unknown")))
        df["view"] = parsed.apply(lambda d: _normalize_view(deep_get(d, ["exam", "view"], "unknown")))
        df["laterality"] = parsed.apply(lambda d: _normalize_laterality(deep_get(d, ["exam", "laterality"], "unknown")))
        df["view_laterality"] = df["view"].astype(str) + " / " + df["laterality"].astype(str)
    else:
        df["age_group"] = "unknown"
        df["view"] = "unknown"
        df["laterality"] = "unknown"
        df["view_laterality"] = "unknown"

    if "machine" in df.columns:
        df["machine_family"] = df["machine"].apply(_machine_family)
    else:
        df["machine_family"] = "unknown"

    if "segmentation" in df.columns:
        df["has_segmentation"] = df["segmentation"].apply(_has_segmentation)
    else:
        df["has_segmentation"] = "unknown"

    # Make plotting values compact and consistent.
    df["birads_numeric"] = df["birads_numeric"].astype(int).astype(str)
    return df


# -----------------------------
# Deterministic eval transform / dataset
# -----------------------------

class EvalMammographyTransform(nn.Module):
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


class MedJEPAPCADataset(Dataset):
    def __init__(self, df: pd.DataFrame, bin_path: str | Path, full_num_rows: int,
                 image_shape: tuple[int, int], dtype: str, transform: nn.Module,
                 normalize_mode: str = "uint16", percentile_low: float = 1.0, percentile_high: float = 99.0):
        self.df = df.reset_index(drop=True)
        self.bin_path = Path(bin_path)
        self.full_num_rows = full_num_rows
        self.image_shape = image_shape
        self.dtype = np.dtype(dtype)
        self.transform = transform
        self.normalize_mode = normalize_mode
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self._imgs: Optional[np.memmap] = None

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
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_tensor(int(row["original_index"]))
        x = self.transform(x).unsqueeze(0)  # [V=1,1,H,W]
        return x, idx


def collate_batch(batch):
    views = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return views, indices


# -----------------------------
# Sampling / features / plots
# -----------------------------

def sample_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.max_samples <= 0 or len(df) <= args.max_samples:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    if args.sampling == "random":
        idx = rng.choice(len(df), size=args.max_samples, replace=False)
        return df.iloc[np.sort(idx)].reset_index(drop=True)

    if args.sampling == "balanced_collapsed":
        classes = ["routine", "follow_up", "biopsy"]
        per_class = max(1, args.max_samples // len(classes))
        parts = []
        for cls in classes:
            sub = df[df["collapsed_birads"] == cls]
            if len(sub) == 0:
                continue
            n = min(per_class, len(sub))
            parts.append(sub.sample(n=n, random_state=args.seed))
        out = pd.concat(parts, ignore_index=True)
        if len(out) < args.max_samples:
            remaining = df.drop(index=out.index, errors="ignore")
        return out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    raise ValueError(f"Unknown sampling mode: {args.sampling}")


@torch.inference_mode()
def extract_features(df: pd.DataFrame, args: argparse.Namespace, model: nn.Module, aug_cfg: dict[str, Any], device: torch.device,
                     full_num_rows: int, image_size: int) -> np.ndarray:
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
    model.eval()
    feats: list[torch.Tensor] = []
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    for views, _ in tqdm(loader, desc="Extracting embeddings"):
        views = views.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            emb, _ = model(views)
        feats.append(emb.float().cpu())
    return torch.cat(feats, dim=0).numpy()


def make_label_series(df: pd.DataFrame, col: str, max_categories: int) -> pd.Series:
    if col not in df.columns:
        raise KeyError(col)
    s = df[col].fillna("missing").astype(str)
    vc = s.value_counts(dropna=False)
    if len(vc) > max_categories:
        keep = set(vc.index[:max_categories - 1])
        s = s.where(s.isin(keep), other="Other")
    return s


ORDINAL_COLOR_ORDERS: dict[str, list[str]] = {
    "collapsed_birads": ["routine", "follow_up", "biopsy"],
    "birads_numeric": ["1", "2", "3", "4", "5"],
    "has_segmentation": ["no_segmentation", "has_segmentation"],
}

BLUE_PURPLE_RED = LinearSegmentedColormap.from_list(
    "blue_purple_red", ["#2166AC", "#7B3294", "#B2182B"]
)


def ordinal_order_for_column(col: str, labels: pd.Series) -> Optional[list[str]]:
    vals = set(labels.dropna().astype(str).unique())
    if col in ORDINAL_COLOR_ORDERS:
        order = [v for v in ORDINAL_COLOR_ORDERS[col] if v in vals]
        # Keep any unexpected values at the end in stable lexical order.
        order += sorted(vals - set(order))
        return order
    if col == "age_group":
        known = [v for v in vals if v != "unknown"]
        known = sorted(known, key=_age_group_sort_key)
        if "unknown" in vals:
            known.append("unknown")
        return known
    return None


def colors_for_labels(col: str, labels: pd.Series, max_legend: int) -> tuple[list[str], dict[str, Any], bool]:
    """Return ordered categories, color map and whether the coloring is ordinal."""
    labels = labels.astype(str)
    ordinal_order = ordinal_order_for_column(col, labels)
    if ordinal_order is not None and len(ordinal_order) >= 2:
        # Unknown/Other get neutral grey; true scale categories get blue->purple->red.
        scale_cats = [c for c in ordinal_order if c not in {"unknown", "Other", "missing"}]
        color_map: dict[str, Any] = {}
        if len(scale_cats) == 1:
            color_map[scale_cats[0]] = BLUE_PURPLE_RED(0.5)
        else:
            for i, cat in enumerate(scale_cats):
                color_map[cat] = BLUE_PURPLE_RED(i / max(1, len(scale_cats) - 1))
        for cat in ordinal_order:
            if cat not in color_map:
                color_map[cat] = "#8C8C8C"
        return ordinal_order, color_map, True

    cats = list(labels.value_counts().index)
    cmap = plt.get_cmap("tab20", max(1, len(cats)))
    color_map = {cat: cmap(i) for i, cat in enumerate(cats)}
    return cats, color_map, False


def plot_pca2(ax, z: np.ndarray, labels: pd.Series, title: str, col: str, max_legend: int = 14) -> None:
    labels = labels.astype(str)
    cats, color_map, is_ordinal = colors_for_labels(col, labels, max_legend)
    # For ordinal categories, plot lower values first and higher values later so high-severity points remain visible.
    for cat in cats:
        mask = labels.to_numpy() == cat
        if mask.any():
            ax.scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.65, label=str(cat), color=color_map[cat])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    if len(cats) <= max_legend:
        title_label = "ordered scale" if is_ordinal else None
        ax.legend(title=title_label, markerscale=2, fontsize=8, title_fontsize=8, loc="best", frameon=True)
    else:
        ax.text(0.02, 0.98, f"{len(cats)} categories; legend omitted", transform=ax.transAxes, va="top", fontsize=8)


def plot_pca3(ax, z: np.ndarray, labels: pd.Series, title: str, col: str, max_legend: int = 10) -> None:
    labels = labels.astype(str)
    cats, color_map, is_ordinal = colors_for_labels(col, labels, max_legend)
    for cat in cats:
        mask = labels.to_numpy() == cat
        if mask.any():
            ax.scatter(z[mask, 0], z[mask, 1], z[mask, 2], s=8, alpha=0.55, label=str(cat), color=color_map[cat])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if len(cats) <= max_legend:
        title_label = "ordered scale" if is_ordinal else None
        ax.legend(title=title_label, markerscale=2, fontsize=7, title_fontsize=7, loc="best", frameon=True)


def add_text_page(pdf: PdfPages, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_pdf_report(df: pd.DataFrame, features: np.ndarray, args: argparse.Namespace, output_pdf: Path) -> None:
    pca3 = PCA(n_components=3, random_state=args.seed)
    z3 = pca3.fit_transform(features)
    evr = pca3.explained_variance_ratio_

    # Default report columns: clinical labels first, then interpretable domain/context labels.
    # Raw columns such as exam/context/segmentation are intentionally not shown by default because
    # they often contain long file paths or JSON strings. Use --include-raw-columns to add them.
    candidate_cols = [
        "collapsed_birads",
        "birads_numeric",
        "age_group",
        "view",
        "laterality",
        "view_laterality",
        "dataset",
        "machine_family",
        "machine",
        "has_segmentation",
        "split",
    ]
    if args.include_raw_columns:
        candidate_cols.extend(["exam", "context", "segmentation", "modality"])
    plot_cols = [c for c in candidate_cols if c in df.columns and df[c].nunique(dropna=False) > 1]

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        lines = [
            "MedJEPA PCA latent-space report",
            "================================",
            "",
            f"Checkpoint: {args.checkpoint}",
            f"Split: {args.split}",
            f"Sampling: {args.sampling}",
            f"Rows plotted: {len(df):,}",
            f"Feature dimension: {features.shape[1]}",
            f"PCA explained variance: PC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%, PC3={evr[2]*100:.2f}%",
            f"Cumulative PC1-PC3: {evr[:3].sum()*100:.2f}%",
            "",
            "Collapsed BI-RADS counts:",
            str(df["collapsed_birads"].value_counts().to_dict()),
            "",
            "BI-RADS numeric counts:",
            str(df["birads_numeric"].value_counts().sort_index().to_dict()),
            "",
            "Included colorings:",
            ", ".join(plot_cols),
            "",
            "Color convention:",
            "Ordinal/scale labels use blue -> purple -> red from low to high.",
            "Examples: routine -> follow_up -> biopsy; BI-RADS 1 -> 5; younger -> older age groups.",
            "Nominal labels use categorical colors.",
        ]
        add_text_page(pdf, lines)

        # 2D overview page: 2x2 most important colorings.
        overview_cols = [c for c in ["collapsed_birads", "birads_numeric", "dataset", "machine_family"] if c in plot_cols]
        if overview_cols:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes_flat = axes.ravel()
            for ax in axes_flat:
                ax.axis("off")
            for ax, col in zip(axes_flat, overview_cols):
                ax.axis("on")
                labels = make_label_series(df, col, args.max_categories)
                plot_pca2(ax, z3[:, :2], labels, f"PCA 2D colored by {col}", col=col)
            fig.suptitle(f"Same PCA coordinates, different labels\nPC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Individual larger 2D pages for each metadata column.
        for col in plot_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = make_label_series(df, col, args.max_categories)
            plot_pca2(ax, z3[:, :2], labels, f"PCA 2D colored by {col}\nPC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%", col=col)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Additional PC-pair diagnostic pages for clinical labels.
        for col in ["collapsed_birads", "birads_numeric", "age_group", "view", "laterality", "dataset", "machine_family"]:
            if col not in plot_cols:
                continue
            labels = make_label_series(df, col, args.max_categories)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            pairs = [(0, 1), (0, 2), (1, 2)]
            for ax, (a, b) in zip(axes, pairs):
                cats, color_map, _ = colors_for_labels(col, labels.astype(str), args.max_categories)
                for cat in cats:
                    mask = labels.to_numpy().astype(str) == str(cat)
                    ax.scatter(z3[mask, a], z3[mask, b], s=8, alpha=0.6, label=str(cat), color=color_map[cat])
                ax.set_xlabel(f"PC{a+1}")
                ax.set_ylabel(f"PC{b+1}")
                ax.set_title(f"PC{a+1} vs PC{b+1}")
                ax.grid(alpha=0.2)
            handles, legend_labels = axes[0].get_legend_handles_labels()
            if len(legend_labels) <= 14:
                fig.legend(handles, legend_labels, loc="lower center", ncol=min(7, len(legend_labels)), fontsize=8)
                fig.tight_layout(rect=[0, 0.08, 1, 0.92])
            else:
                fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.suptitle(f"PCA pair diagnostics colored by {col}", fontsize=14)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # 3D pages for the most important labels.
        for col in ["collapsed_birads", "birads_numeric", "age_group", "view", "laterality", "dataset", "machine_family"]:
            if col not in plot_cols:
                continue
            labels = make_label_series(df, col, args.max_categories)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            plot_pca3(ax, z3, labels, f"PCA 3D colored by {col}\nPC1={evr[0]*100:.2f}%, PC2={evr[1]*100:.2f}%, PC3={evr[2]*100:.2f}%", col=col)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create one multi-page PCA PDF report from a saved MedJEPA checkpoint.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to final_lejepa_checkpoint.pt or state_dict .pt")
    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--bin", required=True, type=str)
    p.add_argument("--train-csv", type=str, default="")
    p.add_argument("--val-csv", type=str, default="")
    p.add_argument("--test-csv", type=str, default="")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--output-pdf", required=True, type=str)
    p.add_argument("--aug-config", type=str, default="", help="Optional. If omitted, uses checkpoint augmentation_config if available.")

    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--sampling", choices=["random", "balanced_collapsed"], default="balanced_collapsed")
    p.add_argument("--max-categories", type=int, default=12)
    p.add_argument("--include-raw-columns", action="store_true", help="Also plot raw exam/context/segmentation/modality columns. Usually noisy.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image-height", type=int, default=512)
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--memmap-dtype", type=str, default="uint16")
    p.add_argument("--normalize-mode", type=str, default="uint16", choices=["uint16", "per_image_percentile"])
    p.add_argument("--percentile-low", type=float, default=1.0)
    p.add_argument("--percentile-high", type=float, default=99.0)
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg_dict = checkpoint.get("config", {})
        aug_cfg = checkpoint.get("augmentation_config", {})
    else:
        state_dict = checkpoint
        cfg_dict = {}
        aug_cfg = {}

    if args.aug_config:
        with open(args.aug_config, "r", encoding="utf-8") as f:
            aug_cfg = json.load(f)
    if not aug_cfg:
        raise ValueError("No augmentation config found. Pass --aug-config or use a checkpoint containing augmentation_config.")

    image_size = int(cfg_dict.get("image_size") or deep_get(aug_cfg, ["image", "output_size"], 384))
    model_cfg = ModelConfig(
        image_size=image_size,
        backbone_name=str(cfg_dict.get("backbone_name", cfg_dict.get("backbone", "vit_small_patch8_224"))),
        backbone_output_dim=int(cfg_dict.get("backbone_output_dim", 512)),
        projection_dim=int(cfg_dict.get("projection_dim", 16)),
        projector_hidden_dim=int(cfg_dict.get("projector_hidden_dim", 2048)),
        drop_path_rate=float(cfg_dict.get("drop_path_rate", 0.1)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading split metadata...")
    df, full_num_rows = build_eval_dataframe(args)
    df = sample_dataframe(df, args)
    df = add_derived_metadata(df)
    print(f"Rows selected for PCA: {len(df):,}")
    print("Collapsed counts:", df["collapsed_birads"].value_counts().to_dict())
    for c in ["age_group", "view", "laterality", "dataset", "machine_family", "has_segmentation"]:
        if c in df.columns:
            print(f"{c} counts:", df[c].value_counts(dropna=False).head(12).to_dict())

    print("Building model...")
    model = ViTEncoder(model_cfg)
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("WARNING missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("WARNING unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    model = model.to(device)

    print("Extracting features...")
    features = extract_features(df, args, model, aug_cfg, device, full_num_rows, image_size)
    print("Feature matrix:", features.shape)

    print("Creating PCA PDF report...")
    create_pdf_report(df, features, args, Path(args.output_pdf))
    print(f"Saved PCA report to: {args.output_pdf}")


if __name__ == "__main__":
    main()
