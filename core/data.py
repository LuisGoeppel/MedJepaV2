"""Dataset identity, labels, metadata and memory-mapped mammograms."""

from __future__ import annotations
import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Optional, Dict, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from .config import deep_get

CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def set_seed(seed: int, *, benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if benchmark:
        torch.backends.cudnn.benchmark = True


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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_labels(df: pd.DataFrame, prefer_collapsed: bool = False) -> pd.DataFrame:
    """Keep transfer's existing collapsed-label preference explicit."""
    df = df.copy()
    label_col = "original_birads" if "original_birads" in df else "birads"
    if label_col in df:
        df["birads_numeric"] = df[label_col].apply(normalize_birads_value)
    else:
        df["birads_numeric"] = np.nan
    use_collapsed = (
        prefer_collapsed
        and "collapsed_birads" in df
        and set(df["collapsed_birads"].dropna().astype(str)).issubset(CLASS_NAMES)
    )
    if use_collapsed:
        df = df[df["collapsed_birads"].isin(CLASS_NAMES)].copy()
    else:
        if label_col not in df:
            raise ValueError("CSV requires original_birads or birads")
        df = df[df["birads_numeric"].isin([1, 2, 3, 4, 5])].copy()
        df["birads_numeric"] = df["birads_numeric"].astype(int)
        df["collapsed_birads"] = df["birads_numeric"].apply(collapse_birads_numeric)
    df["target_collapsed"] = df["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
    return df


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value) or not str(value).strip():
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(str(value))
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError, TypeError):
            pass
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
    m = re.search(r"(\d{2,3})\s*[-â€“]\s*(\d{2,3})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"\d{2,3}", s)
    if m:
        age = int(m.group(0))
        lo = (age // 10) * 10
        return f"{lo}-{lo + 9}"
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

    for column in ("dataset", "machine"):
        if column not in df:
            df[column] = "unknown"
        df[column] = df[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    return df


def ensure_original_index(split_df: pd.DataFrame, full_df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = split_df.copy()
    if "original_index" not in df:
        if "id" not in df or "id" not in full_df_raw:
            raise ValueError(f"{split_name}: original_index or a unique id is required")
        ids = full_df_raw["id"].astype(str)
        if ids.duplicated().any():
            raise ValueError(
                f"{split_name}: full CSV has duplicate ids; supply original_index to avoid ambiguous image matching"
            )
        mapping = pd.Series(np.arange(len(full_df_raw)), index=ids)
        df["original_index"] = df["id"].astype(str).map(mapping)
    index = pd.to_numeric(df["original_index"], errors="coerce")
    valid = index.notna() & (index >= 0) & (index < len(full_df_raw)) & (index == np.floor(index))
    if not valid.all():
        raise ValueError(f"{split_name}: invalid or unmapped original_index in {int((~valid).sum())} rows")
    df["original_index"] = index.astype(np.int64)
    if df["original_index"].duplicated().any():
        raise ValueError(f"{split_name}: duplicate image rows")
    return df


def verify_no_patient_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    for name, frame in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        if "patient" not in frame or frame["patient"].isna().any():
            raise ValueError(f"{name}: patient identifiers are required to verify split independence")
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


def load_splits(
    full_csv: str, train_csv: str, val_csv: str, test_csv: str, *, prefer_collapsed: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """First result is the RAW CSV: its row count describes the physical BIN."""
    full_raw = read_csv_clean(full_csv)
    splits = []
    for name, path in zip(("train", "val", "test"), (train_csv, val_csv, test_csv)):
        frame = ensure_original_index(read_csv_clean(path), full_raw, name)
        frame = add_derived_metadata(prepare_labels(frame, prefer_collapsed=prefer_collapsed))
        frame["split"] = name
        splits.append(frame.reset_index(drop=True))
    verify_no_patient_leakage(*splits)
    indices = [set(frame["original_index"]) for frame in splits]
    if any(indices[i] & indices[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Image overlap between train/val/test splits")
    return full_raw, *splits


def validate_bin(bin_path: str | Path, n_rows: int, image_height: int, image_width: int, dtype: str) -> None:
    expected = n_rows * image_height * image_width * np.dtype(dtype).itemsize
    actual = Path(bin_path).stat().st_size
    if expected != actual:
        raise ValueError(f"Full CSV and BIN do not match: expected {expected} bytes, got {actual}")


class MammographyDataset(Dataset):
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
        percentile_fallback: bool = False,
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
        self.percentile_fallback = percentile_fallback
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
            if self.percentile_fallback:
                # Legacy sweeps used scalar percentile calls. NumPy can retain
                # float32 here while the vector call below promotes to float64.
                lo = np.percentile(arr, self.percentile_low)
                hi = np.percentile(arr, self.percentile_high)
            else:
                lo, hi = np.percentile(arr, [self.percentile_low, self.percentile_high])
            if hi <= lo and self.percentile_fallback:
                lo, hi = float(arr.min()), float(arr.max())
            arr = np.zeros_like(arr, dtype=np.float32) if hi <= lo else np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        return torch.from_numpy(arr).unsqueeze(0).float().clamp(0, 1)

    def image_at(self, idx: int) -> torch.Tensor:
        return self._load_tensor(int(self.df.iloc[idx]["original_index"]))


class MGSupDataset(MammographyDataset):
    def __getitem__(self, idx: int):
        return self.transform(self.image_at(idx)), torch.tensor(
            int(self.df.iloc[idx]["target_collapsed"]), dtype=torch.long
        )


class MedJEPADataset(MammographyDataset):
    def __init__(
        self,
        df,
        bin_path,
        full_num_rows,
        image_shape,
        dtype,
        transform,
        num_views,
        normalize_mode,
        percentile_low,
        percentile_high,
    ):
        super().__init__(
            df, bin_path, full_num_rows, image_shape, dtype, transform, normalize_mode, percentile_low, percentile_high
        )
        self.num_views = int(num_views)

    def __getitem__(self, idx: int):
        image = self.image_at(idx)
        views = torch.stack([self.transform(image.clone()) for _ in range(self.num_views)])
        return views, int(self.df.iloc[idx]["target_collapsed"])


class EvalDataset(MammographyDataset):
    def __getitem__(self, idx: int):
        return self.transform(self.image_at(idx)).unsqueeze(0), idx


def collate_batch(batch):
    views = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return views, labels


@dataclass
class BinSpec:
    dtype: str
    height: int
    width: int
    channels: int
    exact_match: bool


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
        if n_rows * h * w * c * dtype.itemsize == actual:
            return BinSpec(dtype=dtype_name, height=h, width=w, channels=c, exact_match=True)
    raise ValueError(f"BIN size {actual} does not match a supported shape/dtype for {n_rows} raw CSV rows.")


def open_memmap(bin_path: Path, spec: BinSpec, n_rows: int) -> np.memmap:
    dtype = np.dtype(spec.dtype)
    shape = (
        (n_rows, spec.height, spec.width) if spec.channels == 1 else (n_rows, spec.height, spec.width, spec.channels)
    )
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)


def summarize_df(df: pd.DataFrame, columns=("collapsed_birads", "machine_family", "view", "dataset")) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"rows": int(len(df))}
    if "patient" in df.columns:
        summary["patients"] = int(df["patient"].nunique(dropna=True))
    for col in columns:
        if col in df.columns:
            summary[f"{col}_counts"] = {
                str(k): int(v) for k, v in df[col].fillna("missing").astype(str).value_counts().to_dict().items()
            }
    return summary


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


def infer_view_from_row(row: pd.Series, include_exam: bool = True) -> str:
    cols = [
        "view",
        "ViewPosition",
        "view_position",
        "viewposition",
        "projection",
        "position",
        "id",
        "context",
        "findings",
        "image_path",
        "path",
        "filename",
        "file",
        "dicom_path",
        "png_path",
        "jpg_path",
        "original_path",
        "exam",
    ]
    for col in cols if include_exam else [c for c in cols if c != "exam"]:
        if col in row.index:
            v = normalize_view(row[col])
            if v is not None:
                return v
    return "unknown"


def parse_birads_number(value: object, policy: str = "resolution") -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "missing", "unknown"}:
        return None
    if policy == "data_ablation":
        try:
            numeric = float(text)
            if np.isfinite(numeric):
                return int(numeric)
        except ValueError:
            pass
        match = re.search(r"([0-6])", text)
        return int(match.group(1)) if match else None
    m = re.search(r"\(([0-6])\)", text)
    if m:
        return int(m.group(1))
    try:
        f = float(text)
        if np.isfinite(f):
            return int(f)
    except Exception:
        pass
    m = re.search(r"(^|[^0-9])([0-6])([^0-9]|$)", text)
    if m:
        return int(m.group(2))
    return None


def collapse_birads_value(value: object, policy: str = "resolution") -> Optional[str]:
    if pd.isna(value):
        return None
    raw = str(value).strip().lower()
    if raw in {"", "nan", "none", "missing", "unknown"}:
        return None
    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if norm in CLASS_TO_INDEX:
        return norm

    # Native MG actionability strings.
    # Order matters: "probably benign" should be follow_up, not routine.
    if "biopsy" in raw or "suspicious" in raw or "malignan" in raw:
        return "biopsy"
    if "follow" in raw or "probably benign" in raw or "probably_benign" in norm:
        return "follow_up"
    if "routine" in raw or "healthy" in raw or "negative" in raw or raw == "benign" or norm == "benign":
        return "routine"

    n = parse_birads_number(value, policy=policy)
    if n is None:
        return None
    if n in {1, 2}:
        return "routine"
    if n in {0, 3}:
        return "follow_up"
    if n in {4, 5, 6}:
        return "biopsy"
    return None


def mode_or_first(s: pd.Series) -> str:
    mode = s.mode()
    return str(mode.iloc[0]) if len(mode) > 0 else str(s.iloc[0])


def make_patient_disjoint_pools(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if "patient" not in df or df["patient"].isna().any():
        raise ValueError("Patient identifiers are required for supervised splits.")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0) or min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive and sum to one.")

    groups = (
        df.groupby("patient", dropna=False)
        .agg(n_rows=("patient", "size"), label=("collapsed_birads", mode_or_first))
        .reset_index()
    )
    try:
        train_groups, tmp_groups = train_test_split(
            groups, train_size=train_ratio, random_state=seed, shuffle=True, stratify=groups["label"]
        )
    except ValueError:
        train_groups, tmp_groups = train_test_split(
            groups, train_size=train_ratio, random_state=seed, shuffle=True, stratify=None
        )
    val_fraction = val_ratio / (val_ratio + test_ratio)
    try:
        val_groups, test_groups = train_test_split(
            tmp_groups, train_size=val_fraction, random_state=seed + 1, shuffle=True, stratify=tmp_groups["label"]
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            tmp_groups, train_size=val_fraction, random_state=seed + 1, shuffle=True, stratify=None
        )

    train_ids = set(train_groups["patient"])
    val_ids = set(val_groups["patient"])
    test_ids = set(test_groups["patient"])
    train_df = df[df["patient"].isin(train_ids)].copy()
    val_df = df[df["patient"].isin(val_ids)].copy()
    test_df = df[df["patient"].isin(test_ids)].copy()
    train_pat = set(train_df["patient"].dropna().astype(str))
    val_pat = set(val_df["patient"].dropna().astype(str))
    test_pat = set(test_df["patient"].dropna().astype(str))
    info = {
        "split_type": "patient_disjoint",
        "patient_overlap": {
            "train_vs_val": len(train_pat & val_pat),
            "train_vs_test": len(train_pat & test_pat),
            "val_vs_test": len(val_pat & test_pat),
        },
    }
    verify_no_patient_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df, info


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


def prepare_baseline_metadata(df, *, policy):
    """Preserve the two sweeps' column precedence and numeric-string parsing."""
    if policy not in {"data_ablation", "resolution"}:
        raise ValueError(f"Unknown baseline label policy: {policy}")
    out = standardize_columns(df) if policy == "data_ablation" else df.rename(columns=lambda c: str(c).strip()).copy()
    out["source_index"] = np.arange(len(out), dtype=np.int64)
    candidates = (
        ["birads_numeric", "birads", "original_birads"]
        if policy == "data_ablation"
        else ["birads", "original_birads", "birads_numeric"]
    )
    column = next((c for c in ["collapsed_birads", *candidates] if c in out), None)
    if column is None:
        raise ValueError("No BI-RADS label column found.")
    out["collapsed_birads"] = out[column].map(lambda value: collapse_birads_value(value, policy=policy))
    out = out[out["collapsed_birads"].isin(CLASS_NAMES)].copy()
    if out.empty:
        raise ValueError("No valid supervised labels remain.")
    out["target"] = out["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
    if "machine_family" not in out:
        out["machine_family"] = out["machine"].map(infer_machine_family) if "machine" in out else "unknown"
    if "view" not in out:
        out["view"] = out.apply(lambda row: infer_view_from_row(row, include_exam=policy == "resolution"), axis=1)
    return out


def parse_birads_numeric(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    for k in ["1", "2", "3", "4", "5", "6"]:
        if s == k or s.startswith(k + ".") or s.startswith(k + " ") or f"({k})" in s:
            return int(k)
    try:
        val = int(float(s))
        return val if val in {1, 2, 3, 4, 5, 6} else None
    except Exception:
        return None


def collapse_label_from_row(row: pd.Series) -> str:
    # Prefer already-derived collapsed_birads when present.
    if "collapsed_birads" in row and not pd.isna(row["collapsed_birads"]):
        s = str(row["collapsed_birads"]).strip().lower()
        s = s.replace("-", "_").replace(" ", "_")
        if s in CLASS_TO_INDEX:
            return s

    # Then support the actionability strings used in mg-only-all.csv.
    if "birads" in row and not pd.isna(row["birads"]):
        s = str(row["birads"]).strip().lower()
        if "suspicious" in s or "malignan" in s or "biopsy" in s:
            return "biopsy"
        if "follow" in s or "probably benign" in s:
            return "follow_up"
        if "routine" in s or "healthy" in s:
            return "routine"

    # Fallback to original_birads / numeric birads.
    for col in ["original_birads", "birads_numeric", "birads"]:
        if col in row:
            n = parse_birads_numeric(row[col])
            if n in (1, 2):
                return "routine"
            if n == 3:
                return "follow_up"
            if n in (4, 5, 6):
                return "biopsy"
    return "unknown"


def prepare_baseline_split(df, full_df, split_name):
    df = df.copy()
    df["collapsed_birads"] = df.apply(collapse_label_from_row, axis=1)
    df = df[df["collapsed_birads"].isin(CLASS_NAMES)].copy()
    df["target"] = df["collapsed_birads"].map(CLASS_TO_INDEX).astype(int)
    # Older standalone runs used exam first; core validates the resulting indices.
    if "original_index" not in df and "exam" in df and "exam" in full_df:
        keys = full_df["exam"].astype(str)
        if keys.duplicated().any():
            raise ValueError("full_csv exam column is not unique; supply original_index.")
        df["original_index"] = df["exam"].astype(str).map(pd.Series(np.arange(len(full_df)), index=keys))
    return ensure_original_index(df, full_df, split_name).reset_index(drop=True)


def validate_split_frames(train, val, test):
    verify_no_patient_leakage(train, val, test)
    indices = [set(df["original_index"]) for df in (train, val, test)]
    if any(indices[a] & indices[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Image overlap between supervised splits")


class BaselineDataset(MammographyDataset):
    """Adapt baseline column names to the common memmap reader."""

    def __init__(
        self,
        df,
        bin_path,
        full_num_rows,
        image_shape,
        dtype,
        image_size,
        train,
        normalize_mode="uint16",
        percentile_low=1.0,
        percentile_high=99.0,
        hflip_p=0.5,
        interpolation="bilinear",
        percentile_fallback=False,
    ):
        from .transforms import BaselineTransform

        frame = df.rename(columns={"source_index": "original_index"}) if "original_index" not in df else df
        super().__init__(
            frame,
            bin_path,
            full_num_rows,
            image_shape,
            dtype,
            BaselineTransform(image_size, hflip_p if train else 0.0, interpolation),
            normalize_mode,
            percentile_low,
            percentile_high,
            percentile_fallback,
        )

    def __getitem__(self, idx):
        return self.transform(self.image_at(idx)), torch.tensor(int(self.df.iloc[idx]["target"]), dtype=torch.long)


def baseline_dataset_from_memmap(df, mmap, image_size, train, augment):
    return BaselineDataset(
        df,
        mmap.filename,
        mmap.shape[0],
        mmap.shape[1:3],
        mmap.dtype,
        image_size,
        train,
        "per_image_percentile",
        hflip_p=0.5 if augment else 0.0,
        interpolation="area",
        percentile_fallback=True,
    )
