"""Dataset identity, labels, metadata and memory-mapped mammograms."""

from __future__ import annotations
import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from .config import deep_get

CLASS_NAMES = ["routine", "follow_up", "biopsy"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    m = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})", s)
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
