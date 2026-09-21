#!/usr/bin/env python3
"""
Create a human-readable MG dataset overview HTML report plus a machine-readable JSON summary.

Patched version:
  - derives collapsed_birads from the MG CSV's `birads` actionability strings if no
    collapsed_birads column exists.
  - also derives birads_numeric from original_birads when possible.
  - includes collapsed_birads distributions, crosstables, plots, and image examples.

Usage:
  python create_mg_dataset_overview_patched.py \
    /pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg

Outputs, by default:
  <dataset_dir>/mg_dataset_overview.html
  <dataset_dir>/mg_dataset_overview.json

The script reads only the top-level CSV and BIN files and does not inspect split folders.
It overwrites only the two output report files above, unless --output-html/--output-json are set.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV_NAME = "mg-only-all.csv"
DEFAULT_BIN_NAME = "mg-only-all.bin"
CLASS_ORDER = ["routine", "follow_up", "biopsy"]


@dataclass
class BinSpec:
    dtype: str
    height: int
    width: int
    channels: int
    row_bytes: int
    expected_bytes: int
    actual_bytes: int
    exact_match: bool


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def file_info(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": int(st.st_size),
        "size_human": human_bytes(int(st.st_size)),
        "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
    }


def find_unique_file(dataset_dir: Path, preferred_name: str, suffix: str, label: str) -> Path:
    preferred = dataset_dir / preferred_name
    if preferred.exists():
        return preferred

    candidates = sorted([p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower()])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Could not find a top-level {label} file with suffix {suffix} in {dataset_dir}")
    raise RuntimeError(
        f"Found multiple top-level {label} files in {dataset_dir}; pass --csv-name/--bin-name explicitly. "
        f"Candidates: {[p.name for p in candidates]}"
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def infer_machine_family_from_machine(machine: object) -> str:
    if pd.isna(machine):
        return "unknown"
    text = str(machine).lower()

    if any(token in text for token in ["hologic", "lorad", "selenia", "dimensions", "3dimensions"]):
        return "Hologic/Lorad"
    if any(token in text for token in ["howtek", "lumisys", "lumysis"]):
        return "Howtek/Lumysis"
    if any(token in text for token in ["senographe", "ge healthcare", "general electric"]):
        return "GE/Senographe"
    if re.search(r"(^|[^a-z])ge([^a-z]|$)", text):
        return "GE/Senographe"
    return "unknown"


def normalize_view_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "MISSING", "UNKNOWN"}:
        return None

    cleaned = re.sub(r"[^A-Z0-9_ /.-]+", " ", text)
    tokens = set(re.split(r"[\s_./-]+", cleaned))

    # Keep this conservative. In this MG CSV, CC is often identifiable, while many
    # other views remain unknown because the metadata does not encode them clearly.
    for candidate in ["MLO", "CC", "LMO", "LM", "ML", "XCCL", "XCCM", "FB"]:
        if candidate in tokens or cleaned == candidate:
            return candidate

    m = re.search(r"(^|[^A-Z])(MLO|CC|XCCL|XCCM|LMO|LM|ML)([^A-Z]|$)", cleaned)
    if m:
        return m.group(2)

    return None


def infer_view_from_row(row: pd.Series) -> str:
    explicit_cols = ["view", "ViewPosition", "view_position", "viewposition", "projection", "position"]
    text_cols = [
        "id",
        "context",
        "findings",
        "exam",
        "image_path",
        "path",
        "filename",
        "file",
        "dicom_path",
        "png_path",
        "jpg_path",
        "original_path",
    ]

    for col in explicit_cols + text_cols:
        if col in row.index:
            value = normalize_view_value(row[col])
            if value is not None:
                return value

    return "unknown"


def normalize_laterality_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "MISSING", "UNKNOWN"}:
        return None

    cleaned = re.sub(r"[^A-Z0-9_ /.-]+", " ", text)
    tokens = set(re.split(r"[\s_./-]+", cleaned))

    if "LEFT" in tokens or "L" in tokens:
        return "L"
    if "RIGHT" in tokens or "R" in tokens:
        return "R"

    m = re.search(r"(^|[^A-Z])(LEFT|RIGHT|L|R)([^A-Z]|$)", cleaned)
    if m:
        return "L" if m.group(2) in {"LEFT", "L"} else "R"

    return None


def infer_laterality_from_row(row: pd.Series) -> str:
    explicit_cols = ["laterality", "Laterality", "ImageLaterality", "image_laterality"]
    text_cols = [
        "id",
        "context",
        "findings",
        "exam",
        "image_path",
        "path",
        "filename",
        "file",
        "dicom_path",
        "png_path",
        "jpg_path",
        "original_path",
    ]

    for col in explicit_cols + text_cols:
        if col in row.index:
            value = normalize_laterality_value(row[col])
            if value is not None:
                return value

    return "unknown"


def parse_birads_number(value: object) -> Optional[int]:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "missing", "unknown"}:
        return None

    # Prefer explicit original_birads values such as "(4) suspicious".
    m = re.search(r"\(([0-6])\)", text)
    if m:
        return int(m.group(1))

    # Numeric-like values.
    try:
        f = float(text)
        if np.isfinite(f):
            return int(f)
    except Exception:
        pass

    # Fallback for values such as "BI-RADS 4A".
    m = re.search(r"(^|[^0-9])([0-6])([^0-9]|$)", text)
    if m:
        return int(m.group(2))

    return None


def collapse_birads_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    raw = str(value).strip().lower()
    if raw in {"", "nan", "none", "missing", "unknown"}:
        return None

    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    if norm in {"routine", "follow_up", "biopsy"}:
        return norm

    # Native actionability strings in mg-only-all.csv:
    #   healthy/routine
    #   probably benign (follow up)
    #   suspicious/malignancy-likely (biopsy)
    # Order matters: probably benign should be follow_up, not routine.
    if "biopsy" in raw or "suspicious" in raw or "malignan" in raw:
        return "biopsy"
    if "follow" in raw or "probably benign" in raw or "probably_benign" in norm:
        return "follow_up"
    if "routine" in raw or "healthy" in raw or "negative" in raw or raw == "benign" or norm == "benign":
        return "routine"

    n = parse_birads_number(value)
    if n is None:
        return None
    if n in {1, 2}:
        return "routine"
    if n in {0, 3}:
        return "follow_up"
    if n in {4, 5, 6}:
        return "biopsy"

    return None


def derive_collapsed_birads(df: pd.DataFrame, derived: Dict[str, str], warnings: List[str]) -> pd.DataFrame:
    out = df.copy()

    if "collapsed_birads" in out.columns:
        out["collapsed_birads"] = out["collapsed_birads"].map(collapse_birads_value)
        derived["collapsed_birads"] = "standardized from existing collapsed_birads"
    else:
        # Prefer birads because this MG CSV already stores actionability labels there.
        candidate_cols = [c for c in ["birads", "original_birads", "birads_numeric"] if c in out.columns]
        if not candidate_cols:
            warnings.append("Could not derive collapsed_birads because none of birads/original_birads/birads_numeric exist.")
            return out

        chosen = None
        best_non_missing = -1
        best_values = None
        for col in candidate_cols:
            values = out[col].map(collapse_birads_value)
            non_missing = int(values.notna().sum())
            if non_missing > best_non_missing:
                chosen = col
                best_non_missing = non_missing
                best_values = values

        assert chosen is not None and best_values is not None
        out["collapsed_birads"] = best_values
        derived["collapsed_birads"] = f"derived from {chosen}"

        if best_non_missing == 0:
            warnings.append(
                "Derived collapsed_birads has zero valid rows. Check birads/original_birads value formats."
            )

    invalid = int(out["collapsed_birads"].isna().sum()) if "collapsed_birads" in out.columns else len(out)
    if invalid > 0:
        warnings.append(f"collapsed_birads is missing/invalid for {invalid} rows after derivation.")

    return out


def enrich_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    out = standardize_columns(df)
    derived: Dict[str, str] = {}
    warnings: List[str] = []

    if "machine_family" not in out.columns:
        if "machine" in out.columns:
            out["machine_family"] = out["machine"].map(infer_machine_family_from_machine)
            derived["machine_family"] = "derived from machine"
        else:
            out["machine_family"] = "unknown"
            derived["machine_family"] = "filled with unknown because machine column was absent"
            warnings.append("machine_family could not be derived from machine because machine column is absent.")

    if "view" not in out.columns:
        out["view"] = out.apply(infer_view_from_row, axis=1)
        derived["view"] = "derived from available columns such as id/context/exam/path/findings"

    if "laterality" not in out.columns:
        out["laterality"] = out.apply(infer_laterality_from_row, axis=1)
        derived["laterality"] = "derived from available columns such as id/context/exam/path/findings"

    if "birads_numeric" not in out.columns and "original_birads" in out.columns:
        out["birads_numeric"] = out["original_birads"].map(parse_birads_number)
        derived["birads_numeric"] = "derived from original_birads"

    out = derive_collapsed_birads(out, derived, warnings)

    return out, derived, warnings


def infer_bin_spec(bin_path: Path, n_rows: int) -> BinSpec:
    actual = int(bin_path.stat().st_size)
    candidates = [
        ("uint16", np.dtype("uint16"), 512, 512, 1),
        ("uint8", np.dtype("uint8"), 512, 512, 1),
        ("float32", np.dtype("float32"), 512, 512, 1),
        ("uint16", np.dtype("uint16"), 224, 224, 1),
        ("uint8", np.dtype("uint8"), 224, 224, 1),
    ]

    for dtype_name, dtype, h, w, c in candidates:
        row_bytes = int(h * w * c * dtype.itemsize)
        expected = int(n_rows * row_bytes)
        if expected == actual:
            return BinSpec(dtype_name, h, w, c, row_bytes, expected, actual, True)

    # Fallback for the known MG file.
    dtype = np.dtype("uint16")
    h, w, c = 512, 512, 1
    row_bytes = int(h * w * c * dtype.itemsize)
    expected = int(n_rows * row_bytes)
    return BinSpec("uint16", h, w, c, row_bytes, expected, actual, False)


def open_memmap(bin_path: Path, spec: BinSpec, n_rows: int) -> np.memmap:
    dtype = np.dtype(spec.dtype)
    if spec.channels == 1:
        shape = (n_rows, spec.height, spec.width)
    else:
        shape = (n_rows, spec.height, spec.width, spec.channels)
    return np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)


def top_counts(series: pd.Series, n: int = 12, include_missing: bool = True) -> Dict[str, int]:
    s = series.copy()
    if include_missing:
        s = s.fillna("missing")
    return {str(k): int(v) for k, v in s.astype(str).value_counts().head(n).to_dict().items()}


def full_counts(series: pd.Series, include_missing: bool = True) -> Dict[str, int]:
    s = series.copy()
    if include_missing:
        s = s.fillna("missing")
    return {str(k): int(v) for k, v in s.astype(str).value_counts().to_dict().items()}


def full_percent(series: pd.Series, include_missing: bool = True) -> Dict[str, float]:
    counts = full_counts(series, include_missing=include_missing)
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: round(100.0 * v / total, 3) for k, v in counts.items()}


def column_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        row: Dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": missing,
            "missing_percent": round(100.0 * missing / max(1, len(df)), 3),
            "unique": int(s.nunique(dropna=True)),
        }
        if s.dtype == object or s.nunique(dropna=True) <= 50:
            values = [str(x) for x in s.dropna().astype(str).drop_duplicates().head(8).tolist()]
            if values:
                row["examples"] = values
        rows.append(row)
    return rows


def rows_per_group_summary(df: pd.DataFrame, col: str) -> Optional[Dict[str, Any]]:
    if col not in df.columns:
        return None
    counts = df.groupby(col, dropna=False).size()
    if counts.empty:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": int(counts.min()),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "max": int(counts.max()),
    }


def plot_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_bar_plot(series: pd.Series, title: str, max_categories: int = 12) -> str:
    counts = series.fillna("missing").astype(str).value_counts().head(max_categories)
    fig_height = max(3.0, 0.42 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    return plot_to_base64(fig)


def compute_crosstable(df: pd.DataFrame, row_col: str, col_col: str, max_rows: int = 20, max_cols: int = 20) -> Optional[Dict[str, Any]]:
    if row_col not in df.columns or col_col not in df.columns:
        return None

    table = pd.crosstab(
        df[row_col].fillna("missing").astype(str),
        df[col_col].fillna("missing").astype(str),
    )

    if table.empty:
        return None

    # Keep all known class rows for collapsed_birads/original small labels, otherwise top categories.
    if table.shape[0] > max_rows:
        keep_rows = table.sum(axis=1).sort_values(ascending=False).head(max_rows).index
        table = table.loc[keep_rows]
    if table.shape[1] > max_cols:
        keep_cols = table.sum(axis=0).sort_values(ascending=False).head(max_cols).index
        table = table.loc[:, keep_cols]

    row_percent = table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    col_percent = table.div(table.sum(axis=0).replace(0, np.nan), axis=1) * 100.0

    return {
        "rows": row_col,
        "columns": col_col,
        "counts": {
            str(idx): {str(col): int(val) for col, val in row.items()}
            for idx, row in table.iterrows()
        },
        "row_percent": {
            str(idx): {str(col): round(float(val), 2) if np.isfinite(val) else 0.0 for col, val in row.items()}
            for idx, row in row_percent.iterrows()
        },
        "column_percent": {
            str(idx): {str(col): round(float(val), 2) if np.isfinite(val) else 0.0 for col, val in row.items()}
            for idx, row in col_percent.iterrows()
        },
    }


def make_heatmap(table: pd.DataFrame, title: str, value_label: str) -> str:
    fig_width = max(5.5, 1.0 * table.shape[1] + 2.0)
    fig_height = max(3.4, 0.55 * table.shape[0] + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    arr = table.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticklabels([str(x) for x in table.columns], rotation=35, ha="right")
    ax.set_yticklabels([str(x) for x in table.index])
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                if value_label == "count":
                    label = f"{int(val)}"
                else:
                    label = f"{val:.1f}%"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label=value_label)
    return plot_to_base64(fig)


def crosstable_plots(df: pd.DataFrame, row_col: str, col_col: str) -> Optional[Dict[str, str]]:
    if row_col not in df.columns or col_col not in df.columns:
        return None

    counts = pd.crosstab(
        df[row_col].fillna("missing").astype(str),
        df[col_col].fillna("missing").astype(str),
    )
    if counts.empty:
        return None

    if counts.shape[0] > 20:
        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).head(20).index]
    if counts.shape[1] > 20:
        counts = counts.loc[:, counts.sum(axis=0).sort_values(ascending=False).head(20).index]

    row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0

    return {
        "counts_heatmap": make_heatmap(counts, f"{row_col} × {col_col} counts", "count"),
        "row_percent_heatmap": make_heatmap(row_pct.fillna(0), f"{row_col} × {col_col} row %", "row %"),
    }


def image_to_base64(img: np.ndarray) -> str:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(arr, cmap="gray")
    ax.axis("off")
    return plot_to_base64(fig)


def sample_one_per_category(df: pd.DataFrame, col: str, rng: np.random.Generator, max_categories: int = 12) -> Dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].fillna("missing").astype(str).value_counts()
    if col == "collapsed_birads":
        categories = [c for c in CLASS_ORDER if c in set(counts.index)]
        categories += [c for c in counts.index.tolist() if c not in categories]
        categories = categories[:max_categories]
    else:
        categories = counts.head(max_categories).index.tolist()

    result: Dict[str, int] = {}
    values = df[col].fillna("missing").astype(str)
    for cat in categories:
        idxs = df.index[values == cat].to_numpy()
        if len(idxs) == 0:
            continue
        result[str(cat)] = int(rng.choice(idxs))
    return result


def row_metadata(row: pd.Series) -> Dict[str, Any]:
    keys = ["id", "patient", "dataset", "birads", "collapsed_birads", "original_birads", "machine", "machine_family", "view", "laterality"]
    out: Dict[str, Any] = {}
    for key in keys:
        if key in row.index:
            value = row[key]
            if pd.isna(value):
                out[key] = None
            else:
                out[key] = str(value)
    return out


def build_image_examples(
    df: pd.DataFrame,
    mmap: np.memmap,
    rng: np.random.Generator,
    random_n: int = 12,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    examples: Dict[str, List[Dict[str, Any]]] = {}
    html_sections: Dict[str, Any] = {}

    def add_example_group(group_name: str, index_map: Dict[str, int]) -> None:
        json_items: List[Dict[str, Any]] = []
        html_items: List[Dict[str, Any]] = []
        for label, idx in index_map.items():
            row = df.loc[idx]
            meta = row_metadata(row)
            json_items.append(meta)
            img = mmap[int(idx)]
            html_items.append({"label": label, "meta": meta, "image_b64": image_to_base64(img)})
        examples[group_name] = json_items
        html_sections[group_name] = html_items

    if len(df) > 0:
        random_count = min(random_n, len(df))
        random_idxs = rng.choice(df.index.to_numpy(), size=random_count, replace=False)
        add_example_group("random_examples", {f"random_{i+1}": int(idx) for i, idx in enumerate(random_idxs)})

    for col, group_name, max_cats in [
        ("collapsed_birads", "examples_by_collapsed_birads", 8),
        ("original_birads", "examples_by_original_birads", 8),
        ("view", "examples_by_view", 8),
        ("machine_family", "examples_by_machine_family", 8),
        ("dataset", "examples_by_top_datasets", 5),
    ]:
        index_map = sample_one_per_category(df, col, rng, max_categories=max_cats)
        if index_map:
            add_example_group(group_name, index_map)

    return examples, html_sections


def sample_image_intensity_summary(mmap: np.memmap, n_rows: int, rng: np.random.Generator, sample_n: int = 512) -> Dict[str, Any]:
    if n_rows == 0:
        return {}
    n = min(sample_n, n_rows)
    idxs = rng.choice(np.arange(n_rows), size=n, replace=False)
    means = []
    stds = []
    p1s = []
    p99s = []
    near_empty = 0
    for idx in idxs:
        arr = np.asarray(mmap[int(idx)]).astype(np.float32)
        means.append(float(arr.mean()))
        std = float(arr.std())
        stds.append(std)
        p1s.append(float(np.percentile(arr, 1)))
        p99s.append(float(np.percentile(arr, 99)))
        if std < 1e-6:
            near_empty += 1
    return {
        "sampled_images": int(n),
        "near_empty_or_constant_in_sample": int(near_empty),
        "mean_intensity_median": float(np.median(means)),
        "mean_intensity_p05": float(np.percentile(means, 5)),
        "mean_intensity_p95": float(np.percentile(means, 95)),
        "std_intensity_median": float(np.median(stds)),
        "std_intensity_p05": float(np.percentile(stds, 5)),
        "std_intensity_p95": float(np.percentile(stds, 95)),
        "p1_intensity_median": float(np.median(p1s)),
        "p99_intensity_median": float(np.median(p99s)),
    }


def dataframe_to_html_table(df: pd.DataFrame, classes: str = "data-table compact", escape: bool = True) -> str:
    return df.to_html(index=False, classes=classes, escape=escape, border=0)


def dict_table_html(d: Dict[str, Any], key_name: str = "Field", value_name: str = "Value") -> str:
    rows = [{key_name: k, value_name: json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in d.items()]
    return dataframe_to_html_table(pd.DataFrame(rows))


def counts_table_html(counts: Dict[str, int]) -> str:
    df = pd.DataFrame([{"value": k, "count": v} for k, v in counts.items()])
    total = int(df["count"].sum()) if not df.empty else 0
    if total > 0:
        df["percent"] = df["count"].apply(lambda x: round(100.0 * x / total, 3))
    return dataframe_to_html_table(df)


def crosstable_html(ct: Dict[str, Any], mode: str = "counts") -> str:
    data = ct[mode]
    table = pd.DataFrame.from_dict(data, orient="index")
    table.index.name = ct["rows"]
    table = table.reset_index()
    return table.to_html(index=False, classes="data-table compact", border=0)


def build_html(
    dataset_dir: Path,
    csv_info: Dict[str, Any],
    bin_info: Dict[str, Any],
    bin_spec: BinSpec,
    overview: Dict[str, Any],
    col_summary: List[Dict[str, Any]],
    distributions: Dict[str, Dict[str, Any]],
    distribution_plots: Dict[str, str],
    crosstables: Dict[str, Dict[str, Any]],
    crosstable_plot_images: Dict[str, Dict[str, str]],
    image_sections: Dict[str, Any],
    warnings: List[str],
    derived_columns: Dict[str, str],
) -> str:
    now = datetime.now(timezone.utc).isoformat()

    important_cols = [
        "collapsed_birads",
        "birads",
        "original_birads",
        "birads_numeric",
        "dataset",
        "machine_family",
        "machine",
        "view",
        "laterality",
        "patient",
        "exam",
        "segmentation",
        "modality",
        "race",
    ]
    important_rows = []
    for col in important_cols:
        if col not in distributions and col not in [r["column"] for r in col_summary]:
            continue
        match = next((r for r in col_summary if r["column"] == col), None)
        if match is None:
            continue
        top = distributions.get(col, {}).get("counts", top_counts(pd.Series(dtype=object)))
        important_rows.append(
            {
                "column": col,
                "missing": match["missing"],
                "missing %": f"{match['missing_percent']:.2f}",
                "unique": match["unique"],
                "top values": ", ".join([f"{k}: {v}" for k, v in list(top.items())[:6]]),
            }
        )

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<title>MG Dataset Overview</title>
<style>
  body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.35; color: #222; max-width: 1500px; }
  h1, h2, h3, h4 { margin-bottom: 0.35em; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  section { margin-top: 28px; }
  .small-note { color: #555; font-size: 0.92em; }
  .warning { color: #8a4b00; font-weight: 600; }
  .warning-box { border: 1px solid #d9a441; background: #fff8e8; padding: 12px 16px; margin: 16px 0; }
  .data-table { border-collapse: collapse; margin: 8px 0 16px 0; font-size: 0.9em; }
  .data-table th, .data-table td { border: 1px solid #bbb; padding: 5px 7px; vertical-align: top; }
  .data-table th { background: #eee; }
  .compact { font-size: 0.86em; }
  .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
  .plot-card { border: 1px solid #ddd; padding: 10px; }
  .plot-img { max-width: 100%; height: auto; }
  .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; margin: 8px 0 18px 0; }
  .image-card { border: 1px solid #ccc; padding: 8px; background: #fafafa; }
  .image-card img { width: 100%; height: auto; display: block; }
  .caption { font-size: 0.76em; color: #333; margin-top: 6px; word-break: break-word; }
  code { background: #f3f3f3; padding: 1px 4px; }
</style>
</head>
<body>
""")

    html_parts.append("<h1>MG Dataset Overview</h1>\n")
    html_parts.append(
        f"<p class='small-note'>Created at {html.escape(now)}. This report uses only the top-level CSV and BIN files in "
        f"<code>{html.escape(str(dataset_dir))}</code>. Train/validation/test split folders are intentionally not inspected here.</p>\n"
    )

    if warnings:
        html_parts.append("<div class='warning-box'><h3>Warnings / takeaways</h3><ul>")
        for w in warnings:
            html_parts.append(f"<li class='warning'>{html.escape(str(w))}</li>")
        html_parts.append("</ul></div>")

    if derived_columns:
        html_parts.append("<h3>Derived columns</h3>")
        html_parts.append(dict_table_html(derived_columns, "column", "derivation"))

    html_parts.append("<section><h2>1. Dataset overview</h2>")
    html_parts.append("<h3>Dataset facts</h3>")
    html_parts.append(dict_table_html(overview))

    html_parts.append("<h3>Source files</h3>")
    html_parts.append(
        dataframe_to_html_table(
            pd.DataFrame(
                [
                    {"file": "CSV", **csv_info},
                    {"file": "BIN", **bin_info},
                ]
            )[["file", "name", "size_human", "modified_utc", "path"]]
        )
    )

    html_parts.append("<h3>BIN specification</h3>")
    html_parts.append(dict_table_html(asdict(bin_spec)))

    html_parts.append("<h3>Important columns</h3>")
    html_parts.append(dataframe_to_html_table(pd.DataFrame(important_rows), escape=True))

    html_parts.append("<h3>Important distributions</h3><div class='plot-grid'>")
    for col, img_b64 in distribution_plots.items():
        html_parts.append(
            f"<div class='plot-card'><h3>{html.escape(col)}</h3>"
            f"<img class='plot-img' src='data:image/png;base64,{img_b64}'/>"
            f"{counts_table_html(distributions[col]['counts'])}</div>"
        )
    html_parts.append("</div></section>")

    html_parts.append("<section><h2>2. Crosstables and confounding checks</h2>")
    for name, ct in crosstables.items():
        html_parts.append(f"<h3>{html.escape(ct['rows'])} × {html.escape(ct['columns'])}</h3>")
        html_parts.append("<h4>Counts</h4>")
        html_parts.append(crosstable_html(ct, "counts"))
        html_parts.append("<h4>Row percentages</h4>")
        html_parts.append(crosstable_html(ct, "row_percent"))
        plots = crosstable_plot_images.get(name)
        if plots:
            html_parts.append("<div class='plot-grid'>")
            html_parts.append(
                f"<div class='plot-card'><img class='plot-img' src='data:image/png;base64,{plots['counts_heatmap']}'/></div>"
            )
            html_parts.append(
                f"<div class='plot-card'><img class='plot-img' src='data:image/png;base64,{plots['row_percent_heatmap']}'/></div>"
            )
            html_parts.append("</div>")
    html_parts.append("</section>")

    html_parts.append("<section><h2>3. Representative images</h2>")
    for group_name, items in image_sections.items():
        html_parts.append(f"<h3>{html.escape(group_name)}</h3><div class='image-grid'>")
        for item in items:
            meta = item["meta"]
            caption = "<br/>".join([f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}" for k, v in meta.items()])
            label = html.escape(str(item.get("label", "")))
            html_parts.append(
                f"<div class='image-card'><div><b>{label}</b></div>"
                f"<img src='data:image/png;base64,{item['image_b64']}'/>"
                f"<div class='caption'>{caption}</div></div>"
            )
        html_parts.append("</div>")
    html_parts.append("</section>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MG dataset overview HTML + JSON reports.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--csv-name", type=str, default=DEFAULT_CSV_NAME)
    parser.add_argument("--bin-name", type=str, default=DEFAULT_BIN_NAME)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-images", action="store_true", help="Skip embedded image examples for faster/smaller reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    csv_path = find_unique_file(dataset_dir, args.csv_name, ".csv", "CSV")
    bin_path = find_unique_file(dataset_dir, args.bin_name, ".bin", "BIN")
    output_html = args.output_html or dataset_dir / "mg_dataset_overview.html"
    output_json = args.output_json or dataset_dir / "mg_dataset_overview.json"

    print(f"Reading CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"Rows: {len(df_raw):,}; columns: {len(df_raw.columns):,}")

    df, derived_columns, warnings = enrich_dataframe(df_raw)

    print("Derived columns:", derived_columns)
    if "collapsed_birads" in df.columns:
        print("collapsed_birads counts:", df["collapsed_birads"].fillna("missing").astype(str).value_counts().to_dict())
    if "machine_family" in df.columns:
        print("machine_family counts:", df["machine_family"].fillna("missing").astype(str).value_counts().to_dict())
    if "view" in df.columns:
        print("view counts:", df["view"].fillna("missing").astype(str).value_counts().to_dict())

    bin_spec = infer_bin_spec(bin_path, len(df_raw))
    if not bin_spec.exact_match:
        warnings.append(
            "BIN size did not exactly match known candidates; using fallback 512x512 uint16. "
            "Check image shape/dtype if this is unexpected."
        )
    mmap = open_memmap(bin_path, bin_spec, len(df_raw))

    overview: Dict[str, Any] = {
        "rows": int(len(df_raw)),
        "columns": int(len(df_raw.columns)),
        "csv_size": human_bytes(csv_path.stat().st_size),
        "bin_size": human_bytes(bin_path.stat().st_size),
        "image_shape": f"{bin_spec.height}x{bin_spec.width}",
        "image_dtype": bin_spec.dtype,
        "csv_bin_consistent": bool(bin_spec.exact_match),
    }
    if "patient" in df.columns:
        overview["unique_patients"] = int(df["patient"].nunique(dropna=True))
        overview["rows_per_patient"] = rows_per_group_summary(df, "patient")
    if "exam" in df.columns:
        overview["unique_exams"] = int(df["exam"].nunique(dropna=True))
        overview["rows_per_exam"] = rows_per_group_summary(df, "exam")

    # Warning heuristics.
    if "machine_family" in df.columns:
        mf_pct = full_percent(df["machine_family"])
        if mf_pct:
            top_mf, top_pct = max(mf_pct.items(), key=lambda kv: kv[1])
            if top_pct > 70:
                warnings.append(f"machine_family is imbalanced: '{top_mf}' accounts for {top_pct:.1f}% of rows.")
    if "view" in df.columns:
        view_counts = full_counts(df["view"])
        unknown = view_counts.get("unknown", 0) + view_counts.get("missing", 0)
        if unknown > 0:
            warnings.append(f"view is unknown or could not be inferred for {100.0 * unknown / max(1, len(df)):.1f}% of rows.")
    if "collapsed_birads" in df.columns:
        cb = full_counts(df["collapsed_birads"])
        if cb:
            top_cls, top_count = max(cb.items(), key=lambda kv: kv[1])
            warnings.append(f"collapsed_birads distribution: most common class '{top_cls}' has {top_count:,} rows.")

    dist_cols = [
        "collapsed_birads",
        "birads",
        "original_birads",
        "birads_numeric",
        "dataset",
        "machine_family",
        "machine",
        "view",
        "laterality",
        "race",
    ]
    distributions: Dict[str, Dict[str, Any]] = {}
    distribution_plots: Dict[str, str] = {}
    for col in dist_cols:
        if col in df.columns:
            distributions[col] = {
                "counts": full_counts(df[col]),
                "percent": full_percent(df[col]),
            }
            distribution_plots[col] = make_bar_plot(df[col], col, max_categories=12)

    crosstable_pairs = [
        ("collapsed_birads", "machine_family"),
        ("collapsed_birads", "dataset"),
        ("collapsed_birads", "view"),
        ("collapsed_birads", "laterality"),
        ("collapsed_birads", "original_birads"),
        ("original_birads", "machine_family"),
        ("original_birads", "dataset"),
        ("dataset", "machine_family"),
        ("dataset", "view"),
        ("machine_family", "view"),
    ]
    crosstables: Dict[str, Dict[str, Any]] = {}
    crosstable_plot_images: Dict[str, Dict[str, str]] = {}
    for row_col, col_col in crosstable_pairs:
        ct = compute_crosstable(df, row_col, col_col)
        if ct is not None:
            name = f"{row_col}__{col_col}"
            crosstables[name] = ct
            plots = crosstable_plots(df, row_col, col_col)
            if plots is not None:
                crosstable_plot_images[name] = plots

    rng = np.random.default_rng(args.seed)
    sampled_image_intensity = sample_image_intensity_summary(mmap, len(df_raw), rng)

    image_examples: Dict[str, Any] = {}
    image_sections: Dict[str, Any] = {}
    if not args.no_images:
        image_examples, image_sections = build_image_examples(df, mmap, rng)

    report_json: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "csv_file": file_info(csv_path),
        "bin_file": file_info(bin_path),
        "source_columns": [str(c) for c in df_raw.columns],
        "derived_columns": derived_columns,
        "overview": overview,
        "row_count": int(len(df_raw)),
        "column_count": int(len(df_raw.columns)),
        "column_summary": column_summary(df),
        "bin_spec": asdict(bin_spec),
        "distributions": distributions,
        "crosstables": crosstables,
        "image_examples": image_examples,
        "sampled_image_intensity_summary": sampled_image_intensity,
        "warnings": warnings,
    }

    html_report = build_html(
        dataset_dir=dataset_dir,
        csv_info=file_info(csv_path),
        bin_info=file_info(bin_path),
        bin_spec=bin_spec,
        overview=overview,
        col_summary=column_summary(df),
        distributions=distributions,
        distribution_plots=distribution_plots,
        crosstables=crosstables,
        crosstable_plot_images=crosstable_plot_images,
        image_sections=image_sections,
        warnings=warnings,
        derived_columns=derived_columns,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8")
    output_html.write_text(html_report, encoding="utf-8")

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote HTML: {output_html}")


if __name__ == "__main__":
    main()
