#!/usr/bin/env python3
"""
Create machine-family-specific MG train/val/test CSV splits.

Default use case:
  - Keep only rows from machine_family == "Hologic/Lorad"
  - Optionally also keep only one view, e.g. CC or MLO
  - Write train/val/test CSVs plus a JSON summary

The script supports two split modes:

1) filter_existing (default)
   Filters existing train/val/test CSVs.
   This preserves the original patient-disjoint split and makes results easier to compare
   against previous full-dataset experiments.

2) resplit
   Creates a new patient-disjoint split from the filtered full CSV.
   This is useful if you want fresh 80/10/10 splits inside the Hologic/Lorad subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


from core.config import slugify, save_json
from core.data import infer_machine_family as infer_machine_family_from_machine, summarize_df

DEFAULT_MG_DIR = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/breastTumor/mg")
DEFAULT_FULL_CSV = DEFAULT_MG_DIR / "mg-only-all.csv"
DEFAULT_TRAIN_CSV = DEFAULT_MG_DIR / "splits" / "mg_train.csv"
DEFAULT_VAL_CSV = DEFAULT_MG_DIR / "splits" / "mg_val.csv"
DEFAULT_TEST_CSV = DEFAULT_MG_DIR / "splits" / "mg_test.csv"


def ensure_machine_family_column(df: pd.DataFrame) -> pd.DataFrame:
    if "machine_family" in df.columns:
        return df

    if "machine" not in df.columns:
        raise ValueError("CSV has neither 'machine_family' nor 'machine'. Cannot filter by machine family.")

    df = df.copy()
    df["machine_family"] = df["machine"].map(infer_machine_family_from_machine)
    return df


def normalize_series_for_matching(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold()


def filter_df(
    df: pd.DataFrame,
    machine_family: str,
    view: Optional[str] = None,
) -> pd.DataFrame:
    df = ensure_machine_family_column(df)

    family_mask = normalize_series_for_matching(df["machine_family"]) == machine_family.strip().casefold()
    out = df.loc[family_mask].copy()

    if view is not None:
        if "view" not in out.columns:
            raise ValueError("Requested --view filter, but CSV has no 'view' column.")
        view_mask = normalize_series_for_matching(out["view"]) == view.strip().casefold()
        out = out.loc[view_mask].copy()

    return out


def mode_or_first(values: pd.Series) -> object:
    values = values.dropna()
    if values.empty:
        return "unknown"
    mode = values.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return values.iloc[0]


def stratified_patient_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_col: str,
    stratify_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a patient-disjoint split, stratifying approximately by patient-level majority class."""
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in CSV.")
    if stratify_col not in df.columns:
        raise ValueError(f"Stratification column '{stratify_col}' not found in CSV.")

    if df[group_col].isna().any():
        raise ValueError("Missing group identifiers prevent a patient-disjoint split.")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive.")
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    ratios = ratios / ratios.sum()
    train_ratio, val_ratio, test_ratio = ratios.tolist()

    groups = (
        df.groupby(group_col, dropna=False)
        .agg(
            n_rows=(group_col, "size"),
            stratify_label=(stratify_col, mode_or_first),
        )
        .reset_index()
    )

    if len(groups) < 3:
        raise ValueError(f"Too few unique groups for train/val/test split: {len(groups)}")

    try:
        from sklearn.model_selection import train_test_split

        train_groups, temp_groups = train_test_split(
            groups,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=groups["stratify_label"],
        )

        temp_val_fraction = val_ratio / (val_ratio + test_ratio)

        try:
            val_groups, test_groups = train_test_split(
                temp_groups,
                train_size=temp_val_fraction,
                random_state=seed + 1,
                shuffle=True,
                stratify=temp_groups["stratify_label"],
            )
        except ValueError:
            val_groups, test_groups = train_test_split(
                temp_groups,
                train_size=temp_val_fraction,
                random_state=seed + 1,
                shuffle=True,
                stratify=None,
            )

    except Exception as exc:
        print(f"[WARN] Stratified patient split failed ({type(exc).__name__}: {exc}).")
        print("[WARN] Falling back to shuffled patient split without stratification.")

        shuffled = groups.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))

        train_groups = shuffled.iloc[:n_train]
        val_groups = shuffled.iloc[n_train : n_train + n_val]
        test_groups = shuffled.iloc[n_train + n_val :]

    train_ids = set(train_groups[group_col])
    val_ids = set(val_groups[group_col])
    test_ids = set(test_groups[group_col])

    train_df = df[df[group_col].isin(train_ids)].copy()
    val_df = df[df[group_col].isin(val_ids)].copy()
    test_df = df[df[group_col].isin(test_ids)].copy()

    return train_df, val_df, test_df


def patient_overlap_report(splits: Dict[str, pd.DataFrame], group_col: str = "patient") -> Dict[str, int]:
    if not splits:
        return {}
    first = next(iter(splits.values()))
    if group_col not in first.columns:
        return {}

    groups = {name: set(df[group_col].dropna().astype(str)) for name, df in splits.items()}
    names = list(groups)
    report = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            report[f"{a}_vs_{b}"] = len(groups[a] & groups[b])
    return report


def summarize_split(df):
    return summarize_df(
        df, columns=["collapsed_birads", "birads", "birads_numeric", "view", "machine_family", "dataset", "machine"]
    )


def write_outputs(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str,
    args: argparse.Namespace,
    filtered_full: Optional[pd.DataFrame] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, df in splits.items():
        out_path = output_dir / f"{prefix}_{split_name}.csv"
        df.to_csv(out_path, index=False)
        paths[split_name] = str(out_path)

    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_paths": paths,
        "split_summaries": {name: summarize_split(df) for name, df in splits.items()},
        "patient_overlap": patient_overlap_report(splits, group_col=args.group_col),
    }

    if filtered_full is not None:
        summary["filtered_full_summary"] = summarize_split(filtered_full)

    summary_path = output_dir / f"{prefix}_summary.json"
    save_json(summary, summary_path)

    print("\nWrote split CSVs:")
    for name, path in paths.items():
        print(f"  {name:5s}: {path}")

    print(f"\nWrote summary:\n  {summary_path}")

    print("\nSplit summary:")
    for name, df in splits.items():
        s = summarize_split(df)
        print(f"  {name:5s}: rows={s.get('rows')} patients={s.get('patients', 'n/a')}")
        if "collapsed_birads_counts" in s:
            print(f"         collapsed_birads={s['collapsed_birads_counts']}")
        if "view_counts" in s:
            print(f"         view={s['view_counts']}")
        if "machine_family_counts" in s:
            print(f"         machine_family={s['machine_family_counts']}")

    if summary["patient_overlap"]:
        print("\nPatient overlap:")
        for k, v in summary["patient_overlap"].items():
            print(f"  {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MG train/val/test CSV splits restricted to one machine family, optionally one view."
    )

    parser.add_argument("--full-csv", type=Path, default=DEFAULT_FULL_CSV)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default=None)

    parser.add_argument("--machine-family", type=str, default="Hologic/Lorad")
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        help="Optional exact view filter, e.g. CC or MLO. Off by default.",
    )

    parser.add_argument(
        "--mode",
        choices=["filter_existing", "resplit"],
        default="filter_existing",
        help=(
            "filter_existing: filter existing train/val/test CSVs. "
            "resplit: make a fresh patient-disjoint split from the filtered full CSV."
        ),
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-col", type=str, default="patient")
    parser.add_argument("--stratify-col", type=str, default="collapsed_birads")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    family_slug = slugify(args.machine_family)
    view_slug = f"_view_{slugify(args.view)}" if args.view else ""
    mode_slug = "filtered_existing" if args.mode == "filter_existing" else "resplit"
    prefix = args.prefix or f"mg_{family_slug}{view_slug}_{mode_slug}"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_MG_DIR / f"splits_{family_slug}{view_slug}_{mode_slug}"

    print("Creating MG subset split")
    print(f"  machine_family: {args.machine_family}")
    print(f"  view filter:    {args.view if args.view else '(off)'}")
    print(f"  mode:           {args.mode}")
    print(f"  output_dir:     {output_dir}")
    print(f"  prefix:         {prefix}")

    if args.mode == "filter_existing":
        train_df = filter_df(pd.read_csv(args.train_csv), args.machine_family, args.view)
        val_df = filter_df(pd.read_csv(args.val_csv), args.machine_family, args.view)
        test_df = filter_df(pd.read_csv(args.test_csv), args.machine_family, args.view)

        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        filtered_full = None
        if args.full_csv.exists():
            filtered_full = filter_df(pd.read_csv(args.full_csv), args.machine_family, args.view)

        write_outputs(splits, output_dir, prefix, args, filtered_full=filtered_full)

    else:
        full_df = pd.read_csv(args.full_csv)
        if "original_index" not in full_df:
            full_df["original_index"] = np.arange(len(full_df))
        filtered_full = filter_df(full_df, args.machine_family, args.view)

        train_df, val_df, test_df = stratified_patient_split(
            filtered_full,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            group_col=args.group_col,
            stratify_col=args.stratify_col,
        )

        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        write_outputs(splits, output_dir, prefix, args, filtered_full=filtered_full)


if __name__ == "__main__":
    main()
