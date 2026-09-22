#!/usr/bin/env python3
"""
Analyze how much collapsed BI-RADS can be predicted from external factors alone.

This script computes factor-only baselines for collapsed BI-RADS using columns such as
view, dataset, machine_family, machine, and laterality. It is useful for checking
whether metadata/confounders alone can improve balanced accuracy above the 3-class
no-information baseline of ~0.333.

Outputs:
  - factor_birads_report.txt
  - factor_birads_report.json
  - <factor>_crosstab_counts.csv
  - <factor>_crosstab_row_normalized.csv
  - <factor>_p_factor_given_class.csv

Example:
  python -m analysis.analyze_mg_factor_birads_baselines \
    --csv /path/to/mg_test.csv \
    --out-dir /path/to/output \
    --factors view machine laterality
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


from core.data import CLASS_NAMES as CLASS_ORDER, _safe_json_loads as parse_json_maybe, read_csv_clean
from core.config import save_json


def collapse_birads_value(x):
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()

    # Important: check follow_up before benign/routine,
    # because "probably benign" contains "benign".
    if "follow" in s or "probably benign" in s or s.startswith("(3)") or s == "3":
        return "follow_up"
    if (
        "biopsy" in s
        or "suspicious" in s
        or "malign" in s
        or s.startswith("(4)")
        or s.startswith("(5)")
        or s in {"4", "5"}
    ):
        return "biopsy"
    if (
        "routine" in s
        or "healthy" in s
        or "negative" in s
        or "benign" in s
        or s.startswith("(1)")
        or s.startswith("(2)")
        or s in {"1", "2"}
    ):
        return "routine"

    return "unknown"


def normalize_view(x):
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return "unknown"
    if "cranial" in s or s == "cc":
        return "CC"
    if "mediolateral" in s or "mlo" in s:
        return "MLO"
    return str(x).strip()


def normalize_laterality(x):
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    if s in {"left", "l"}:
        return "left"
    if s in {"right", "r"}:
        return "right"
    return "unknown"


def machine_to_family(x):
    if pd.isna(x):
        return "unknown"
    s = str(x).lower()
    if "hologic" in s or "lorad" in s:
        return "Hologic/Lorad"
    if "ge" in s or "senographe" in s:
        return "GE/Senographe"
    if "howtek" in s or "lumysis" in s:
        return "Howtek/Lumysis"
    return "unknown"


def add_derived_columns(df):
    df = df.copy()

    if "collapsed_birads" not in df.columns:
        if "birads" in df.columns:
            df["collapsed_birads"] = df["birads"].map(collapse_birads_value)
        elif "original_birads" in df.columns:
            df["collapsed_birads"] = df["original_birads"].map(collapse_birads_value)
        else:
            raise ValueError("Could not find collapsed_birads, birads, or original_birads column.")
    else:
        df["collapsed_birads"] = df["collapsed_birads"].map(collapse_birads_value)

    contexts = None
    if "context" in df.columns:
        contexts = df["context"].map(parse_json_maybe)

    if "view" not in df.columns or df["view"].isna().mean() > 0.5:
        if contexts is not None:
            df["view"] = contexts.map(lambda d: d.get("exam", {}).get("view", "unknown"))
        else:
            df["view"] = "unknown"
    df["view"] = df["view"].map(normalize_view)

    if "laterality" not in df.columns or df["laterality"].isna().mean() > 0.5:
        if contexts is not None:
            df["laterality"] = contexts.map(lambda d: d.get("exam", {}).get("laterality", "unknown"))
        else:
            df["laterality"] = "unknown"
    df["laterality"] = df["laterality"].map(normalize_laterality)

    if "machine_family" not in df.columns:
        if "machine" in df.columns:
            df["machine_family"] = df["machine"].map(machine_to_family)
        else:
            df["machine_family"] = "unknown"

    if "dataset" not in df.columns:
        df["dataset"] = "unknown"

    if "machine" not in df.columns:
        df["machine"] = "unknown"

    return df


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float((y_true == y_pred).mean())
    recalls = {}
    f1s = {}

    for c in CLASS_ORDER:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        support = int((y_true == c).sum())

        recall = tp / support if support > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recalls[c] = recall
        f1s[c] = f1

    balanced_acc = float(np.nanmean(list(recalls.values())))
    macro_f1 = float(np.nanmean(list(f1s.values())))

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "recall_by_class": recalls,
        "f1_by_class": f1s,
    }


def mutual_information_bits(tab):
    arr = tab.to_numpy(dtype=float)
    n = arr.sum()
    if n <= 0:
        return 0.0

    pxy = arr / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i, 0] > 0 and py[0, j] > 0:
                mi += pxy[i, j] * math.log2(pxy[i, j] / (px[i, 0] * py[0, j]))
    return float(mi)


def cramers_v(tab):
    arr = tab.to_numpy(dtype=float)
    n = arr.sum()
    if n <= 0:
        return 0.0

    row_sum = arr.sum(axis=1, keepdims=True)
    col_sum = arr.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n

    mask = expected > 0
    chi2 = ((arr[mask] - expected[mask]) ** 2 / expected[mask]).sum()
    r, k = arr.shape
    denom = n * max(1, min(r - 1, k - 1))
    return float(math.sqrt(chi2 / denom))


def analyze_factor(df, factor, out_dir, max_levels):
    d = df[[factor, "collapsed_birads"]].copy()
    d[factor] = d[factor].fillna("unknown").astype(str)
    d["collapsed_birads"] = d["collapsed_birads"].fillna("unknown").astype(str)
    d = d[d["collapsed_birads"].isin(CLASS_ORDER)].copy()

    original_num_levels = int(d[factor].nunique())
    grouped = False

    vc = d[factor].value_counts()
    if original_num_levels > max_levels:
        keep = set(vc.head(max_levels).index)
        d[factor] = d[factor].where(d[factor].isin(keep), "Other")
        grouped = True

    tab = pd.crosstab(d[factor], d["collapsed_birads"]).reindex(columns=CLASS_ORDER, fill_value=0)
    tab = tab.sort_index()

    # Baseline with no external information: predict one constant class.
    # For balanced accuracy, any constant class gives 1 / num_classes if all classes exist.
    no_info_pred = np.array(["routine"] * len(d))
    no_info_metrics = evaluate_predictions(d["collapsed_birads"].to_numpy(), no_info_pred)

    # Standard group-majority classifier: for each factor value, predict the most frequent class P(Y|factor).
    majority_map = tab.idxmax(axis=1).to_dict()
    majority_pred = d[factor].map(majority_map).to_numpy()
    majority_metrics = evaluate_predictions(d["collapsed_birads"].to_numpy(), majority_pred)

    # Balanced-accuracy-optimal classifier using only factor:
    # For each factor value z, choose class y maximizing P(Z=z | Y=y).
    # This maximizes macro recall / balanced accuracy, not plain accuracy.
    class_totals = tab.sum(axis=0).replace(0, np.nan)
    p_z_given_y = tab.div(class_totals, axis=1)
    balanced_map = p_z_given_y.idxmax(axis=1).to_dict()
    balanced_pred = d[factor].map(balanced_map).to_numpy()
    balanced_metrics = evaluate_predictions(d["collapsed_birads"].to_numpy(), balanced_pred)

    tab.to_csv(out_dir / f"{factor}_crosstab_counts.csv")
    tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).to_csv(out_dir / f"{factor}_crosstab_row_normalized.csv")
    p_z_given_y.to_csv(out_dir / f"{factor}_p_factor_given_class.csv")

    result = {
        "factor": factor,
        "n": int(len(d)),
        "original_num_levels": original_num_levels,
        "num_levels_used": int(d[factor].nunique()),
        "grouped_rare_levels_into_other": grouped,
        "unknown_fraction": float((d[factor] == "unknown").mean()),
        "class_counts": d["collapsed_birads"].value_counts().reindex(CLASS_ORDER, fill_value=0).astype(int).to_dict(),
        "factor_counts": d[factor].value_counts().to_dict(),
        "mutual_information_bits": mutual_information_bits(tab),
        "cramers_v": cramers_v(tab),
        "no_info_constant_routine": no_info_metrics,
        "group_majority_factor_only": {
            "mapping": majority_map,
            "metrics": majority_metrics,
            "delta_bal_acc_vs_no_info": majority_metrics["balanced_accuracy"] - no_info_metrics["balanced_accuracy"],
        },
        "balanced_optimal_factor_only": {
            "mapping": balanced_map,
            "metrics": balanced_metrics,
            "delta_bal_acc_vs_no_info": balanced_metrics["balanced_accuracy"] - no_info_metrics["balanced_accuracy"],
        },
    }

    return result


def write_text_report(results, output_path):
    lines = []
    lines.append("MG collapsed BI-RADS factor-only baseline report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "  no_info_constant_routine: predicts routine for every image. Balanced accuracy is the 3-class no-information baseline."
    )
    lines.append("  group_majority_factor_only: predicts the most common BI-RADS class within each factor value.")
    lines.append(
        "  balanced_optimal_factor_only: best possible deterministic classifier for balanced accuracy using only that factor."
    )
    lines.append("")

    for r in results:
        lines.append("-" * 60)
        lines.append(f"Factor: {r['factor']}")
        lines.append(f"N: {r['n']}")
        lines.append(f"Levels used: {r['num_levels_used']} / original {r['original_num_levels']}")
        lines.append(f"Unknown fraction: {r['unknown_fraction']:.4f}")
        lines.append(f"Class counts: {r['class_counts']}")
        lines.append(f"Mutual information: {r['mutual_information_bits']:.6f} bits")
        lines.append(f"Cramer's V: {r['cramers_v']:.6f}")
        lines.append("")

        base = r["no_info_constant_routine"]
        gm = r["group_majority_factor_only"]
        bo = r["balanced_optimal_factor_only"]

        lines.append(
            f"No-info BA:             {base['balanced_accuracy']:.4f} | acc={base['accuracy']:.4f} | macro_f1={base['macro_f1']:.4f}"
        )
        lines.append(
            f"Group-majority BA:      {gm['metrics']['balanced_accuracy']:.4f} | delta={gm['delta_bal_acc_vs_no_info']:+.4f} | acc={gm['metrics']['accuracy']:.4f} | macro_f1={gm['metrics']['macro_f1']:.4f}"
        )
        lines.append(
            f"Balanced-optimal BA:    {bo['metrics']['balanced_accuracy']:.4f} | delta={bo['delta_bal_acc_vs_no_info']:+.4f} | acc={bo['metrics']['accuracy']:.4f} | macro_f1={bo['metrics']['macro_f1']:.4f}"
        )
        lines.append("")
        lines.append(f"Balanced-optimal mapping: {bo['mapping']}")
        lines.append(f"Group-majority mapping:   {gm['mapping']}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="One or more CSVs. Multiple CSVs are concatenated.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--factors", nargs="+", default=["dataset", "machine_family", "view", "laterality"])
    ap.add_argument("--max-levels", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for p in args.csv:
        p = Path(p)
        d = read_csv_clean(p)
        d["_source_csv"] = str(p)
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df = add_derived_columns(df)

    results = []
    for factor in args.factors:
        if factor not in df.columns:
            print(f"Skipping missing factor: {factor}")
            continue
        print(f"Analyzing factor: {factor}")
        results.append(analyze_factor(df, factor, out_dir, args.max_levels))

    save_json(results, out_dir / "factor_birads_report.json")

    write_text_report(results, out_dir / "factor_birads_report.txt")

    print("")
    print(f"Wrote report to: {out_dir / 'factor_birads_report.txt'}")
    print(f"Wrote JSON to:   {out_dir / 'factor_birads_report.json'}")


if __name__ == "__main__":
    main()
