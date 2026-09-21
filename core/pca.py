"""PCA sampling and PDF visualization of already-extracted embeddings."""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

ORDINAL_COLOR_ORDERS = {
    "collapsed_birads": ["routine", "follow_up", "biopsy"],
    "birads_numeric": ["1", "2", "3", "4", "5"],
    "has_segmentation": ["no_segmentation", "has_segmentation"],
}
BLUE_PURPLE_RED = LinearSegmentedColormap.from_list("blue_purple_red", ["#2166AC", "#7B3294", "#B2182B"])


def _age_group_sort_key(label: Any) -> int:
    s = str(label)
    m = re.search(r"\d{2,3}", s)
    if m:
        return int(m.group(0))
    return 10_000


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
        return out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    raise ValueError(f"Unknown sampling mode: {args.sampling}")


def make_label_series(df: pd.DataFrame, col: str, max_categories: int) -> pd.Series:
    if col not in df.columns:
        raise KeyError(col)
    s = df[col].fillna("missing").astype(str)
    vc = s.value_counts(dropna=False)
    if len(vc) > max_categories:
        keep = set(vc.index[: max_categories - 1])
        s = s.where(s.isin(keep), other="Other")
    return s


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
    if min(features.shape) < 3:
        raise ValueError("PCA report requires at least three rows and three feature dimensions")
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
    if getattr(args, "color_by", None):
        candidate_cols = args.color_by
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
            f"PCA explained variance: PC1={evr[0] * 100:.2f}%, PC2={evr[1] * 100:.2f}%, PC3={evr[2] * 100:.2f}%",
            f"Cumulative PC1-PC3: {evr[:3].sum() * 100:.2f}%",
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
        overview_cols = [
            c for c in ["collapsed_birads", "birads_numeric", "dataset", "machine_family"] if c in plot_cols
        ]
        if overview_cols:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes_flat = axes.ravel()
            for ax in axes_flat:
                ax.axis("off")
            for ax, col in zip(axes_flat, overview_cols):
                ax.axis("on")
                labels = make_label_series(df, col, args.max_categories)
                plot_pca2(ax, z3[:, :2], labels, f"PCA 2D colored by {col}", col=col)
            fig.suptitle(
                f"Same PCA coordinates, different labels\nPC1={evr[0] * 100:.2f}%, PC2={evr[1] * 100:.2f}%", fontsize=14
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Individual larger 2D pages for each metadata column.
        for col in plot_cols:
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = make_label_series(df, col, args.max_categories)
            plot_pca2(
                ax,
                z3[:, :2],
                labels,
                f"PCA 2D colored by {col}\nPC1={evr[0] * 100:.2f}%, PC2={evr[1] * 100:.2f}%",
                col=col,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Additional PC-pair diagnostic pages for clinical labels.
        for col in [
            "collapsed_birads",
            "birads_numeric",
            "age_group",
            "view",
            "laterality",
            "dataset",
            "machine_family",
        ]:
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
                ax.set_xlabel(f"PC{a + 1}")
                ax.set_ylabel(f"PC{b + 1}")
                ax.set_title(f"PC{a + 1} vs PC{b + 1}")
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
        for col in [
            "collapsed_birads",
            "birads_numeric",
            "age_group",
            "view",
            "laterality",
            "dataset",
            "machine_family",
        ]:
            if col not in plot_cols:
                continue
            labels = make_label_series(df, col, args.max_categories)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            plot_pca3(
                ax,
                z3,
                labels,
                f"PCA 3D colored by {col}\nPC1={evr[0] * 100:.2f}%, PC2={evr[1] * 100:.2f}%, PC3={evr[2] * 100:.2f}%",
                col=col,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def run_pca_report(features: dict, settings, seed: int, checkpoint: str, output_pdf: Path) -> None:
    from dataclasses import asdict

    args = argparse.Namespace(**asdict(settings), seed=seed, checkpoint=checkpoint)
    names = ["train", "val", "test"] if settings.split == "all" else [settings.split]
    metadata = pd.concat([features[name][1] for name in names], ignore_index=True)
    embeddings = np.concatenate([features[name][0].numpy() for name in names])
    metadata["_feature_row"] = np.arange(len(metadata))
    selected = sample_dataframe(metadata, args)
    values = embeddings[selected.pop("_feature_row").to_numpy(dtype=int)]
    create_pdf_report(selected, values, args, output_pdf)
