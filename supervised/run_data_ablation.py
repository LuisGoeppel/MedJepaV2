#!/usr/bin/env python3
"""
Compare supervised training-set sizes and dataset restrictions.

Creates supervised ViT baseline runs for:
  train sizes: 2k, 10k, 50k
  dataset settings:
    1) full MG
    2) machine_family == Hologic/Lorad
    3) machine_family == Hologic/Lorad and view == CC by default

For each dataset setting, the script first creates one fixed patient-disjoint
train/val/test pool. Then it samples exactly N balanced collapsed_birads
training examples from the train pool. Validation/test pools stay fixed across
2k/10k/50k for the same dataset setting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


from core.config import slugify, human_seconds
from core.data import (
    CLASS_NAMES,
    set_seed,
    infer_bin_spec,
    open_memmap,
    summarize_df,
    make_patient_disjoint_pools,
    prepare_baseline_metadata,
)
from core.models import build_supervised_model, unwrap_model
from core.supervised import evaluate, fit_supervised, make_baseline_loaders
from core.plotting import plot_baseline_history as plot_training_curves, plot_confusion
from core.plotting import plot_data_ablation_summary


@dataclass
class ExperimentSpec:
    name: str
    setting_name: str
    train_size: int
    machine_family: Optional[str]
    view: Optional[str]


def balanced_train_sample(
    train_pool: pd.DataFrame, train_size: int, seed: int
) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, int]]:
    counts = train_pool["collapsed_birads"].value_counts().to_dict()
    base = train_size // len(CLASS_NAMES)
    remainder = train_size % len(CLASS_NAMES)
    quotas = {cls: base + (1 if i < remainder else 0) for i, cls in enumerate(CLASS_NAMES)}
    missing = {
        cls: quotas[cls] - int(counts.get(cls, 0)) for cls in CLASS_NAMES if int(counts.get(cls, 0)) < quotas[cls]
    }
    if missing:
        reason = f"not enough training samples for balanced size {train_size}; missing quotas: {missing}; available counts: {counts}"
        return None, reason, quotas
    parts = []
    for i, cls in enumerate(CLASS_NAMES):
        subset = train_pool[train_pool["collapsed_birads"] == cls]
        parts.append(subset.sample(n=quotas[cls], random_state=seed + i))
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed + 100).copy()
    return sampled, None, quotas


def save_checkpoint(
    out_path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    experiment: ExperimentSpec,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    ckpt = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "class_names": CLASS_NAMES,
        "model_config": {
            "backbone": args.backbone,
            "image_size": args.image_size,
            "num_classes": len(CLASS_NAMES),
            "in_chans": 1,
            "pretrained": bool(args.pretrained),
        },
        "experiment": experiment.__dict__,
    }
    torch.save(ckpt, out_path)


def train_one_experiment(
    args: argparse.Namespace,
    experiment: ExperimentSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mmap: np.memmap,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    set_seed(args.seed, benchmark=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "sampled_train.csv", index=False)
    val_df.to_csv(output_dir / "val_pool.csv", index=False)
    test_df.to_csv(output_dir / "test_pool.csv", index=False)
    split_summary = {"train": summarize_df(train_df), "val": summarize_df(val_df), "test": summarize_df(test_df)}
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "experiment": experiment.__dict__,
        "split_summary": split_summary,
        "class_names": CLASS_NAMES,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    train_loader, val_loader, test_loader = make_baseline_loaders(
        (train_df, val_df, test_df), mmap, args.image_size, args
    )

    model = build_supervised_model(args.backbone, args.image_size, args.pretrained, len(CLASS_NAMES)).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using torch.nn.DataParallel over {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    start_time = time.time()
    fit = fit_supervised(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=args.epochs,
        amp=args.amp,
        amp_dtype=torch.float16,
        criterion=criterion,
        scheduler=scheduler,
        output_dir=output_dir,
        fixed_labels=True,
        eval_criterion=criterion,
        log_every_epochs=args.log_every_epochs,
        progress_factory=tqdm,
        checkpoint_names={"balanced_accuracy": "best_model.pt"},
        save_checkpoint=lambda path, epoch, metrics: save_checkpoint(path, model, args, experiment, epoch, metrics),
    )
    history = fit["history"]
    best_epoch = fit["best_epochs"]["balanced_accuracy"]
    best_val_metrics = fit["best_metrics"]["balanced_accuracy"]

    save_checkpoint(output_dir / "final_model.pt", model, args, experiment, args.epochs, history[-1] if history else {})
    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    unwrap_model(model).load_state_dict(best_ckpt["model_state_dict"], strict=True)
    test_metrics = evaluate(
        model, test_loader, device=device, criterion=criterion, amp=args.amp, amp_dtype=torch.float16, fixed_labels=True
    )
    metrics = {
        "status": "completed",
        "experiment": experiment.__dict__,
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_val_metrics,
        "test_metrics_best_model": test_metrics,
        "runtime_seconds": float(time.time() - start_time),
        "runtime_human": human_seconds(time.time() - start_time),
        "split_summary": split_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion(
        test_metrics["confusion_matrix"],
        f"{experiment.name}: test confusion matrix",
        output_dir / "confusion_matrix_best.png",
    )
    return metrics


def make_experiment_specs(args: argparse.Namespace) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    settings = [
        ("full", None, None),
        ("hologic_lorad", args.machine_family, None),
        (f"hologic_lorad_{slugify(args.single_view)}", args.machine_family, args.single_view),
    ]
    for size in args.train_sizes:
        for setting_name, machine_family, view in settings:
            specs.append(
                ExperimentSpec(
                    name=f"{setting_name}_balanced_{size}",
                    setting_name=setting_name,
                    train_size=int(size),
                    machine_family=machine_family,
                    view=view,
                )
            )
    return specs


def filter_setting(df: pd.DataFrame, machine_family: Optional[str], view: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if machine_family is not None:
        out = out[
            out["machine_family"].astype(str).str.strip().str.casefold() == machine_family.strip().casefold()
        ].copy()
    if view is not None:
        out = out[out["view"].astype(str).str.strip().str.casefold() == view.strip().casefold()].copy()
    return out


def create_summary_outputs(output_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in all_results:
        exp = result.get("experiment", {})
        test = result.get("test_metrics_best_model", {})
        split = result.get("split_summary", {})
        row = {
            "status": result.get("status"),
            "experiment": exp.get("name"),
            "setting": exp.get("setting_name"),
            "train_size_requested": exp.get("train_size"),
            "machine_family": exp.get("machine_family"),
            "view": exp.get("view"),
            "reason": result.get("reason"),
            "best_epoch": result.get("best_epoch"),
            "test_accuracy": test.get("accuracy"),
            "test_balanced_accuracy": test.get("balanced_accuracy"),
            "test_macro_f1": test.get("macro_f1"),
            "runtime_human": result.get("runtime_human"),
        }
        for split_name in ["train", "val", "test"]:
            if split_name in split:
                row[f"{split_name}_rows"] = split[split_name].get("rows")
                row[f"{split_name}_patients"] = split[split_name].get("patients")
                counts = split[split_name].get("collapsed_birads_counts", {})
                for cls in CLASS_NAMES:
                    row[f"{split_name}_{cls}"] = counts.get(cls, 0)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")

    plot_data_ablation_summary(output_dir, summary_df, all_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised ViT BI-RADS baseline experiments on MG data.")
    parser.add_argument("--mg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-name", type=str, default="mg-only-all.csv")
    parser.add_argument("--bin-name", type=str, default="mg-only-all.bin")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[2000, 10000, 50000])
    parser.add_argument("--machine-family", type=str, default="Hologic/Lorad")
    parser.add_argument("--single-view", type=str, default="CC")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-parallel", action="store_true", help="Use torch.nn.DataParallel if multiple GPUs are visible."
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Override train sizes and epochs for a fast end-to-end check."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.train_sizes = [300]
        args.epochs = 2
        args.batch_size = min(args.batch_size, 64)
        args.output_dir = args.output_dir / "smoke_test"
    set_seed(args.seed, benchmark=False)
    csv_path = args.mg_dir / args.csv_name
    bin_path = args.mg_dir / args.bin_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"BIN not found: {bin_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV...")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded rows: {len(df_raw):,}")
    df = prepare_baseline_metadata(df_raw, policy="data_ablation")
    print(f"Rows after keeping valid collapsed_birads classes: {len(df):,}")
    print("collapsed_birads counts:", df["collapsed_birads"].value_counts().to_dict())
    print("machine_family counts:", df["machine_family"].value_counts().to_dict())
    print("view counts:", df["view"].value_counts().to_dict())
    bin_spec = infer_bin_spec(bin_path, len(df_raw))
    print("BIN spec:", bin_spec)
    mmap = open_memmap(bin_path, bin_spec, len(df_raw))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("Visible CUDA devices:", torch.cuda.device_count())

    specs = make_experiment_specs(args)
    setting_specs = [
        ("full", None, None),
        ("hologic_lorad", args.machine_family, None),
        (f"hologic_lorad_{slugify(args.single_view)}", args.machine_family, args.single_view),
    ]
    setting_pools: Dict[str, Dict[str, Any]] = {}
    for setting_name, machine_family, view in setting_specs:
        setting_df = filter_setting(df, machine_family, view)
        print(f"\nSetting {setting_name}: {len(setting_df):,} rows")
        print("  collapsed_birads:", setting_df["collapsed_birads"].value_counts().to_dict())
        if len(setting_df) == 0:
            setting_pools[setting_name] = {"status": "empty", "reason": "no rows after filter"}
            continue
        try:
            train_pool, val_pool, test_pool, split_info = make_patient_disjoint_pools(setting_df, seed=args.seed)
            setting_pools[setting_name] = {
                "status": "ok",
                "train_pool": train_pool,
                "val_pool": val_pool,
                "test_pool": test_pool,
                "split_info": split_info,
            }
            print("  train pool:", summarize_df(train_pool))
            print("  val pool:  ", summarize_df(val_pool))
            print("  test pool: ", summarize_df(test_pool))
            print("  split info:", split_info)
        except Exception as exc:
            setting_pools[setting_name] = {"status": "failed", "reason": str(exc)}
            print(f"  [SKIP SETTING] {exc}")

    all_results: List[Dict[str, Any]] = []
    for spec in specs:
        print("\n" + "=" * 80)
        print(f"Experiment: {spec.name}")
        print("=" * 80)
        pools = setting_pools.get(spec.setting_name)
        exp_dir = args.output_dir / spec.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        if pools is None or pools.get("status") != "ok":
            reason = pools.get("reason", "setting pool not available") if pools else "setting pool not available"
            print(f"[SKIP] {reason}")
            result = {"status": "skipped", "reason": reason, "experiment": spec.__dict__}
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            all_results.append(result)
            create_summary_outputs(args.output_dir, all_results)
            continue
        train_pool, val_pool, test_pool = pools["train_pool"], pools["val_pool"], pools["test_pool"]
        sampled_train, reason, quotas = balanced_train_sample(
            train_pool, train_size=spec.train_size, seed=args.seed + spec.train_size
        )
        if sampled_train is None:
            print(f"[SKIP] {reason}")
            result = {
                "status": "skipped",
                "reason": reason,
                "experiment": spec.__dict__,
                "requested_quotas": quotas,
                "train_pool_summary": summarize_df(train_pool),
                "val_pool_summary": summarize_df(val_pool),
                "test_pool_summary": summarize_df(test_pool),
            }
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            all_results.append(result)
            create_summary_outputs(args.output_dir, all_results)
            continue
        print("Sampled train:", summarize_df(sampled_train))
        print("Val pool:", summarize_df(val_pool))
        print("Test pool:", summarize_df(test_pool))
        try:
            metrics = train_one_experiment(args, spec, sampled_train, val_pool, test_pool, mmap, device, exp_dir)
            all_results.append(metrics)
        except Exception as exc:
            print(f"[FAILED] {spec.name}: {type(exc).__name__}: {exc}")
            result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "experiment": spec.__dict__}
            (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            all_results.append(result)
        create_summary_outputs(args.output_dir, all_results)
    create_summary_outputs(args.output_dir, all_results)
    print("\nAll experiments finished or skipped.")


if __name__ == "__main__":
    main()
