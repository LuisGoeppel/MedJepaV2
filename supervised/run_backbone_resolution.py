#!/usr/bin/env python3
"""
Supervised MG BI-RADS resolution/backbone baseline.

Runs:
  image sizes: 224, 384, 512
  backbones: vit_small_patch16_224 and resnet18 by default
  train subset: 50k examples from the train pool, including all follow_up and biopsy
                examples from the train pool, filled with routine examples
  loss: weighted cross entropy, weights computed from the sampled train subset

Outputs one folder per experiment plus summary.csv/json/report.pdf.
"""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    BinSpec,
    infer_bin_spec,
    open_memmap,
    summarize_df,
    make_patient_disjoint_pools,
    prepare_baseline_metadata,
)
from core.models import build_supervised_model, unwrap_model
from core.supervised import evaluate, fit_supervised, make_baseline_loaders, compute_class_weights
from core.plotting import plot_baseline_history as plot_training_curves, plot_confusion
from core.plotting import plot_resolution_summary


@dataclass
class ExperimentSpec:
    name: str
    backbone: str
    image_size: int


def make_minority_inclusive_train_subset(
    train_pool: pd.DataFrame,
    train_size: int,
    seed: int,
    include_classes: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    parts = []
    used_indices = set()
    included_counts = {}
    for cls in include_classes:
        cls_df = train_pool[train_pool["collapsed_birads"] == cls]
        parts.append(cls_df)
        used_indices.update(cls_df.index.tolist())
        included_counts[cls] = int(len(cls_df))
    n_included = sum(len(p) for p in parts)
    remaining = train_size - n_included
    if remaining < 0:
        raise ValueError(f"train_size={train_size} is smaller than included minority count={n_included}.")
    routine_df = train_pool[(train_pool["collapsed_birads"] == "routine") & (~train_pool.index.isin(used_indices))]
    if len(routine_df) < remaining:
        raise ValueError(f"Need {remaining} routine examples, but only {len(routine_df)} are available.")
    routine_sample = routine_df.sample(n=remaining, random_state=seed)
    parts.append(routine_sample)
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed + 100).copy()
    info = {
        "requested_train_size": int(train_size),
        "included_all_classes": list(include_classes),
        "included_counts": included_counts,
        "routine_fill_count": int(remaining),
        "final_counts": {str(k): int(v) for k, v in sampled["collapsed_birads"].value_counts().to_dict().items()},
    }
    return sampled, info


def save_checkpoint(
    out_path: Path,
    model: nn.Module,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict[str, Any],
    class_weights: Dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "epoch": int(epoch),
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "class_weights": class_weights,
            "model_config": {
                "backbone": spec.backbone,
                "image_size": int(spec.image_size),
                "num_classes": len(CLASS_NAMES),
                "in_chans": 1,
                "pretrained": bool(args.pretrained),
            },
            "experiment": spec.__dict__,
        },
        out_path,
    )


def train_one_experiment(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mmap: np.memmap,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    set_seed(args.seed + spec.image_size + len(spec.backbone), benchmark=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {"train": summarize_df(train_df), "val": summarize_df(val_df), "test": summarize_df(test_df)}
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "experiment": spec.__dict__,
        "split_summary": split_summary,
        "class_names": CLASS_NAMES,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    train_loader, val_loader, test_loader = make_baseline_loaders(
        (train_df, val_df, test_df), mmap, spec.image_size, args
    )
    model = build_supervised_model(spec.backbone, spec.image_size, args.pretrained, len(CLASS_NAMES)).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} CUDA devices")
        model = nn.DataParallel(model)

    weights_tensor = compute_class_weights(
        train_df["target"].to_numpy(), normalization="balanced", dtype=np.float64
    ).to(device)
    weights_dict = {name: float(weights_tensor[i]) for i, name in enumerate(CLASS_NAMES)}
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_learning_rate
    )
    start = time.time()
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
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_names={
            "balanced_accuracy": "best_by_val_balanced_accuracy.pt",
            "macro_f1": "best_by_val_macro_f1.pt",
            "loss": "best_by_val_loss.pt",
        },
        save_checkpoint=lambda path, epoch, metrics: save_checkpoint(
            path, model, spec, args, epoch, metrics, weights_dict
        ),
    )
    history = fit["history"]
    best_epochs = {"val_" + key: value for key, value in fit["best_epochs"].items()}

    save_checkpoint(
        output_dir / "final_model.pt",
        model,
        spec,
        args,
        history[-1]["epoch"] if history else 0,
        history[-1] if history else {},
        weights_dict,
    )

    def eval_ckpt(filename: str) -> Optional[Dict[str, Any]]:
        path = output_dir / filename
        if not path.exists():
            return None
        ckpt = torch.load(path, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=True)
        test = evaluate(
            model, test_loader, device, args.amp, criterion=criterion, amp_dtype=torch.float16, fixed_labels=True
        )
        return {
            "checkpoint": filename,
            "epoch": int(ckpt.get("epoch", 0)),
            "val_metrics_at_checkpoint": ckpt.get("metrics", {}),
            "test_metrics": test,
        }

    test_by_ckpt = {
        "best_by_val_balanced_accuracy": eval_ckpt("best_by_val_balanced_accuracy.pt"),
        "best_by_val_macro_f1": eval_ckpt("best_by_val_macro_f1.pt"),
        "best_by_val_loss": eval_ckpt("best_by_val_loss.pt"),
    }
    primary = test_by_ckpt["best_by_val_balanced_accuracy"]
    primary_test = primary["test_metrics"] if primary is not None else {}

    result = {
        "status": "completed",
        "experiment": spec.__dict__,
        "class_weights": weights_dict,
        "best_epochs": best_epochs,
        "test_by_checkpoint": test_by_ckpt,
        "test_metrics_primary_best_bal_acc": primary_test,
        "runtime_seconds": float(time.time() - start),
        "runtime_human": human_seconds(time.time() - start),
        "split_summary": split_summary,
        "epochs_run": len(history),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    plot_training_curves(history, output_dir / "training_curves.png")
    if primary_test and "confusion_matrix" in primary_test:
        plot_confusion(
            primary_test["confusion_matrix"],
            f"{spec.name}: test confusion matrix",
            output_dir / "confusion_matrix_best_bal_acc.png",
        )

    model = optimizer = scheduler = train_loader = val_loader = test_loader = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def make_specs(args: argparse.Namespace) -> List[ExperimentSpec]:
    specs = []
    for image_size in args.image_sizes:
        for backbone in args.backbones:
            name = f"{slugify(backbone)}_img{image_size}_minority_inclusive_{args.train_size}_weighted_ce"
            specs.append(ExperimentSpec(name=name, backbone=backbone, image_size=int(image_size)))
    return specs


def _namespace_to_jsonable(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _namespace_from_jsonable(d: Dict[str, Any]) -> argparse.Namespace:
    path_keys = {"mg_dir", "output_dir"}
    out = {}
    for k, v in d.items():
        out[k] = Path(v) if k in path_keys and v is not None else v
    return argparse.Namespace(**out)


def _worker_run_one_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one experiment on one assigned GPU.

    The parent process creates the shared split/subset CSV files and owns summary aggregation.
    Each worker reopens the BIN memmap independently and writes only its own experiment directory.
    """
    spec = ExperimentSpec(**payload["spec"])
    args = _namespace_from_jsonable(payload["args"])
    gpu_id = payload.get("gpu_id")
    bin_spec = BinSpec(**payload["bin_spec"])
    n_raw_rows = int(payload["n_raw_rows"])
    bin_path = Path(payload["bin_path"])
    train_csv = Path(payload["train_csv"])
    val_csv = Path(payload["val_csv"])
    test_csv = Path(payload["test_csv"])

    exp_dir = args.output_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "experiment.log"

    try:
        with log_path.open("a", buffering=1, encoding="utf-8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
            print("=" * 90)
            print(f"Experiment: {spec.name}")
            print(f"Assigned GPU: {gpu_id}")
            print(f"Started at UTC: {datetime.now(timezone.utc).isoformat()}")
            print("=" * 90)

            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)
            mmap = open_memmap(bin_path, bin_spec, n_raw_rows)

            if torch.cuda.is_available():
                if gpu_id is None:
                    device = torch.device("cuda")
                else:
                    torch.cuda.set_device(gpu_id)
                    device = torch.device(f"cuda:{gpu_id}")
            else:
                device = torch.device("cpu")

            print("device:", device)
            result = train_one_experiment(
                spec=spec,
                args=args,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                mmap=mmap,
                device=device,
                output_dir=exp_dir,
            )
            result["assigned_gpu"] = gpu_id
            result["log_path"] = str(log_path)
            print(f"Finished at UTC: {datetime.now(timezone.utc).isoformat()}")
            return result
    except Exception as exc:
        tb = traceback.format_exc()
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write("\n[FAILED]\n")
            log_f.write(tb)
            log_f.write("\n")
        result = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "assigned_gpu": gpu_id,
            "log_path": str(log_path),
            "experiment": spec.__dict__,
        }
        (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result


def _gpu_worker_loop(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Process experiments sequentially on one GPU.

    This avoids the common scheduling bug where two experiments are accidentally assigned to the
    same GPU while another GPU is idle.
    """
    while True:
        payload = task_queue.get()
        if payload is None:
            break
        payload["gpu_id"] = gpu_id
        result = _worker_run_one_experiment(payload)
        result_queue.put(result)


def run_experiments_parallel(
    specs: List[ExperimentSpec],
    args: argparse.Namespace,
    bin_path: Path,
    bin_spec: BinSpec,
    n_raw_rows: int,
    sampled_train_path: Path,
    val_pool_path: Path,
    test_pool_path: Path,
) -> List[Dict[str, Any]]:
    gpu_ids = [int(x) for x in (args.parallel_gpus or [])]
    if len(gpu_ids) == 0:
        raise ValueError("parallel_gpus must contain at least one GPU id")

    args_for_workers = argparse.Namespace(**vars(args))
    args_for_workers.data_parallel = False

    base_payloads: List[Dict[str, Any]] = []
    for spec in specs:
        base_payloads.append(
            {
                "spec": spec.__dict__,
                "args": _namespace_to_jsonable(args_for_workers),
                "gpu_id": None,  # filled by the per-GPU worker
                "bin_path": str(bin_path),
                "bin_spec": bin_spec.__dict__,
                "n_raw_rows": int(n_raw_rows),
                "train_csv": str(sampled_train_path),
                "val_csv": str(val_pool_path),
                "test_csv": str(test_pool_path),
            }
        )

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    for payload in base_payloads:
        task_queue.put(payload)
    for _ in gpu_ids:
        task_queue.put(None)

    print(f"Launching {len(base_payloads)} experiments across GPUs {gpu_ids}.")
    print("Each GPU runs at most one experiment at a time; when it finishes, it takes the next queued experiment.")

    workers: List[mp.Process] = []
    for gpu_id in gpu_ids:
        proc = ctx.Process(target=_gpu_worker_loop, args=(gpu_id, task_queue, result_queue), daemon=False)
        proc.start()
        workers.append(proc)

    results: List[Dict[str, Any]] = []
    try:
        for _ in range(len(base_payloads)):
            result = result_queue.get()
            exp = result.get("experiment", {})
            print(
                f"[DONE] {exp.get('name')} status={result.get('status')} "
                f"gpu={result.get('assigned_gpu')} log={result.get('log_path')}"
            )
            results.append(result)
            create_summary_outputs(args.output_dir, results)
    finally:
        for proc in workers:
            proc.join(timeout=5)
        for proc in workers:
            if proc.is_alive():
                proc.terminate()

    order = {spec.name: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order.get((r.get("experiment", {}) or {}).get("name", ""), 10**9))
    return results


def create_summary_outputs(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    rows = []
    for result in results:
        exp = result.get("experiment", {})
        primary = result.get("test_metrics_primary_best_bal_acc", {}) or {}
        split = result.get("split_summary", {}) or {}
        best = result.get("best_epochs", {}) or {}
        row = {
            "status": result.get("status"),
            "experiment": exp.get("name"),
            "backbone": exp.get("backbone"),
            "image_size": exp.get("image_size"),
            "epochs_run": result.get("epochs_run"),
            "best_epoch_val_bal_acc": best.get("val_balanced_accuracy"),
            "best_epoch_val_macro_f1": best.get("val_macro_f1"),
            "best_epoch_val_loss": best.get("val_loss"),
            "test_accuracy": primary.get("accuracy"),
            "test_balanced_accuracy": primary.get("balanced_accuracy"),
            "test_macro_f1": primary.get("macro_f1"),
            "test_loss": primary.get("loss"),
            "runtime_human": result.get("runtime_human"),
            "reason": result.get("reason"),
        }
        per_class = primary.get("per_class_recall", {}) or {}
        for cls in CLASS_NAMES:
            row[f"test_recall_{cls}"] = per_class.get(cls)
        for split_name in ["train", "val", "test"]:
            if split_name in split:
                row[f"{split_name}_rows"] = split[split_name].get("rows")
                counts = split[split_name].get("collapsed_birads_counts", {})
                for cls in CLASS_NAMES:
                    row[f"{split_name}_{cls}"] = counts.get(cls, 0)
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    plot_resolution_summary(output_dir, summary_df, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised MG resolution/backbone baselines.")
    parser.add_argument("--mg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-name", type=str, default="mg-only-all.csv")
    parser.add_argument("--bin-name", type=str, default="mg-only-all.bin")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[224, 384, 512])
    parser.add_argument("--backbones", type=str, nargs="+", default=["vit_small_patch16_224", "resnet18"])
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--include-all-classes", type=str, nargs="+", default=["follow_up", "biopsy"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument(
        "--parallel-gpus",
        type=int,
        nargs="+",
        default=None,
        help="Run independent experiments in parallel on these GPU ids, e.g. --parallel-gpus 0 1 2 3. Do not combine with --data-parallel.",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.image_sizes = [224]
        args.backbones = [args.backbones[0]]
        args.train_size = 600
        args.epochs = 2
        args.patience = 0
        args.batch_size = min(args.batch_size, 64)
        args.num_workers = min(args.num_workers, 2)
        args.output_dir = args.output_dir / "smoke_test"

    set_seed(args.seed, benchmark=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, bin_path = args.mg_dir / args.csv_name, args.mg_dir / args.bin_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"BIN not found: {bin_path}")

    print("Loading CSV...")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded rows: {len(df_raw):,}")
    df = prepare_baseline_metadata(df_raw, policy="resolution")
    print(f"Rows after keeping valid target classes: {len(df):,}")
    print("collapsed_birads counts:", df["collapsed_birads"].value_counts().to_dict())
    print("machine_family counts:", df["machine_family"].value_counts().to_dict())
    print("view counts:", df["view"].value_counts().to_dict())

    bin_spec = infer_bin_spec(bin_path, len(df_raw))
    print("BIN spec:", bin_spec)
    mmap = open_memmap(bin_path, bin_spec, len(df_raw))

    print("Creating fixed patient-disjoint train/val/test pools...")
    train_pool, val_pool, test_pool, split_info = make_patient_disjoint_pools(
        df, args.seed, args.train_ratio, args.val_ratio, args.test_ratio
    )
    print("split info:", split_info)
    print("train pool:", summarize_df(train_pool))
    print("val pool:", summarize_df(val_pool))
    print("test pool:", summarize_df(test_pool))

    sampled_train, sampling_info = make_minority_inclusive_train_subset(
        train_pool, args.train_size, args.seed, args.include_all_classes
    )
    print("sampled train:", summarize_df(sampled_train))
    print("sampling info:", sampling_info)

    sampled_train_path = args.output_dir / f"sampled_train_{args.train_size}.csv"
    val_pool_path = args.output_dir / "val_pool.csv"
    test_pool_path = args.output_dir / "test_pool.csv"
    sampled_train.to_csv(sampled_train_path, index=False)
    val_pool.to_csv(val_pool_path, index=False)
    test_pool.to_csv(test_pool_path, index=False)
    global_split_summary = {
        "split_info": split_info,
        "sampling_info": sampling_info,
        "train_pool": summarize_df(train_pool),
        "sampled_train": summarize_df(sampled_train),
        "val_pool": summarize_df(val_pool),
        "test_pool": summarize_df(test_pool),
        "paths": {
            "sampled_train": str(sampled_train_path),
            "val_pool": str(val_pool_path),
            "test_pool": str(test_pool_path),
        },
    }
    (args.output_dir / "global_split_summary.json").write_text(
        json.dumps(global_split_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    specs = make_specs(args)

    if args.parallel_gpus is not None and len(args.parallel_gpus) > 0:
        if args.data_parallel:
            raise ValueError("Do not combine --parallel-gpus with --data-parallel. Use one process per GPU instead.")
        results = run_experiments_parallel(
            specs=specs,
            args=args,
            bin_path=bin_path,
            bin_spec=bin_spec,
            n_raw_rows=len(df_raw),
            sampled_train_path=sampled_train_path,
            val_pool_path=val_pool_path,
            test_pool_path=test_pool_path,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        if torch.cuda.is_available():
            print("visible CUDA devices:", torch.cuda.device_count())

        results: List[Dict[str, Any]] = []
        for spec in specs:
            print("\n" + "=" * 90)
            print(f"Experiment: {spec.name}")
            print("=" * 90)
            exp_dir = args.output_dir / spec.name
            try:
                result = train_one_experiment(spec, args, sampled_train, val_pool, test_pool, mmap, device, exp_dir)
                results.append(result)
            except Exception as exc:
                print(f"[FAILED] {spec.name}: {type(exc).__name__}: {exc}")
                exp_dir.mkdir(parents=True, exist_ok=True)
                result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "experiment": spec.__dict__}
                (exp_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                results.append(result)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            create_summary_outputs(args.output_dir, results)

    create_summary_outputs(args.output_dir, results)
    print("\nAll experiments finished or failed/skipped.")


if __name__ == "__main__":
    main()
