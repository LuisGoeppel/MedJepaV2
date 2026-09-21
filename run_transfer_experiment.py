"""Label-efficiency experiment: nested budgets, transfer stages and run orchestration.

This file owns the experiment protocol. Shared batch training, data, models and
plotting live in core/. Importing this file never starts an experiment.
"""

from __future__ import annotations
import gc
import hashlib
import json
import math
import re
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
from core.config import ExperimentConfig, parse_transfer_config, save_json
from core.data import CLASS_NAMES, load_splits, verify_no_patient_leakage, set_seed, validate_bin
from core.models import (
    ViTBackboneClassifier,
    clone_state_dict_to_cpu,
    restore_initial_model_state,
    maybe_update_config_from_checkpoint,
    load_jepa_backbone,
    set_backbone_trainable,
)
from core.transforms import make_supervised_transform
from core.supervised import compute_class_weights, evaluate, make_supervised_loaders, train_supervised_epoch
from core.plotting import plot_history, plot_confusion, plot_class_distribution, plot_summary_outputs


def parse_budget(s: str) -> Optional[int]:
    s = str(s).strip().lower().replace("_", "")
    if s in {"full", "all", "none", "0", "-1"}:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k)?", s)
    if not m:
        raise ValueError(f"Invalid subset size: {s}. Use e.g. 2000, 2k, 50000, full.")
    val = float(m.group(1))
    if m.group(2) == "k":
        val *= 1000
    n = int(round(val))
    if n <= 0:
        raise ValueError(f"Subset size must be positive or full, got {s}")
    return n


def budget_name(s: str) -> str:
    n = parse_budget(s)
    return "full" if n is None else str(n)


def stable_int_from_text(text: str, modulo: int = 1_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _budget_to_n(budget_text: str, full_n: int) -> int:
    n = parse_budget(budget_text)
    return full_n if n is None else min(int(n), full_n)


def raw_balance_degree(n: int, min_n: int, full_n: int, curve: str) -> float:
    if full_n <= min_n:
        return 0.0 if n >= full_n else 1.0
    if n <= min_n:
        return 1.0
    if n >= full_n:
        return 0.0
    if curve == "log":
        val = 1.0 - math.log(float(n) / float(min_n)) / math.log(float(full_n) / float(min_n))
    elif curve == "linear":
        val = 1.0 - (float(n) - float(min_n)) / (float(full_n) - float(min_n))
    else:
        raise ValueError(f"Unknown balance curve: {curve}")
    return float(np.clip(val, 0.0, 1.0))


def feasible_balance_cap(n: int, capacities: np.ndarray, natural_probs: np.ndarray) -> float:
    """Maximum b in p(b)=b*(1/K)+(1-b)*p_natural that does not request
    more unique samples from any class than are available.
    """
    if n <= 0:
        return 0.0
    k = len(capacities)
    balanced_p = 1.0 / float(k)
    caps = [1.0]
    for c in range(k):
        denom = balanced_p - float(natural_probs[c])
        if denom > 1e-12:
            upper = (float(capacities[c]) / float(n) - float(natural_probs[c])) / denom
            caps.append(upper)
    return float(np.clip(min(caps), 0.0, 1.0))


def blended_probs(balance_degree: float, natural_probs: np.ndarray) -> np.ndarray:
    k = len(natural_probs)
    return balance_degree * np.full(k, 1.0 / k, dtype=np.float64) + (1.0 - balance_degree) * natural_probs


def allocate_counts(
    n: int,
    probs: np.ndarray,
    capacities: np.ndarray,
    minimum_counts: np.ndarray,
) -> np.ndarray:
    """Integer allocation close to n*probs, respecting capacity and nested minima."""
    desired = np.asarray(probs, dtype=np.float64) * float(n)
    counts = np.floor(desired).astype(np.int64)
    counts = np.minimum(counts, capacities)
    counts = np.maximum(counts, minimum_counts)

    # If monotonic minima pushed us over budget, reduce only above minima.
    while int(counts.sum()) > n:
        candidates = np.where(counts > minimum_counts)[0]
        if len(candidates) == 0:
            raise RuntimeError(f"Cannot satisfy nested allocation: minima sum={int(minimum_counts.sum())} > budget={n}")
        # Remove from the class furthest above its desired count.
        surplus = counts[candidates].astype(np.float64) - desired[candidates]
        c = int(candidates[int(np.argmax(surplus))])
        counts[c] -= 1

    # Fill remaining slots toward desired counts first, then toward remaining capacity.
    while int(counts.sum()) < n:
        candidates = np.where(counts < capacities)[0]
        if len(candidates) == 0:
            raise RuntimeError(f"Ran out of capacity while allocating budget {n}.")
        deficits = desired[candidates] - counts[candidates].astype(np.float64)
        if float(deficits.max()) > 0:
            c = int(candidates[int(np.argmax(deficits))])
        else:
            remaining = capacities[candidates] - counts[candidates]
            c = int(candidates[int(np.argmax(remaining))])
        counts[c] += 1

    return counts


def build_subset_plan(
    train_df: pd.DataFrame,
    budget_texts: tuple[str, ...],
    strategy: str,
    curve: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Build all subsets for one seed at once, guaranteeing nestedness."""
    full_n = len(train_df)
    labels = train_df["target_collapsed"].to_numpy(dtype=np.int64)
    capacities = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.int64)
    natural_probs = capacities.astype(np.float64) / float(full_n)

    budget_pairs = [(b, _budget_to_n(b, full_n)) for b in budget_texts]
    numeric_nonfull = sorted({n for b, n in budget_pairs if parse_budget(b) is not None})
    min_n = numeric_nonfull[0] if numeric_nonfull else full_n

    # One class-specific random order per seed. Prefixes create nested subsets.
    rng = np.random.default_rng(seed)
    class_orders: dict[int, np.ndarray] = {}
    for c in range(len(CLASS_NAMES)):
        idx = np.where(labels == c)[0]
        class_orders[c] = rng.permutation(idx)

    # Build target counts in increasing budget order.
    unique_ns = sorted(set(n for _, n in budget_pairs))
    previous = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    specs_by_n: dict[int, dict[str, Any]] = {}

    for n in unique_ns:
        if n >= full_n:
            raw_b = 0.0
            effective_b = 0.0
            probs = natural_probs.copy()
            target_counts = capacities.copy()
        elif strategy == "natural":
            raw_b = 0.0
            effective_b = 0.0
            probs = natural_probs.copy()
            target_counts = allocate_counts(n, probs, capacities, previous)
        elif strategy == "balanced":
            raw_b = 1.0
            effective_b = feasible_balance_cap(n, capacities, natural_probs)
            probs = blended_probs(effective_b, natural_probs)
            target_counts = allocate_counts(n, probs, capacities, previous)
        elif strategy == "progressive":
            raw_b = raw_balance_degree(n, min_n, full_n, curve)
            effective_b = min(raw_b, feasible_balance_cap(n, capacities, natural_probs))
            probs = blended_probs(effective_b, natural_probs)
            target_counts = allocate_counts(n, probs, capacities, previous)
        else:
            raise ValueError(f"Unknown subset strategy: {strategy}")

        if np.any(target_counts < previous):
            raise RuntimeError("Internal error: subset target counts are not nested.")
        previous = target_counts.copy()

        specs_by_n[n] = {
            "budget_n": int(n),
            "raw_balance_degree": float(raw_b),
            "effective_balance_degree": float(effective_b),
            "target_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "target_counts": {CLASS_NAMES[i]: int(target_counts[i]) for i in range(len(CLASS_NAMES))},
        }

    plan: dict[str, dict[str, Any]] = {}
    previous_set: set[int] = set()

    for budget_text, n in sorted(budget_pairs, key=lambda x: x[1]):
        spec = dict(specs_by_n[n])
        counts_arr = np.array([spec["target_counts"][name] for name in CLASS_NAMES], dtype=np.int64)
        selected_parts = [class_orders[c][: counts_arr[c]] for c in range(len(CLASS_NAMES))]
        selected = np.concatenate(selected_parts)

        # Deterministic order per budget, without affecting which rows are selected.
        order_rng = np.random.default_rng(seed + stable_int_from_text(f"subset-order::{budget_name(budget_text)}"))
        selected = selected.copy()
        order_rng.shuffle(selected)
        subset_df = train_df.iloc[selected].reset_index(drop=True)

        current_set = set(int(x) for x in selected.tolist())
        nested_ok = previous_set.issubset(current_set)
        if not nested_ok:
            raise RuntimeError(f"Nested-subset invariant failed at budget {budget_text}.")
        previous_set = current_set

        spec.update(
            {
                "budget_text": budget_text,
                "budget_resolved": budget_name(budget_text),
                "strategy": strategy,
                "balance_curve": curve,
                "seed": int(seed),
                "nested_ok": True,
                "rows": int(len(subset_df)),
                "actual_counts": {
                    CLASS_NAMES[i]: int((subset_df["target_collapsed"].to_numpy() == i).sum())
                    for i in range(len(CLASS_NAMES))
                },
                "df": subset_df,
            }
        )
        plan[budget_name(budget_text)] = spec

    return plan


def make_optimizer(
    model: ViTBackboneClassifier,
    cfg: ExperimentConfig,
    mode: str,
    stage: str,
) -> torch.optim.Optimizer:
    if mode == "random":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.random_learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if mode == "frozen_jepa" or stage == "head_warmup":
        return torch.optim.AdamW(
            model.head.parameters(),
            lr=cfg.head_learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if mode == "finetune_jepa" and stage == "finetune":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg.backbone_learning_rate},
                {"params": model.head.parameters(), "lr": cfg.finetune_head_learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer mode/stage: {mode}/{stage}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    mode: str,
    stage: str,
    total_epochs: int,
):
    total_epochs = max(1, int(total_epochs))

    if mode == "finetune_jepa" and stage == "head_warmup":
        # V2: keep head warm-up LR constant. Do not decay to eta_min just to reset it.
        return None

    if mode == "random" and cfg.random_warmup_epochs > 0 and total_epochs > 1:
        warmup_epochs = min(int(cfg.random_warmup_epochs), max(1, total_epochs - 1))
        start_factor = float(np.clip(cfg.random_warmup_start_factor, 1e-4, 1.0))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=cfg.min_learning_rate,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=cfg.min_learning_rate,
    )


def get_current_lrs(
    optimizer: torch.optim.Optimizer,
    mode: str,
    stage: str,
) -> tuple[float, float]:
    if len(optimizer.param_groups) == 1:
        lr = float(optimizer.param_groups[0]["lr"])
        if mode == "frozen_jepa" or stage == "head_warmup":
            return 0.0, lr
        return lr, lr
    return float(optimizer.param_groups[0]["lr"]), float(optimizer.param_groups[1]["lr"])


def selection_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    raise ValueError(f"Unknown selection metric: {metric_name}")


def train_one_run(
    cfg: ExperimentConfig,
    mode: str,
    budget_text: str,
    train_used: pd.DataFrame,
    subset_meta: dict[str, Any],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_num_rows: int,
    aug_cfg: dict[str, Any],
    device: torch.device,
    run_seed: int,
    seed_index: int,
    model: ViTBackboneClassifier,
    initial_model_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    run_name = f"budget_{budget_name(budget_text)}__{mode}"
    if cfg.num_seeds > 1:
        out_dir = Path(cfg.output_dir) / f"seed_{run_seed}" / run_name
    else:
        out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same seed for all modes of this repetition. The subset was also generated from this seed.
    set_seed(run_seed)

    if cfg.save_train_used_csv:
        train_used.to_csv(out_dir / "train_used.csv", index=False)

    train_transform = make_supervised_transform(aug_cfg, cfg, train=True)
    eval_transform = make_supervised_transform(aug_cfg, cfg, train=False)
    train_loader, val_loader, test_loader = make_supervised_loaders(
        (train_used, val_df, test_df),
        full_num_rows,
        cfg,
        train_transform,
        eval_transform,
        device,
        run_seed,
    )

    # CRITICAL STABILITY FIX:
    # Reuse the one ViT object created in main(). Constructing another timm ViT after
    # a completed DataLoader/training run reproducibly segfaults in timm trunc_normal_
    # on the single-GPU Determined tasks. Restoring this CPU snapshot is equivalent to
    # starting from the same seeded random initialization, without re-entering
    # timm.create_model().
    restore_initial_model_state(model, initial_model_state)
    model.to(device)
    cfg.embedding_dim = int(model.embedding_dim)

    load_info: Optional[dict[str, Any]] = None
    if mode in {"frozen_jepa", "finetune_jepa"}:
        load_info = load_jepa_backbone(model, cfg.checkpoint, device)

    if mode == "frozen_jepa":
        set_backbone_trainable(model, False)
        stage = "head_warmup"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, cfg, mode, stage, cfg.epochs)
    elif mode == "finetune_jepa":
        set_backbone_trainable(model, False)
        stage = "head_warmup"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(
            optimizer,
            cfg,
            mode,
            stage,
            max(1, min(cfg.head_warmup_epochs, cfg.epochs)),
        )
    elif mode == "random":
        set_backbone_trainable(model, True)
        stage = "finetune"
        optimizer = make_optimizer(model, cfg, mode, stage)
        scheduler = make_scheduler(optimizer, cfg, mode, stage, cfg.epochs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    labels_np = train_used["target_collapsed"].to_numpy(dtype=np.int64)
    class_weights = compute_class_weights(labels_np, len(CLASS_NAMES)).to(device) if cfg.use_class_weights else None

    run_config = {
        "mode": mode,
        "budget": budget_text,
        "budget_resolved": budget_name(budget_text),
        "seed": int(run_seed),
        "seed_index": int(seed_index),
        "subset_strategy": cfg.subset_strategy,
        "subset_meta": {k: v for k, v in subset_meta.items() if k != "df"},
        "train_used_rows": int(len(train_used)),
        "train_used_class_counts": {CLASS_NAMES[i]: int((labels_np == i).sum()) for i in range(3)},
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "class_weights": (
            None
            if class_weights is None
            else {CLASS_NAMES[i]: float(class_weights.detach().cpu()[i]) for i in range(3)}
        ),
        "config": asdict(cfg),
        "load_info": load_info,
    }
    save_json(run_config, out_dir / "config.json")

    if cfg.save_run_plots:
        plot_class_distribution(
            train_used,
            val_df,
            test_df,
            out_dir / "class_distribution.png",
        )

    history: dict[str, list[Any]] = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "val_min_class_recall": [],
        "val_recall_routine": [],
        "val_recall_follow_up": [],
        "val_recall_biopsy": [],
        "val_pred_fraction_routine": [],
        "val_pred_fraction_follow_up": [],
        "val_pred_fraction_biopsy": [],
        "lr_backbone": [],
        "lr_head": [],
        "epoch_time_sec": [],
    }

    best_score = -1.0
    best_val_ba = -1.0
    best_macro_f1 = -1.0
    best_epoch = 0
    best_path = out_dir / f"best_by_val_{cfg.selection_metric}.pt"
    no_improve = 0
    random_min_stop_epoch = int(math.ceil(cfg.epochs * cfg.random_min_epoch_fraction))

    print(f"\n=== Run: {run_name} | seed={run_seed} ===", flush=True)
    print(
        "Model lifecycle: reused shared ViT; reset to initial seeded state; "
        f"JEPA backbone load={'yes' if mode in {'frozen_jepa', 'finetune_jepa'} else 'no'}.",
        flush=True,
    )
    print(
        f"Train rows: {len(train_used):,} | counts: {run_config['train_used_class_counts']} | "
        f"balance raw={subset_meta.get('raw_balance_degree', float('nan')):.4f} "
        f"effective={subset_meta.get('effective_balance_degree', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"Model: backbone={cfg.backbone} image_size={cfg.image_size} "
        f"backbone_num_classes={cfg.backbone_num_classes} embedding_dim={cfg.embedding_dim} "
        f"aug_mode={cfg.supervised_aug_mode}",
        flush=True,
    )
    if mode == "random":
        print(
            f"Random optimization: lr={cfg.random_learning_rate:.2e}, "
            f"warmup={cfg.random_warmup_epochs} epochs, "
            f"earliest early-stop epoch={random_min_stop_epoch}",
            flush=True,
        )
    if load_info is not None:
        print(
            f"Loaded JEPA backbone keys: {load_info['loaded_backbone_keys']} | "
            f"ignored projector: {load_info['ignored_projector_keys']}",
            flush=True,
        )
        if load_info["missing_keys"] or load_info["unexpected_keys"]:
            print(
                f"Backbone load missing keys: {len(load_info['missing_keys'])}, "
                f"unexpected: {len(load_info['unexpected_keys'])}",
                flush=True,
            )

    for epoch in range(1, cfg.epochs + 1):
        if mode == "finetune_jepa" and epoch == cfg.head_warmup_epochs + 1:
            print(
                "Switching finetune_jepa from head warm-up to end-to-end fine-tuning. "
                "Resetting early-stop patience and using the lower finetune head LR.",
                flush=True,
            )
            set_backbone_trainable(model, True)
            stage = "finetune"
            optimizer = make_optimizer(model, cfg, mode, stage)
            scheduler = make_scheduler(
                optimizer,
                cfg,
                mode,
                stage,
                max(1, cfg.epochs - cfg.head_warmup_epochs),
            )
            no_improve = 0

        model.train()
        if mode in {"frozen_jepa", "finetune_jepa"} and stage == "head_warmup":
            model.backbone.eval()

        # Record the learning rates actually used for this epoch.
        lr_backbone, lr_head = get_current_lrs(optimizer, mode, stage)

        epoch_start = time.perf_counter()
        train_loss, train_acc = train_supervised_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=cfg.amp,
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing,
            grad_clip_norm=cfg.grad_clip_norm,
            description=f"{run_name} seed={run_seed} epoch {epoch}/{cfg.epochs}",
        )
        val_metrics = evaluate(model, val_loader, device, cfg.amp)
        elapsed = time.perf_counter() - epoch_start

        history["epoch"].append(epoch)
        history["stage"].append(stage)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["val_weighted_f1"].append(float(val_metrics["weighted_f1"]))
        history["val_min_class_recall"].append(float(val_metrics["min_class_recall"]))
        history["val_recall_routine"].append(float(val_metrics["per_class"]["routine"]["recall"]))
        history["val_recall_follow_up"].append(float(val_metrics["per_class"]["follow_up"]["recall"]))
        history["val_recall_biopsy"].append(float(val_metrics["per_class"]["biopsy"]["recall"]))
        history["val_pred_fraction_routine"].append(float(val_metrics["prediction_fractions"]["routine"]))
        history["val_pred_fraction_follow_up"].append(float(val_metrics["prediction_fractions"]["follow_up"]))
        history["val_pred_fraction_biopsy"].append(float(val_metrics["prediction_fractions"]["biopsy"]))
        history["lr_backbone"].append(float(lr_backbone))
        history["lr_head"].append(float(lr_head))
        history["epoch_time_sec"].append(float(elapsed))

        save_json(history, out_dir / "training_history.json")

        score = selection_value(val_metrics, cfg.selection_metric)
        # BA remains the primary default. Macro-F1 only breaks effectively exact ties.
        improved = score > best_score + 1e-8 or (
            abs(score - best_score) <= 1e-8 and float(val_metrics["macro_f1"]) > best_macro_f1 + 1e-8
        )
        if improved:
            best_score = float(score)
            best_val_ba = float(val_metrics["balanced_accuracy"])
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": int(epoch),
                    "mode": mode,
                    "budget": budget_text,
                    "seed": int(run_seed),
                    "selection_metric": cfg.selection_metric,
                    "selection_score": float(best_score),
                    "model_state_dict": model.state_dict(),
                    "config": run_config,
                    "history": history,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | stage={stage} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"min_recall={val_metrics['min_class_recall']:.4f} | "
            f"best_{cfg.selection_metric}={best_score:.4f}@{best_epoch} | "
            f"lr_backbone={lr_backbone:.2e} lr_head={lr_head:.2e} | "
            f"time={elapsed / 60:.2f} min",
            flush=True,
        )

        if scheduler is not None:
            scheduler.step()

        early_stop_allowed = True
        if mode == "random":
            early_stop_allowed = epoch >= random_min_stop_epoch

        if early_stop_allowed and no_improve >= cfg.patience:
            print(
                f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}. "
                f"Selection metric: {cfg.selection_metric}={best_score:.4f}",
                flush=True,
            )
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was written for {run_name}")

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    val_best_metrics = evaluate(model, val_loader, device, cfg.amp)
    test_metrics = evaluate(model, test_loader, device, cfg.amp)

    result = {
        "mode": mode,
        "budget": budget_text,
        "budget_resolved": budget_name(budget_text),
        "seed": int(run_seed),
        "seed_index": int(seed_index),
        "subset_strategy": cfg.subset_strategy,
        "raw_balance_degree": float(subset_meta.get("raw_balance_degree", 0.0)),
        "effective_balance_degree": float(subset_meta.get("effective_balance_degree", 0.0)),
        "train_used_rows": int(len(train_used)),
        "train_used_class_counts": run_config["train_used_class_counts"],
        "selection_metric": cfg.selection_metric,
        "best_selection_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy_during_training": float(best_val_ba),
        "val_best": val_best_metrics,
        "test": test_metrics,
        "class_names": CLASS_NAMES,
        "output_dir": str(out_dir),
    }
    save_json(result, out_dir / "metrics.json")

    if cfg.save_run_plots:
        plot_history(history, out_dir)
        plot_confusion(
            val_best_metrics["confusion_matrix"],
            f"Val confusion matrix: {run_name}",
            out_dir / "confusion_matrix_val_best.png",
        )
        plot_confusion(
            test_metrics["confusion_matrix"],
            f"Test confusion matrix: {run_name}",
            out_dir / "confusion_matrix_test.png",
        )

    print(
        "Test summary:",
        json.dumps(
            {
                "accuracy": test_metrics["accuracy"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "min_class_recall": test_metrics["min_class_recall"],
                "missing_predicted_classes": test_metrics["missing_predicted_classes"],
                "prediction_fractions": test_metrics["prediction_fractions"],
            },
            indent=2,
        ),
        flush=True,
    )

    # Explicitly release PER-RUN resources. The ViT itself is intentionally kept alive
    # and reused by the next experiment, which avoids the native timm initialization crash.
    del payload
    del optimizer
    if scheduler is not None:
        del scheduler
    del train_loader, val_loader, test_loader

    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _summary_rows(results: list[dict[str, Any]], train_full_n: int) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "seed": r["seed"],
            "seed_index": r["seed_index"],
            "budget": r["budget_resolved"],
            "budget_numeric": int(r["train_used_rows"]),
            "mode": r["mode"],
            "subset_strategy": r["subset_strategy"],
            "raw_balance_degree": r["raw_balance_degree"],
            "effective_balance_degree": r["effective_balance_degree"],
            "train_used_rows": r["train_used_rows"],
            "train_routine": r["train_used_class_counts"]["routine"],
            "train_follow_up": r["train_used_class_counts"]["follow_up"],
            "train_biopsy": r["train_used_class_counts"]["biopsy"],
            "best_epoch": r["best_epoch"],
            "val_balanced_accuracy": r["val_best"]["balanced_accuracy"],
            "val_macro_f1": r["val_best"]["macro_f1"],
            "val_accuracy": r["val_best"]["accuracy"],
            "val_min_class_recall": r["val_best"]["min_class_recall"],
            "test_balanced_accuracy": r["test"]["balanced_accuracy"],
            "test_macro_f1": r["test"]["macro_f1"],
            "test_accuracy": r["test"]["accuracy"],
            "test_weighted_f1": r["test"]["weighted_f1"],
            "test_min_class_recall": r["test"]["min_class_recall"],
            "routine_recall": r["test"]["per_class"]["routine"]["recall"],
            "follow_up_recall": r["test"]["per_class"]["follow_up"]["recall"],
            "biopsy_recall": r["test"]["per_class"]["biopsy"]["recall"],
            "pred_routine_fraction": r["test"]["prediction_fractions"]["routine"],
            "pred_follow_up_fraction": r["test"]["prediction_fractions"]["follow_up"],
            "pred_biopsy_fraction": r["test"]["prediction_fractions"]["biopsy"],
            "missing_predicted_classes": ",".join(r["test"]["missing_predicted_classes"]),
            "output_dir": r["output_dir"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "budget_numeric",
        "raw_balance_degree",
        "effective_balance_degree",
        "train_used_rows",
        "train_routine",
        "train_follow_up",
        "train_biopsy",
        "best_epoch",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_accuracy",
        "val_min_class_recall",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_accuracy",
        "test_weighted_f1",
        "test_min_class_recall",
        "routine_recall",
        "follow_up_recall",
        "biopsy_recall",
        "pred_routine_fraction",
        "pred_follow_up_fraction",
        "pred_biopsy_fraction",
    ]
    agg = df.groupby(["budget", "mode"], as_index=False)[numeric_cols].mean()
    return agg


def write_aggregate_outputs(
    results: list[dict[str, Any]],
    cfg: ExperimentConfig,
    train_full_n: int,
) -> None:
    out_dir = Path(cfg.output_dir)
    df = _summary_rows(results, train_full_n)
    df.to_csv(out_dir / "summary_table.csv", index=False)
    mean_df = _mean_summary(df)
    mean_df.to_csv(out_dir / "summary_mean.csv", index=False)
    save_json(results, out_dir / "summary_results.json")
    try:
        plot_summary_outputs(df, mean_df, out_dir)
    except Exception as exc:
        print(f"WARNING: failed to plot one or more aggregate summaries: {exc}", flush=True)


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if cfg.patience <= 0:
        raise ValueError("--patience must be > 0")
    if not (0.0 <= cfg.random_min_epoch_fraction <= 1.0):
        raise ValueError("--random-min-epoch-fraction must be in [0, 1]")
    if cfg.num_seeds <= 0:
        raise ValueError("--num-seeds must be >= 1")
    if cfg.head_warmup_epochs < 0:
        raise ValueError("--head-warmup-epochs must be >= 0")
    if cfg.random_warmup_epochs < 0:
        raise ValueError("--random-warmup-epochs must be >= 0")


def run_transfer(cfg: ExperimentConfig) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    validate_config(cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hints: dict[str, Any] = {}
    if cfg.auto_model_from_checkpoint:
        checkpoint_hints = maybe_update_config_from_checkpoint(cfg)
        print(
            "Auto model config from checkpoint:",
            json.dumps(checkpoint_hints, indent=2, default=str),
            flush=True,
        )

    save_json(asdict(cfg), out_dir / "experiment_config.json")
    if checkpoint_hints:
        save_json(checkpoint_hints, out_dir / "checkpoint_model_hints.json")

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device), flush=True)

    with open(cfg.aug_config, "r", encoding="utf-8") as f:
        aug_cfg = json.load(f)
    save_json(aug_cfg, out_dir / "augmentation_config_used.json")

    full_df, train_df, val_df, test_df = load_splits(
        cfg.full_csv, cfg.train_csv, cfg.val_csv, cfg.test_csv, prefer_collapsed=True
    )
    leakage = verify_no_patient_leakage(train_df, val_df, test_df)

    validate_bin(cfg.bin_path, len(full_df), cfg.image_height, cfg.image_width, cfg.memmap_dtype)

    split_summary = {
        "patient_leakage": leakage,
        "train": {
            "rows": len(train_df),
            "classes": train_df["collapsed_birads"].value_counts().to_dict(),
        },
        "val": {
            "rows": len(val_df),
            "classes": val_df["collapsed_birads"].value_counts().to_dict(),
        },
        "test": {
            "rows": len(test_df),
            "classes": test_df["collapsed_birads"].value_counts().to_dict(),
        },
    }
    save_json(split_summary, out_dir / "split_summary.json")
    print(json.dumps(split_summary, indent=2), flush=True)

    # ------------------------------------------------------------------
    # Create the timm ViT ONCE for the whole process.
    #
    # On the current single-GPU Determined tasks, constructing a second ViT after a
    # completed training/DataLoader run can crash natively inside timm's trunc_normal_
    # initialization. Every experiment below therefore reuses this model object.
    #
    # The snapshot is taken from the same cfg.seed initialization that the original
    # script used for every run when --num-seeds=1, so the experiment semantics are
    # preserved for the current Hologic/GE runs.
    # ------------------------------------------------------------------
    set_seed(cfg.seed)
    print("Creating shared ViT model once for all transfer runs...", flush=True)
    shared_model = ViTBackboneClassifier(
        cfg.backbone,
        cfg.image_size,
        cfg.embedding_dim,
        cfg.drop_path_rate,
        len(CLASS_NAMES),
        cfg.backbone_num_classes,
    ).to(device)
    cfg.embedding_dim = int(shared_model.embedding_dim)
    initial_model_state = clone_state_dict_to_cpu(shared_model)
    print(
        f"Shared model ready: backbone={cfg.backbone}, "
        f"embedding_dim={cfg.embedding_dim}. "
        "No further timm.create_model() calls will be made.",
        flush=True,
    )

    if cfg.num_seeds > 1:
        print(
            "WARNING: model-reuse stability mode uses the same initial model weights "
            "for all seed repetitions. Dataset subset/order randomness still follows each "
            "run seed. For fully independent model-initialization seeds, run separate "
            "processes/tasks with --num-seeds 1 and different --seed values.",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    start = time.perf_counter()

    for seed_index in range(cfg.num_seeds):
        run_seed = cfg.seed + seed_index
        print(f"\n######## Seed repetition {seed_index + 1}/{cfg.num_seeds}: seed={run_seed} ########", flush=True)

        subset_plan = build_subset_plan(
            train_df,
            cfg.subset_sizes,
            cfg.subset_strategy,
            cfg.balance_curve,
            run_seed,
        )
        subset_plan_serializable = {k: {kk: vv for kk, vv in v.items() if kk != "df"} for k, v in subset_plan.items()}
        save_json(
            subset_plan_serializable,
            out_dir / f"subset_plan_seed_{run_seed}.json",
        )
        print(
            "Subset schedule:",
            json.dumps(
                {
                    k: {
                        "rows": v["rows"],
                        "raw_balance_degree": v["raw_balance_degree"],
                        "effective_balance_degree": v["effective_balance_degree"],
                        "actual_counts": v["actual_counts"],
                    }
                    for k, v in subset_plan.items()
                },
                indent=2,
            ),
            flush=True,
        )

        for budget in cfg.subset_sizes:
            meta = subset_plan[budget_name(budget)]
            train_used = meta["df"]
            for mode in cfg.modes:
                result = train_one_run(
                    cfg,
                    mode,
                    budget,
                    train_used,
                    meta,
                    val_df,
                    test_df,
                    len(full_df),
                    aug_cfg,
                    device,
                    run_seed,
                    seed_index,
                    shared_model,
                    initial_model_state,
                )
                results.append(result)
                write_aggregate_outputs(results, cfg, len(train_df))

    elapsed = time.perf_counter() - start
    save_json(
        {
            "wall_time_sec": elapsed,
            "num_runs": len(results),
            "num_seeds": cfg.num_seeds,
        },
        out_dir / "timing.json",
    )
    write_aggregate_outputs(results, cfg, len(train_df))
    print("Finished all V2 transfer experiments.", flush=True)
    print(
        pd.read_csv(out_dir / "summary_table.csv").to_string(index=False),
        flush=True,
    )


def main():
    run_transfer(parse_transfer_config())


if __name__ == "__main__":
    main()
