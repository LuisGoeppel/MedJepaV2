"""Frozen probe fitting and JSON reporting on shared features."""

from __future__ import annotations
import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from .config import save_json


def apply_machine_grouping(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, min_train_count: int, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    counts = train_df["machine"].value_counts()
    keep = set(counts.index.tolist())
    if min_train_count > 1:
        keep &= set(counts[counts >= min_train_count].index.tolist())
    if top_k and top_k > 0:
        keep &= set(counts.head(top_k).index.tolist())
    if not keep:
        keep = set(counts.head(max(1, top_k or 10)).index.tolist())

    def group(s: pd.Series) -> pd.Series:
        return s.apply(lambda x: x if x in keep else "Other")

    out_train, out_val, out_test = train_df.copy(), val_df.copy(), test_df.copy()
    out_train["machine"] = group(out_train["machine"])
    out_val["machine"] = group(out_val["machine"])
    out_test["machine"] = group(out_test["machine"])
    info = {
        "machine_grouping": {
            "min_train_count": int(min_train_count),
            "top_k": int(top_k),
            "kept_labels": sorted([str(x) for x in keep]),
            "num_kept_labels": int(len(keep)),
            "train_raw_num_labels": int(len(counts)),
        }
    }
    return out_train, out_val, out_test, info


class LinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_counts(labels: list[str]) -> dict[str, int]:
    vc = pd.Series(labels).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def make_label_map(
    train_labels: list[str], val_labels: list[str], test_labels: list[str], ordered: Optional[list[str]] = None
) -> dict[str, int]:
    present = set(train_labels) | set(val_labels) | set(test_labels)
    if ordered:
        names = [x for x in ordered if x in present]
        names += sorted([x for x in present if x not in set(names)])
    else:
        # Put Other/unknown at the end for readability.
        names = sorted([x for x in present if x not in {"Other", "unknown"}])
        if "Other" in present:
            names.append("Other")
        if "unknown" in present:
            names.append("unknown")
    return {name: i for i, name in enumerate(names)}


def labels_to_tensor(labels: list[str], label_to_idx: dict[str, int]) -> torch.Tensor:
    return torch.tensor([label_to_idx[str(x)] for x in labels], dtype=torch.long)


def balanced_subset_indices(y: torch.Tensor, max_total: int, seed: int) -> torch.Tensor:
    if max_total <= 0 or len(y) <= max_total:
        return torch.arange(len(y))
    g = torch.Generator().manual_seed(seed)
    classes = sorted(y.unique().tolist())
    per_class = max(1, max_total // max(1, len(classes)))
    chunks = []
    for c in classes:
        idx = torch.where(y == int(c))[0]
        perm = torch.randperm(len(idx), generator=g)
        chunks.append(idx[perm[: min(per_class, len(idx))]])
    out = torch.cat(chunks)
    return out[torch.randperm(len(out), generator=g)]


def compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y.cpu(), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-12)


def evaluate_probe(
    probe: nn.Module, x: torch.Tensor, y: torch.Tensor, class_names: list[str], device: torch.device
) -> dict[str, Any]:
    probe.eval()
    with torch.inference_mode():
        logits = probe(x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()

    true = y.cpu().numpy()

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
    }


def run_one_probe(
    target_name: str,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    test_x: torch.Tensor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if target_name not in train_df.columns:
        return {"target": target_name, "status": "skipped", "reason": f"Column {target_name!r} not found."}

    train_labels = train_df[target_name].fillna("unknown").astype(str).tolist()
    val_labels = val_df[target_name].fillna("unknown").astype(str).tolist()
    test_labels = test_df[target_name].fillna("unknown").astype(str).tolist()

    ordered = None
    if target_name == "collapsed_birads":
        ordered = ["routine", "follow_up", "biopsy"]
    elif target_name == "laterality":
        ordered = ["left", "right", "bilateral", "unknown"]
    elif target_name == "view":
        ordered = ["CC", "MLO", "mediolateral", "lateromedial", "unknown"]

    label_to_idx = make_label_map(train_labels, val_labels, test_labels, ordered=ordered)
    idx_to_label = [None] * len(label_to_idx)

    for k, v in label_to_idx.items():
        idx_to_label[v] = k

    class_names = [str(x) for x in idx_to_label]
    num_classes = len(class_names)

    if num_classes < 2:
        return {"target": target_name, "status": "skipped", "reason": "Fewer than 2 classes available."}

    train_y = labels_to_tensor(train_labels, label_to_idx)
    val_y = labels_to_tensor(val_labels, label_to_idx)
    test_y = labels_to_tensor(test_labels, label_to_idx)

    # Balanced subsample only for probe training. Evaluation always uses full val/test.
    train_idx = balanced_subset_indices(train_y, args.probe_train_max_samples, args.seed)
    train_x_sub = train_x[train_idx]
    train_y_sub = train_y[train_idx]

    probe = LinearProbe(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_learning_rate, weight_decay=args.probe_weight_decay)
    weights = None if args.no_class_weights else compute_class_weights(train_y_sub, num_classes).to(device)
    loader = DataLoader(
        TensorDataset(train_x_sub, train_y_sub), batch_size=args.probe_batch_size, shuffle=True, drop_last=False
    )

    best_score = -float("inf")
    best_state = None
    best_epoch = -1

    for epoch in tqdm(range(args.probe_epochs), desc=f"Train probe [{target_name}]", leave=False):
        probe.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(xb.size(0))

        val_metrics = evaluate_probe(probe, val_x, val_y, class_names, device)

        if args.select_best_by == "last":
            score = float(epoch)
        elif args.select_best_by == "val_balanced_accuracy":
            score = val_metrics["balanced_accuracy"]
        else:
            score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)

    train_metrics = evaluate_probe(probe, train_x_sub, train_y_sub, class_names, device)
    val_metrics = evaluate_probe(probe, val_x, val_y, class_names, device)
    test_metrics = evaluate_probe(probe, test_x, test_y, class_names, device) if args.evaluate_test else None

    return {
        "target": target_name,
        "status": "ok",
        "num_classes": int(num_classes),
        "class_names": class_names,
        "label_to_index": label_to_idx,
        "counts": {
            "train_full": class_counts(train_labels),
            "train_probe_subset": class_counts([class_names[int(i)] for i in train_y_sub.tolist()]),
            "val": class_counts(val_labels),
            "test": class_counts(test_labels),
        },
        "probe_training": {
            "epochs": int(args.probe_epochs),
            "best_epoch": int(best_epoch),
            "selected_by": args.select_best_by,
            "used_class_weights": bool(not args.no_class_weights),
            "train_samples_used": int(len(train_y_sub)),
        },
        "metrics": {
            "train_probe_subset": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "timing_sec": float(time.perf_counter() - t0),
    }


def run_probe_report(
    features: dict, settings, seed: int, checkpoint: str, output_json: Path, device: torch.device
) -> dict[str, Any]:
    args = argparse.Namespace(**asdict(settings), seed=seed)
    train_x, train_df = features["train"]
    val_x, val_df = features["val"]
    if settings.evaluate_test:
        test_x, test_df = features["test"]
    else:
        test_x, test_df = train_x[:0], train_df.iloc[:0].copy()
    train_df, val_df, test_df, grouping = apply_machine_grouping(
        train_df, val_df, test_df, args.machine_min_train_count, args.machine_top_k
    )
    probes = {
        target: run_one_probe(target, train_x, val_x, test_x, train_df, val_df, test_df, args, device)
        for target in settings.targets
    }
    summary = []
    for target, result in probes.items():
        if result["status"] != "ok":
            continue
        row = {"target": target, "num_classes": result["num_classes"]}
        for split in ("val", "test"):
            if result["metrics"][split] is not None:
                row.update({f"{split}_{key}": value for key, value in result["metrics"][split].items()})
        summary.append(row)
    output = {
        "checkpoint": str(checkpoint),
        "probe_settings": asdict(settings),
        "seed": seed,
        "data": {
            "num_train_rows": len(train_df),
            "num_val_rows": len(val_df),
            "num_test_rows": len(test_df),
            **grouping,
        },
        "summary": summary,
        "probes": probes,
    }
    save_json(output, output_json)
    return output
