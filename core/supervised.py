"""Reusable supervised loaders, a weighted cross-entropy epoch and evaluation."""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from .data import CLASS_NAMES, MGSupDataset, seed_worker
from .config import ExperimentConfig


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.as_tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    total = 0
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp and use_cuda),
            ):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
            pred = logits.argmax(dim=1)
            preds.append(pred.detach().cpu())
            trues.append(y.detach().cpu())
            total_loss += float(loss.item()) * x.size(0)
            total += int(x.size(0))

    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pred_counts = cm.sum(axis=0)
    pred_fracs = pred_counts / max(1, int(pred_counts.sum()))
    min_recall = float(np.min(recall))
    missing_predicted_classes = [CLASS_NAMES[i] for i in range(3) if int(pred_counts[i]) == 0]

    return {
        "loss": float(total_loss / max(1, total)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "min_class_recall": min_recall,
        "missing_predicted_classes": missing_predicted_classes,
        "prediction_counts": {CLASS_NAMES[i]: int(pred_counts[i]) for i in range(3)},
        "prediction_fractions": {CLASS_NAMES[i]: float(pred_fracs[i]) for i in range(3)},
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


def make_supervised_loaders(
    frames, full_num_rows: int, cfg: ExperimentConfig, train_transform, eval_transform, device, seed: int
):
    """Seed training shuffling and release worker processes between epochs/runs."""
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for index, frame in enumerate(frames):
        training = index == 0
        dataset = MGSupDataset(
            frame,
            cfg.bin_path,
            full_num_rows,
            (cfg.image_height, cfg.image_width),
            cfg.memmap_dtype,
            train_transform if training else eval_transform,
            cfg.normalize_mode,
            cfg.percentile_low,
            cfg.percentile_high,
        )
        kwargs = {"num_workers": cfg.num_workers, "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
        if cfg.num_workers > 0:
            kwargs.update(persistent_workers=False, prefetch_factor=2)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=cfg.batch_size if training else cfg.eval_batch_size,
                shuffle=training,
                drop_last=False,
                generator=generator if training else None,
                **kwargs,
            )
        )
    return tuple(loaders)


def train_supervised_epoch(
    model, loader, optimizer, device, *, amp, class_weights, label_smoothing, grad_clip_norm, description
):
    """Train batches; the caller controls frozen modules and model.train()/eval()."""
    total_loss = 0.0
    correct = total = 0
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    for x, y in tqdm(loader, desc=description, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(amp and use_cuda)):
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y, weight=class_weights, label_smoothing=float(label_smoothing))
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(x.size(0))
    return total_loss / max(1, total), correct / max(1, total)
