"""Reusable supervised loaders, a weighted cross-entropy epoch and evaluation."""

from __future__ import annotations
from typing import Any
import time
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
from .config import save_json


def compute_class_weights(
    labels: np.ndarray, num_classes: int = 3, *, normalization="mean", dtype=np.float32
) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(dtype)
    if normalization == "balanced":
        if np.any(counts == 0):
            raise ValueError("Every class requires training samples for balanced weights.")
        return torch.as_tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)
    if normalization != "mean":
        raise ValueError(f"Unknown class-weight normalization: {normalization}")
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.as_tensor(weights, dtype=torch.float32)


def make_baseline_loader(dataset, batch_size, num_workers, train, *, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=bool(train and drop_last),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_baseline_loaders(frames, mmap, image_size, args):
    from .data import baseline_dataset_from_memmap

    return tuple(
        make_baseline_loader(
            baseline_dataset_from_memmap(frame, mmap, image_size, index == 0, not args.no_augment),
            args.batch_size if index == 0 else (args.eval_batch_size or args.batch_size),
            args.num_workers,
            index == 0,
        )
        for index, frame in enumerate(frames)
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    *,
    criterion=None,
    amp_dtype=None,
    fixed_labels=False,
    show_progress=False,
) -> dict[str, Any]:
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    total = 0
    use_cuda = device.type == "cuda"
    amp_dtype = amp_dtype or (torch.bfloat16 if use_cuda else torch.float32)

    with torch.inference_mode():
        batches = tqdm(loader, desc="eval", leave=False) if show_progress else loader
        for x, y in batches:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp and use_cuda),
            ):
                logits = model(x)
                loss = criterion(logits, y) if criterion is not None else F.cross_entropy(logits.float(), y)
            pred = logits.argmax(dim=1)
            preds.append(pred.detach().cpu())
            trues.append(y.detach().cpu())
            total_loss += float(loss.item()) * x.size(0)
            total += int(x.size(0))

    if not trues:
        raise ValueError("Evaluation loader contains no samples.")
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
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2] if fixed_labels else None, average="macro", zero_division=0)
        ),
        "per_class_recall": {CLASS_NAMES[i]: float(recall[i]) for i in range(3)},
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
    model,
    loader,
    optimizer,
    device,
    *,
    amp,
    class_weights,
    label_smoothing,
    grad_clip_norm,
    description,
    amp_dtype=None,
    scaler=None,
    progress_factory=tqdm,
):
    """Train batches; the caller controls frozen modules and model.train()/eval()."""
    total_loss = 0.0
    correct = total = 0
    use_cuda = device.type == "cuda"
    amp_dtype = amp_dtype or (torch.bfloat16 if use_cuda else torch.float32)
    for x, y in progress_factory(loader, desc=description, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(amp and use_cuda)):
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y, weight=class_weights, label_smoothing=float(label_smoothing))
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
        else:
            loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(x.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def fit_supervised(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    *,
    epochs,
    amp,
    criterion,
    output_dir,
    checkpoint_names,
    save_checkpoint,
    amp_dtype=torch.bfloat16,
    scheduler=None,
    lr_for_epoch=None,
    patience=0,
    min_delta=0.0,
    grad_clip_norm=0.0,
    fixed_labels=False,
    history_writer=None,
    eval_criterion=None,
    log_every_epochs=1,
    progress_factory=tqdm,
    eval_progress=False,
):
    """Fit one baseline; experiments own checkpoint payloads and selection policy.

    Every monitored metric resets patience, matching the resolution sweep. Single
    runs pass only balanced_accuracy. The caller supplies a validation criterion
    when weighted evaluation is part of the experiment's protocol.

    Retain the caller's progress wrapper: tqdm.auto can create a DataLoader
    iterator eagerly, consuming an extra seed compared with plain tqdm.
    """
    if epochs < 1 or len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training requires positive epochs and nonempty train/validation loaders.")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=bool(amp and device.type == "cuda"))
        if amp_dtype == torch.float16
        else None
    )
    history, best_metrics = [], {}
    best_values = {key: float("inf") if key == "loss" else -1.0 for key in checkpoint_names}
    best_epochs = {key: 0 for key in checkpoint_names}
    no_improve = 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        if lr_for_epoch is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_for_epoch(epoch - 1)
        model.train()
        train_loss, train_accuracy = train_supervised_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp,
            class_weights=criterion.weight,
            label_smoothing=criterion.label_smoothing,
            grad_clip_norm=grad_clip_norm,
            description=f"Epoch {epoch}/{epochs}",
            amp_dtype=amp_dtype,
            scaler=scaler,
            progress_factory=progress_factory,
        )
        if scheduler is not None:
            scheduler.step()
        val = evaluate(
            model,
            val_loader,
            device,
            amp,
            criterion=eval_criterion,
            amp_dtype=amp_dtype,
            fixed_labels=fixed_labels,
            show_progress=eval_progress,
        )
        row = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val["loss"],
            val_accuracy=val["accuracy"],
            val_balanced_accuracy=val["balanced_accuracy"],
            val_macro_f1=val["macro_f1"],
            lr=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.time() - start,
            epoch_time_sec=time.time() - epoch_start,
        )
        history.append(row)
        if history_writer is None:
            save_json(history, output_dir / "training_history.json")
        else:
            history_writer(history)
        improved = False
        for key, filename in checkpoint_names.items():
            value = val[key]
            better = value < best_values[key] - min_delta if key == "loss" else value > best_values[key] + min_delta
            if better:
                best_values[key], best_epochs[key], best_metrics[key] = value, epoch, val
                save_checkpoint(output_dir / filename, epoch, val)
                improved = True
        no_improve = 0 if improved else no_improve + 1
        if epoch == 1 or epoch == epochs or epoch % max(1, log_every_epochs) == 0:
            print(
                f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_bal_acc={val['balanced_accuracy']:.4f}", flush=True
            )
        if patience > 0 and no_improve >= patience:
            break
    return dict(history=history, best_epochs=best_epochs, best_metrics=best_metrics)
