"""LeJEPA optimization and training workflow."""

from __future__ import annotations
import contextlib
import json
import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from .config import (
    TrainConfig,
    build_config_from_run_config,
    load_run_config,
    save_json,
    deep_get,
    load_analysis_config,
)
from .data import MedJEPADataset, load_splits, set_seed, validate_bin
from .models import ViTEncoder, read_checkpoint
from .transforms import ConfigurableMGAugmentation
from .loss import LeJEPALoss
from .plotting import plot_training_history
from .training_utils import (
    create_dirs,
    collect_runtime_metadata,
    update_experiment_metadata,
    make_loader,
    is_main_process,
    rank0_print,
    setup_distributed,
    cleanup_distributed,
    maybe_wrap_model,
    unwrap_model,
    save_training_checkpoint,
    maybe_cuda_synchronize,
    representation_stats,
    reduce_sum_tensor,
)


def build_optimizer_scheduler(net: nn.Module, loader: DataLoader, cfg: TrainConfig):
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    optimizer_steps_per_epoch = max(1, math.ceil(len(loader) / max(1, cfg.grad_accum_steps)))
    warmup_steps = max(0, optimizer_steps_per_epoch * max(0, cfg.warmup_epochs))
    total_steps = max(1, optimizer_steps_per_epoch * cfg.epochs)
    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.eta_min),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.eta_min)
    return optimizer, scheduler


def train_lejepa(
    net: nn.Module,
    loss_fn: LeJEPALoss,
    loader: DataLoader,
    optimizer,
    scheduler,
    cfg: TrainConfig,
    device: torch.device,
    dirs: dict[str, Path],
    aug_cfg: dict[str, Any],
    start_epoch: int = 0,
    existing_history: Optional[dict[str, list[float]]] = None,
) -> dict[str, list[float]]:
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    accum_steps = max(1, int(cfg.grad_accum_steps))
    diagnostic_every = max(1, int(cfg.diagnostic_every_batches))

    default_history = {
        "optimization_loss": [],
        "lejepa": [],
        "invariance": [],
        "sigreg": [],
        "sigreg_mode": [],
        "proj_std": [],
        "proj_norm": [],
        "raw_proj_std": [],
        "raw_proj_norm": [],
        "raw_proj_effective_rank": [],
        "loss_proj_std": [],
        "loss_proj_norm": [],
        "loss_proj_effective_rank": [],
        "emb_std": [],
        "emb_norm": [],
        "emb_effective_rank": [],
        "lr": [],
        "epoch_time_sec": [],
        "data_wait_and_augmentation_time_sec": [],
        "h2d_transfer_time_sec": [],
        "forward_loss_time_sec": [],
        "backward_optimizer_time_sec": [],
        "metrics_bookkeeping_time_sec": [],
        "data_wait_and_augmentation_time_per_batch_sec": [],
        "h2d_transfer_time_per_batch_sec": [],
        "forward_loss_time_per_batch_sec": [],
        "backward_optimizer_time_per_batch_sec": [],
        "metrics_bookkeeping_time_per_batch_sec": [],
        "data_wait_and_augmentation_fraction": [],
        "h2d_transfer_fraction": [],
        "forward_loss_fraction": [],
        "backward_optimizer_fraction": [],
        "metrics_bookkeeping_fraction": [],
        "num_batches": [],
        "num_optimizer_steps": [],
        "grad_accum_steps": [],
    }
    history = existing_history if existing_history is not None else default_history
    for key, value in default_history.items():
        history.setdefault(key, value)

    if start_epoch >= cfg.epochs:
        rank0_print(
            f"Resume checkpoint already reached epoch {start_epoch}; cfg.epochs={cfg.epochs}. Nothing to train."
        )
        return history

    for epoch in range(start_epoch, cfg.epochs):
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_start_time = time.perf_counter()
        net.train()

        metric_sums = torch.zeros(13, device=device, dtype=torch.float64)
        # [optimization_loss, lejepa, invariance, sigreg,
        #  raw_proj_std, raw_proj_norm, raw_proj_erank,
        #  loss_proj_std, loss_proj_norm, loss_proj_erank,
        #  emb_std, emb_norm, emb_erank]
        count_tensor = torch.zeros(3, device=device, dtype=torch.float64)
        # [batch_count, diagnostic_count, optimizer_step_count]

        timing_sums = {
            "data_wait_and_augmentation_time_sec": 0.0,
            "h2d_transfer_time_sec": 0.0,
            "forward_loss_time_sec": 0.0,
            "backward_optimizer_time_sec": 0.0,
            "metrics_bookkeeping_time_sec": 0.0,
        }

        optimizer.zero_grad(set_to_none=True)
        loader_iter = iter(loader)
        fetch_start = time.perf_counter()
        pbar = tqdm(
            range(len(loader)), desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False, disable=not is_main_process()
        )

        for batch_idx in pbar:
            try:
                views, _labels = next(loader_iter)
            except StopIteration:
                break

            fetch_end = time.perf_counter()
            timing_sums["data_wait_and_augmentation_time_sec"] += fetch_end - fetch_start

            t0 = time.perf_counter()
            views = views.to(device, non_blocking=True)
            maybe_cuda_synchronize(device, cfg)
            timing_sums["h2d_transfer_time_sec"] += time.perf_counter() - t0

            sync_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader))
            no_sync_ctx = net.no_sync() if (isinstance(net, DDP) and not sync_step) else contextlib.nullcontext()

            t0 = time.perf_counter()
            with no_sync_ctx:
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                    emb, proj = net(views)

                # Keep SSL objective in FP32.
                lejepa_loss, inv, sig = loss_fn(proj.float())
                loss = lejepa_loss
                (loss / accum_steps).backward()

            maybe_cuda_synchronize(device, cfg)
            timing_sums["forward_loss_time_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if sync_step:
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                count_tensor[2] += 1.0
            maybe_cuda_synchronize(device, cfg)
            timing_sums["backward_optimizer_time_sec"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.no_grad():
                metric_sums[0] += loss.detach().double()
                metric_sums[1] += lejepa_loss.detach().double()
                metric_sums[2] += inv.detach().double()
                metric_sums[3] += sig.detach().double()
                count_tensor[0] += 1.0

                if (batch_idx % diagnostic_every == 0) or (batch_idx + 1 == len(loader)):
                    raw_std, raw_norm, raw_erank = representation_stats(proj)
                    proj_loss_diag = loss_fn.normalize_for_diagnostics(proj.detach().float())
                    loss_std, loss_norm, loss_erank = representation_stats(proj_loss_diag)
                    emb_std, emb_norm, emb_erank = representation_stats(emb)

                    metric_sums[4] += raw_std.double()
                    metric_sums[5] += raw_norm.double()
                    metric_sums[6] += raw_erank.double()
                    metric_sums[7] += loss_std.double()
                    metric_sums[8] += loss_norm.double()
                    metric_sums[9] += loss_erank.double()
                    metric_sums[10] += emb_std.double()
                    metric_sums[11] += emb_norm.double()
                    metric_sums[12] += emb_erank.double()
                    count_tensor[1] += 1.0
            timing_sums["metrics_bookkeeping_time_sec"] += time.perf_counter() - t0

            fetch_start = time.perf_counter()

        reduce_sum_tensor(metric_sums)
        reduce_sum_tensor(count_tensor)

        global_nb = max(1.0, float(count_tensor[0].item()))
        global_diag_nb = max(1.0, float(count_tensor[1].item()))
        global_opt_steps = int(count_tensor[2].item())

        history["optimization_loss"].append(float(metric_sums[0].item() / global_nb))
        history["lejepa"].append(float(metric_sums[1].item() / global_nb))
        history["invariance"].append(float(metric_sums[2].item() / global_nb))
        history["sigreg"].append(float(metric_sums[3].item() / global_nb))
        history["sigreg_mode"].append(str(cfg.sigreg_mode))

        raw_proj_std = float(metric_sums[4].item() / global_diag_nb)
        raw_proj_norm = float(metric_sums[5].item() / global_diag_nb)
        raw_proj_erank = float(metric_sums[6].item() / global_diag_nb)
        loss_proj_std = float(metric_sums[7].item() / global_diag_nb)
        loss_proj_norm = float(metric_sums[8].item() / global_diag_nb)
        loss_proj_erank = float(metric_sums[9].item() / global_diag_nb)
        emb_std = float(metric_sums[10].item() / global_diag_nb)
        emb_norm = float(metric_sums[11].item() / global_diag_nb)
        emb_erank = float(metric_sums[12].item() / global_diag_nb)

        # Compatibility aliases: proj_* refers to raw projector output, as in v4-v6.
        history["proj_std"].append(raw_proj_std)
        history["proj_norm"].append(raw_proj_norm)
        history["raw_proj_std"].append(raw_proj_std)
        history["raw_proj_norm"].append(raw_proj_norm)
        history["raw_proj_effective_rank"].append(raw_proj_erank)
        history["loss_proj_std"].append(loss_proj_std)
        history["loss_proj_norm"].append(loss_proj_norm)
        history["loss_proj_effective_rank"].append(loss_proj_erank)
        history["emb_std"].append(emb_std)
        history["emb_norm"].append(emb_norm)
        history["emb_effective_rank"].append(emb_erank)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        epoch_time = float(time.perf_counter() - epoch_start_time)
        local_batches = max(1, int(len(loader)))
        history["epoch_time_sec"].append(epoch_time)
        history["num_batches"].append(int(global_nb))
        history["num_optimizer_steps"].append(global_opt_steps)
        history["grad_accum_steps"].append(accum_steps)

        for k, v in timing_sums.items():
            v = float(v)
            history[k].append(v)
            history[k.replace("_time_sec", "_time_per_batch_sec")].append(v / max(1, local_batches))

        denom = max(epoch_time, 1e-12)
        history["data_wait_and_augmentation_fraction"].append(
            float(timing_sums["data_wait_and_augmentation_time_sec"] / denom)
        )
        history["h2d_transfer_fraction"].append(float(timing_sums["h2d_transfer_time_sec"] / denom))
        history["forward_loss_fraction"].append(float(timing_sums["forward_loss_time_sec"] / denom))
        history["backward_optimizer_fraction"].append(float(timing_sums["backward_optimizer_time_sec"] / denom))
        history["metrics_bookkeeping_fraction"].append(float(timing_sums["metrics_bookkeeping_time_sec"] / denom))

        if is_main_process():
            print(
                f"Epoch {epoch + 1:03d} | OptLoss {history['optimization_loss'][-1]:.4f} | "
                f"LeJEPA {history['lejepa'][-1]:.4f} | Inv {history['invariance'][-1]:.4f} | "
                f"SIGReg {history['sigreg'][-1]:.4f} | Mode {cfg.sigreg_mode} | "
                f"RawProjStd {history['raw_proj_std'][-1]:.4f} | LossProjStd {history['loss_proj_std'][-1]:.4f} | "
                f"RawERank {history['raw_proj_effective_rank'][-1]:.2f} | LossERank {history['loss_proj_effective_rank'][-1]:.2f} | "
                f"EmbNorm {history['emb_norm'][-1]:.2f} | EmbERank {history['emb_effective_rank'][-1]:.2f} | "
                f"LR {history['lr'][-1]:.2e} | OptSteps {global_opt_steps} | Time {epoch_time / 60:.2f} min | "
                f"Data/Aug {timing_sums['data_wait_and_augmentation_time_sec'] / 60:.2f} min | "
                f"Fwd+Loss+Bwd {timing_sums['forward_loss_time_sec'] / 60:.2f} min | "
                f"Opt {timing_sums['backward_optimizer_time_sec'] / 60:.2f} min",
                flush=True,
            )
            save_json(history, dirs["metrics"] / "training_history.json")

        epoch_num = epoch + 1
        if (
            cfg.checkpoint_every_epochs
            and cfg.checkpoint_every_epochs > 0
            and epoch_num % cfg.checkpoint_every_epochs == 0
            and epoch_num != cfg.epochs
        ):
            save_training_checkpoint(net, optimizer, scheduler, epoch_num, cfg, aug_cfg, history, dirs, final=False)
            if is_main_process():
                print(f"Saved periodic checkpoint at epoch {epoch_num}.", flush=True)

    return history


def run_training(run_config_path: str) -> Optional[tuple[Path, nn.Module]]:
    warnings.filterwarnings("ignore", category=UserWarning)
    script_start_time = time.perf_counter()

    run_cfg = load_run_config(run_config_path)
    cfg = build_config_from_run_config(run_cfg, run_config_path)
    analysis_path = run_cfg.get("paths", {}).get("analysis_config")
    if analysis_path:
        load_analysis_config(analysis_path)  # Fail on invalid settings before an expensive run.

    distributed, rank, world_size, local_rank, device = setup_distributed()
    set_seed(cfg.seed + rank)
    dirs = create_dirs(cfg.output_dir)

    try:
        with open(cfg.aug_config_path, "r", encoding="utf-8") as f:
            aug_cfg = json.load(f)

        if cfg.image_size <= 0:
            cfg.image_size = int(deep_get(aug_cfg, ["image", "output_size"], 224))
            rank0_print(f"Using image_size from augmentation config: {cfg.image_size}", flush=True)
        else:
            rank0_print(f"Using image_size from config: {cfg.image_size}", flush=True)

        if is_main_process():
            if cfg.save_resolved_config:
                save_json(asdict(cfg), dirs["root"] / "resolved_config.json")
                save_json(run_cfg, dirs["root"] / "run_config_original.json")
            save_json(aug_cfg, dirs["root"] / "augmentation_config_used.json")
            save_json(
                collect_runtime_metadata(cfg, run_cfg, device, distributed, rank, world_size, local_rank),
                dirs["root"] / "experiment_metadata.json",
            )

        timing: dict[str, Any] = {}
        rank0_print("Using device:", device)
        if device.type == "cuda":
            rank0_print("Primary GPU:", torch.cuda.get_device_name(device.index or 0))
            rank0_print("Visible CUDA devices:", torch.cuda.device_count())
            rank0_print("Distributed:", distributed, "rank/world/local_rank:", rank, world_size, local_rank)
            free, total = torch.cuda.mem_get_info(device)
            rank0_print(f"CUDA memory free/total GiB: {free / 1024**3:.2f}/{total / 1024**3:.2f}")

        t0 = time.perf_counter()
        full_df, train_df, val_df, test_df = load_splits(
            cfg.full_csv_path, cfg.train_csv_path, cfg.val_csv_path, cfg.test_csv_path
        )
        if is_main_process():
            split_metadata = {}
            for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                split_metadata[name] = {
                    "rows": int(len(df)),
                    "patients": int(df["patient"].nunique()) if "patient" in df.columns else None,
                    "collapsed_birads": {
                        str(k): int(v) for k, v in df["collapsed_birads"].value_counts().to_dict().items()
                    },
                    "dataset": {str(k): int(v) for k, v in df["dataset"].value_counts().to_dict().items()}
                    if "dataset" in df.columns
                    else {},
                    "machine_family": {str(k): int(v) for k, v in df["machine_family"].value_counts().to_dict().items()}
                    if "machine_family" in df.columns
                    else {},
                    "view": {str(k): int(v) for k, v in df["view"].value_counts().to_dict().items()}
                    if "view" in df.columns
                    else {},
                }
            save_json(split_metadata, dirs["metrics"] / "split_metadata.json")
            update_experiment_metadata(dirs, {"split_metadata": split_metadata})

        validate_bin(cfg.bin_path, len(full_df), cfg.image_height, cfg.image_width, cfg.memmap_dtype)
        timing["csv_split_loading_and_verification_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        train_transform = ConfigurableMGAugmentation(aug_cfg, cfg.image_size, train=True)
        shape = (cfg.image_height, cfg.image_width)
        n_full = len(full_df)

        train_ds = MedJEPADataset(
            train_df,
            cfg.bin_path,
            n_full,
            shape,
            cfg.memmap_dtype,
            train_transform,
            cfg.num_views,
            cfg.normalize_mode,
            cfg.percentile_low,
            cfg.percentile_high,
        )

        if distributed:
            if cfg.batch_size % world_size != 0:
                raise ValueError(
                    f"Global batch_size {cfg.batch_size} must be divisible by world_size {world_size} for DDP."
                )
            per_process_batch_size = cfg.batch_size // world_size
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
        else:
            per_process_batch_size = cfg.batch_size
            train_sampler = None

        train_loader = make_loader(
            train_ds,
            cfg,
            per_process_batch_size,
            shuffle=(not distributed),
            drop_last=True,
            sampler=train_sampler,
        )
        if len(train_loader) == 0:
            raise ValueError("No training batches: reduce batch_size or provide more training images")
        timing["dataset_and_dataloader_setup_sec"] = time.perf_counter() - t0

        rank0_print("Using natural random batch construction.", flush=True)
        rank0_print("Train batches per process:", len(train_loader))
        rank0_print("Global batch size:", cfg.batch_size)
        rank0_print("Per-process batch size:", per_process_batch_size)
        rank0_print("Gradient accumulation steps:", cfg.grad_accum_steps)
        rank0_print("Effective optimizer batch size:", cfg.batch_size * cfg.grad_accum_steps)
        rank0_print("Views per sample:", cfg.num_views)
        rank0_print("Effective augmented views per physical step:", cfg.batch_size * cfg.num_views)
        rank0_print(
            "Effective augmented views per optimizer step:", cfg.batch_size * cfg.num_views * cfg.grad_accum_steps
        )
        rank0_print("SIGReg mode:", cfg.sigreg_mode)
        rank0_print("SIGReg normalize_by_n:", cfg.sigreg_normalize_by_n)
        rank0_print("SIGReg projections:", cfg.sigreg_num_projections)

        t0 = time.perf_counter()
        net = ViTEncoder(cfg).to(device)

        # ViTEncoder may update cfg.backbone_output_dim when backbone_num_classes=0.
        # Save the post-model-setup resolved config so analysis scripts can reconstruct the model.
        if is_main_process() and cfg.save_resolved_config:
            save_json(asdict(cfg), dirs["root"] / "resolved_config.json")
            update_experiment_metadata(
                dirs,
                {
                    "resolved_train_config_after_model_setup": asdict(cfg),
                    "backbone_num_classes": int(cfg.backbone_num_classes),
                    "backbone_output_dim_actual": int(cfg.backbone_output_dim),
                },
            )
        rank0_print("Backbone timm num_classes:", int(cfg.backbone_num_classes), flush=True)
        rank0_print("Backbone embedding dim:", int(cfg.backbone_output_dim), flush=True)

        net = maybe_wrap_model(net, device, distributed, cfg.batch_size)
        loss_fn = LeJEPALoss(cfg).to(device)
        optimizer, scheduler = build_optimizer_scheduler(net, train_loader, cfg)

        start_epoch = 0
        resume_history: Optional[dict[str, list[float]]] = None
        if cfg.resume_checkpoint_path:
            ckpt_path = Path(cfg.resume_checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
            payload = read_checkpoint(ckpt_path, device)
            unwrap_model(net).load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            if "scheduler_state_dict" in payload:
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload.get("epoch", 0))
            resume_history = payload.get("history", None)
            rank0_print(f"Resumed checkpoint {ckpt_path} at completed epoch {start_epoch}.", flush=True)

        timing["model_optimizer_setup_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        history = train_lejepa(
            net,
            loss_fn,
            train_loader,
            optimizer,
            scheduler,
            cfg,
            device,
            dirs,
            aug_cfg,
            start_epoch=start_epoch,
            existing_history=resume_history,
        )
        timing["training_total_sec"] = time.perf_counter() - t0
        timing["training_epoch_time_mean_sec"] = float(np.mean(history.get("epoch_time_sec", [0.0])))
        timing["training_epoch_time_median_sec"] = float(np.median(history.get("epoch_time_sec", [0.0])))
        timing["training_epoch_time_min_sec"] = float(np.min(history.get("epoch_time_sec", [0.0])))
        timing["training_epoch_time_max_sec"] = float(np.max(history.get("epoch_time_sec", [0.0])))
        timing["timing_cuda_synchronize"] = bool(cfg.timing_cuda_synchronize)

        for key in [
            "data_wait_and_augmentation_time_sec",
            "h2d_transfer_time_sec",
            "forward_loss_time_sec",
            "backward_optimizer_time_sec",
            "metrics_bookkeeping_time_sec",
        ]:
            vals = history.get(key, [])
            if vals:
                timing[f"training_{key}_total"] = float(np.sum(vals))
                timing[f"training_{key}_mean_per_epoch"] = float(np.mean(vals))
                timing[f"training_{key}_mean_per_batch"] = float(
                    np.sum(vals) / max(1, np.sum(history.get("num_batches", [1])))
                )

        if is_main_process():
            plot_training_history(history, dirs)

        t0 = time.perf_counter()
        save_training_checkpoint(net, optimizer, scheduler, cfg.epochs, cfg, aug_cfg, history, dirs, final=True)
        timing["model_saving_sec"] = time.perf_counter() - t0
        timing["total_script_wall_time_sec"] = time.perf_counter() - script_start_time

        summary = {
            "final_lejepa_loss": history["lejepa"][-1],
            "final_invariance_loss": history["invariance"][-1],
            "final_sigreg_loss": history["sigreg"][-1],
            "final_proj_std": history["proj_std"][-1],
            "final_raw_proj_std": history["raw_proj_std"][-1],
            "final_raw_proj_effective_rank": history["raw_proj_effective_rank"][-1],
            "final_loss_proj_std": history["loss_proj_std"][-1],
            "final_loss_proj_effective_rank": history["loss_proj_effective_rank"][-1],
            "final_emb_std": history["emb_std"][-1],
            "final_emb_effective_rank": history["emb_effective_rank"][-1],
            "sigreg_mode": cfg.sigreg_mode,
            "sigreg_normalize_by_n": bool(cfg.sigreg_normalize_by_n),
            "sigreg_num_projections": int(cfg.sigreg_num_projections),
            "timing": timing,
            "distributed": {
                "enabled": bool(distributed),
                "world_size": int(world_size),
                "visible_cuda_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "used_ddp": isinstance(net, DDP),
            },
            "num_train_rows": int(len(train_df)),
            "num_val_rows": int(len(val_df)),
            "num_test_rows": int(len(test_df)),
            "output_dir": str(dirs["root"]),
        }

        if is_main_process():
            save_json(summary, dirs["metrics"] / "summary.json")
            update_experiment_metadata(dirs, {"final_summary": summary})
            print("Run finished.")
            print(json.dumps(summary, indent=2))

    finally:
        cleanup_distributed()

    if rank == 0:
        return dirs["models"] / "final_lejepa_checkpoint.pt", unwrap_model(net)
    return None
