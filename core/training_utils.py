"""Training runtime support: DDP, metadata, diagnostics, loaders and checkpoints."""

from __future__ import annotations
import datetime
import json
import os
import platform
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from .config import TrainConfig, save_json
from .data import collate_batch


def create_dirs(output_dir: str) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "models": root / "models",
        "plots": root / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def rank0_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def collect_runtime_metadata(
    cfg: TrainConfig,
    run_cfg: dict[str, Any],
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
    local_rank: int,
) -> dict[str, Any]:
    env_keys = [
        "DET_MASTER",
        "DET_WORKSPACE",
        "SLURM_JOB_ID",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    return {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "schema_version": cfg.schema_version,
        "run_name": cfg.run_name,
        "run_description": cfg.run_description,
        "run_config_path": cfg.run_config_path,
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device": str(device),
        "primary_gpu": torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else None,
        "distributed": {
            "enabled": bool(distributed),
            "rank": int(rank),
            "world_size": int(world_size),
            "local_rank": int(local_rank),
        },
        "environment": {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None},
        "compact_run_config": run_cfg,
        "resolved_train_config": asdict(cfg),
        "v7_feature_switches": {
            "backbone_num_classes": int(cfg.backbone_num_classes),
            "backbone_output_dim": int(cfg.backbone_output_dim),
            "backbone_output_dim_actual": int(cfg.backbone_output_dim),
            "sigreg_mode": cfg.sigreg_mode,
            "sigreg_normalize_by_n": bool(cfg.sigreg_normalize_by_n),
            "removed_positive_pairs": True,
            "removed_batch_construction": True,
            "removed_supervised_contrastive": True,
            "analysis_after_training": bool(run_cfg.get("paths", {}).get("analysis_config")),
        },
    }


def update_experiment_metadata(dirs: dict[str, Path], updates: dict[str, Any]) -> None:
    if not is_main_process():
        return
    path = dirs["root"] / "experiment_metadata.json"
    base: dict[str, Any] = {}
    if path.exists():
        try:
            base = json.loads(path.read_text())
        except Exception:
            base = {}
    base.update(updates)
    save_json(base, path)


def make_loader(
    dataset: Dataset,
    cfg: TrainConfig,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    if sampler is not None:
        shuffle = False
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _effective_rank(x: torch.Tensor) -> torch.Tensor:
    """Entropy effective rank of the centered feature matrix. Cheap enough for sampled diagnostics."""
    x = x.float()
    if x.ndim != 2 or x.size(0) < 2:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on [N, D] is stable for our diagnostic batches; no gradients are used.
    s = torch.linalg.svdvals(x)
    eig = s.square()
    total = eig.sum()
    if total <= 0:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    p = eig / total.clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return torch.exp(entropy)


def representation_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().float().reshape(-1, x.size(-1))
    std = x.std(dim=0).mean()
    norm = x.norm(dim=1).mean()
    erank = _effective_rank(x)
    return std, norm, erank


def maybe_cuda_synchronize(device: torch.device, cfg: TrainConfig) -> None:
    if device.type == "cuda" and cfg.timing_cuda_synchronize:
        torch.cuda.synchronize()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize DDP when launched with torchrun.

    If WORLD_SIZE > 1, v7 always uses DDP. Otherwise it uses a simple single-GPU
    or CPU fallback. There is no DataParallel/multi_gpu mode switch.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested via WORLD_SIZE>1, but CUDA is unavailable.")
        visible = torch.cuda.device_count()
        if local_rank >= visible:
            raise RuntimeError(
                f"DDP local_rank={local_rank} but only {visible} CUDA device(s) are visible. "
                "Check CUDA_VISIBLE_DEVICES and --nproc_per_node."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        return True, rank, world_size, local_rank, device

    # CUDA APIs such as mem_get_info require an explicit index on some PyTorch
    # versions. Preserve the current device rather than assuming GPU zero.
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    return False, 0, 1, 0, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def reduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def unwrap_model(net: nn.Module) -> nn.Module:
    return net.module if isinstance(net, DDP) else net


def maybe_wrap_model(net: nn.Module, device: torch.device, distributed: bool, global_batch_size: int) -> nn.Module:
    if distributed:
        rank0_print(
            f"Using DistributedDataParallel across {get_world_size()} GPUs. Global batch size={global_batch_size}.",
            flush=True,
        )
        return DDP(net, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    rank0_print(
        f"Using single process. Visible CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}.",
        flush=True,
    )
    return net


def save_training_checkpoint(
    net: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    cfg: TrainConfig,
    aug_cfg: dict[str, Any],
    history: dict[str, list[float]],
    dirs: dict[str, Path],
    final: bool = False,
) -> None:
    if not is_main_process():
        return

    model_to_save = unwrap_model(net)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(cfg),
        "augmentation_config": aug_cfg,
        "history": history,
        "script_version": "v7_refactor_from_v6",
    }

    if final:
        ckpt_path = dirs["models"] / "final_lejepa_checkpoint.pt"
        state_path = dirs["models"] / "final_lejepa_encoder_state_dict.pt"
    else:
        ckpt_path = dirs["models"] / f"checkpoint_epoch_{epoch:04d}.pt"
        state_path = dirs["models"] / f"encoder_state_dict_epoch_{epoch:04d}.pt"

    torch.save(payload, ckpt_path)
    torch.save(model_to_save.state_dict(), state_path)
