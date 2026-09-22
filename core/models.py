"""One encoder definition for pretraining, transfer and checkpoint analysis."""

from __future__ import annotations
import math
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional
import torch
from torch import nn
import timm
from .config import TrainConfig, ExperimentConfig


class ViTEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()

        backbone_num_classes = int(getattr(cfg, "backbone_num_classes", cfg.backbone_output_dim))

        self.backbone = build_backbone(cfg.backbone_name, cfg.image_size, backbone_num_classes, cfg.drop_path_rate)

        if backbone_num_classes == 0:
            # timm returns raw backbone features when num_classes=0.
            actual_dim = int(getattr(self.backbone, "num_features", 0))
            if actual_dim <= 0:
                raise RuntimeError(
                    f"Could not infer raw feature dimension for {cfg.backbone_name}. "
                    "Set model.backbone_output_dim explicitly and extend ViTEncoder if this timm model "
                    "does not expose backbone.num_features."
                )
        else:
            # In normal use backbone_num_classes == backbone_output_dim.
            actual_dim = int(backbone_num_classes)
            if int(cfg.backbone_output_dim) != actual_dim:
                print(
                    "WARNING: model.backbone_num_classes and model.backbone_output_dim differ. "
                    f"Using actual embedding dim={actual_dim} from backbone_num_classes={backbone_num_classes}.",
                    flush=True,
                )

        self.embedding_dim = actual_dim
        cfg.backbone_output_dim = actual_dim

        self.proj = build_projector(actual_dim, cfg.projector_hidden_dim, cfg.projection_dim)

    def encode_one(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        return embedding, self.proj(embedding)

    # Batch-first projection shape is required for correct torch.nn.DataParallel gathering.
    # Shape: [B, V, D]. Older versions returned [V, B, D], which breaks when
    # DataParallel sees unequal per-GPU batch sizes and also gathers along the wrong axis.
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, v = x.shape[:2]
        flat = x.flatten(0, 1)
        emb = self.backbone(flat)
        proj = self.proj(emb).reshape(b, v, -1)
        return emb, proj


def build_projector(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class ViTBackboneClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        embedding_dim: int,
        drop_path_rate: float,
        num_classes: int = 3,
        backbone_num_classes: Optional[int] = None,
    ):
        super().__init__()
        timm_num_classes = int(embedding_dim if backbone_num_classes is None else backbone_num_classes)
        self.backbone_num_classes = timm_num_classes
        self.backbone = build_backbone(backbone_name, image_size, timm_num_classes, drop_path_rate)
        if timm_num_classes == 0:
            actual_embedding_dim = int(getattr(self.backbone, "num_features", 0))
            if actual_embedding_dim <= 0:
                raise RuntimeError(f"Could not infer num_features for raw backbone {backbone_name}.")
        else:
            actual_embedding_dim = timm_num_classes
        self.embedding_dim = actual_embedding_dim
        self.head = nn.Sequential(
            nn.LayerNorm(actual_embedding_dim),
            nn.Linear(actual_embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a model state to CPU so the same single model object can be reset safely.

    This avoids repeatedly calling timm.create_model() between experiment runs.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_initial_model_state(
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
) -> None:
    """Restore the exact initial random weights before a new transfer run."""
    model.load_state_dict(initial_state, strict=True)
    model.zero_grad(set_to_none=True)


def strip_module_prefix(k: str) -> str:
    return k[len("module.") :] if k.startswith("module.") else k


def extract_backbone_state(payload: Any) -> tuple[dict[str, torch.Tensor], int, dict[str, torch.Tensor]]:
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("Could not find a state dict in checkpoint payload.")
    backbone_state: dict[str, torch.Tensor] = {}
    full_backbone_state: dict[str, torch.Tensor] = {}
    ignored_projector = 0
    for k, v in state.items():
        k = strip_module_prefix(str(k))
        if k.startswith("backbone."):
            kk = k[len("backbone.") :]
            backbone_state[kk] = v
            full_backbone_state[kk] = v
        elif k.startswith("proj."):
            ignored_projector += 1
    if not backbone_state:
        raise ValueError("No backbone.* keys found in checkpoint. Expected v6/v7 LeJEPA checkpoint.")
    return backbone_state, ignored_projector, full_backbone_state


def _as_dict_maybe(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        return dict(vars(x))
    return {}


def model_hints(payload: dict[str, Any]) -> dict[str, Any]:
    backbone_state, ignored_projector, _ = extract_backbone_state(payload)
    ckpt_cfg = _as_dict_maybe(payload.get("config") if isinstance(payload, dict) else {})

    hints: dict[str, Any] = {
        "ignored_projector_keys": int(ignored_projector),
    }

    for key in ["backbone_name", "image_size", "backbone_output_dim", "backbone_num_classes"]:
        if key in ckpt_cfg and ckpt_cfg[key] is not None:
            hints[key] = ckpt_cfg[key]

    patch_w = backbone_state.get("patch_embed.proj.weight")
    if isinstance(patch_w, torch.Tensor) and patch_w.ndim == 4:
        hints["patch_size"] = int(patch_w.shape[-1])
        hints["vit_feature_dim"] = int(patch_w.shape[0])

    pos = backbone_state.get("pos_embed")
    if isinstance(pos, torch.Tensor) and pos.ndim == 3 and "patch_size" in hints:
        n_patches = int(pos.shape[1]) - 1
        grid = int(round(math.sqrt(max(1, n_patches))))
        if grid * grid == n_patches and "image_size" not in hints:
            hints["image_size"] = int(grid * int(hints["patch_size"]))

    head_w = backbone_state.get("head.weight")
    if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
        hints["backbone_num_classes"] = int(head_w.shape[0])
        hints["backbone_output_dim"] = int(head_w.shape[0])
        hints["vit_feature_dim"] = int(head_w.shape[1])
    else:
        if "backbone_num_classes" not in hints:
            hints["backbone_num_classes"] = 0
        if "backbone_output_dim" not in hints and "vit_feature_dim" in hints:
            hints["backbone_output_dim"] = int(hints["vit_feature_dim"])

    return hints


def infer_model_hints_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    return {"checkpoint": str(checkpoint_path), **model_hints(read_checkpoint(checkpoint_path))}


def maybe_update_config_from_checkpoint(cfg: ExperimentConfig) -> dict[str, Any]:
    hints = infer_model_hints_from_checkpoint(cfg.checkpoint)

    if "backbone_name" in hints:
        cfg.backbone = str(hints["backbone_name"])
    elif "patch_size" in hints:
        ps = int(hints["patch_size"])
        if re.search(r"patch\d+", cfg.backbone):
            cfg.backbone = re.sub(r"patch\d+", f"patch{ps}", cfg.backbone)

    if "image_size" in hints:
        cfg.image_size = int(hints["image_size"])
    if "backbone_num_classes" in hints:
        cfg.backbone_num_classes = int(hints["backbone_num_classes"])
    if "backbone_output_dim" in hints:
        cfg.embedding_dim = int(hints["backbone_output_dim"])

    return hints


def load_jepa_backbone(
    model: ViTBackboneClassifier,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    payload = read_checkpoint(checkpoint_path, device)
    backbone_state, ignored_projector, _ = extract_backbone_state(payload)

    try:
        load_info = model.backbone.load_state_dict(backbone_state, strict=True)
    except RuntimeError as exc:
        hints = infer_model_hints_from_checkpoint(checkpoint_path)
        raise RuntimeError(
            "Failed to load JEPA backbone. This is usually a model mismatch.\n"
            f"Checkpoint hints: {json.dumps(hints, indent=2, default=str)}\n"
            f"Current model: backbone={model.backbone.__class__.__name__}, "
            f"backbone_num_classes={model.backbone_num_classes}, embedding_dim={model.embedding_dim}.\n"
            "Check --backbone, --image-size, --embedding-dim and --backbone-num-classes, "
            "or leave --auto-model-from-checkpoint enabled.\n"
            f"Original error: {exc}"
        ) from exc

    return {
        "checkpoint": str(checkpoint_path),
        "loaded_backbone_keys": int(len(backbone_state)),
        "ignored_projector_keys": int(ignored_projector),
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
        "checkpoint_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "checkpoint_config": payload.get("config") if isinstance(payload, dict) else None,
        "model_backbone_num_classes": int(model.backbone_num_classes),
        "model_embedding_dim": int(model.embedding_dim),
    }


def set_backbone_trainable(model: ViTBackboneClassifier, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)


def build_backbone(name: str, image_size: int, num_classes: int, drop_path_rate: float) -> nn.Module:
    return timm.create_model(
        name, pretrained=False, num_classes=num_classes, drop_path_rate=drop_path_rate, img_size=image_size, in_chans=1
    )


def read_checkpoint(path: str | Path, device="cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Expected a training checkpoint containing model_state_dict and config")
    return payload


def load_encoder(
    path: str | Path, device: torch.device, model: Optional[ViTEncoder] = None
) -> tuple[ViTEncoder, TrainConfig, dict[str, Any]]:
    payload = read_checkpoint(path)
    raw = _as_dict_maybe(payload.get("config"))
    if not raw or not payload.get("augmentation_config"):
        raise ValueError("Analysis requires checkpoint config and augmentation_config")
    hints = model_hints(payload)
    allowed = {field.name for field in fields(TrainConfig)}
    cfg = TrainConfig(**{key: value for key, value in raw.items() if key in allowed})
    for key in ("backbone_name", "image_size", "backbone_output_dim", "backbone_num_classes"):
        if key in hints:
            setattr(cfg, key, hints[key])
    # Reuse the trained object for post-training reports. Re-entering timm model
    # initialization after a training run has caused native crashes on the cluster.
    if model is None:
        model = ViTEncoder(cfg)
    state = {strip_module_prefix(str(key)): value for key, value in payload["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg, payload["augmentation_config"]


def build_supervised_model(backbone, image_size, pretrained=False, num_classes=3, **kwargs):
    """Create one-channel timm classifiers; CNNs do not take img_size."""
    options = dict(pretrained=pretrained, num_classes=num_classes, in_chans=1, **kwargs)
    try:
        return timm.create_model(backbone, img_size=image_size, **options)
    except TypeError as exc:
        if "img_size" not in str(exc):
            raise
        return timm.create_model(backbone, **options)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model
