"""Run settings and JSON configuration; no ML dependencies required."""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class TrainConfig:
    """Flattened v7 config used internally by the training code."""

    # Run metadata
    schema_version: str = "medjepa_v7_config_0.1"
    run_name: str = "mg_v7_run"
    run_description: str = ""
    run_config_path: str = ""
    save_resolved_config: bool = True

    # Runtime
    seed: int = 42
    num_workers: int = 4
    output_dir: str = "outputs_medjepa_v7"

    # Data / paths
    full_csv_path: str = ""
    train_csv_path: str = ""
    val_csv_path: str = ""
    test_csv_path: str = ""
    bin_path: str = ""
    aug_config_path: str = ""
    image_height: int = 512
    image_width: int = 512
    memmap_dtype: str = "uint16"
    image_size: int = 224
    normalize_mode: str = "uint16"
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    # SSL views
    num_views: int = 4

    # Model
    backbone_name: str = "vit_small_patch8_224"
    # Passed to timm.create_model(..., num_classes=...).
    # Use 0 to remove the timm classification/head layer and use raw ViT features.
    # For vit_small_patch16_224 this usually means 384-dimensional raw features.
    backbone_num_classes: int = 512
    # Embedding dimension consumed by the projector. When backbone_num_classes=0,
    # this is inferred from the timm backbone and overwritten after model creation.
    backbone_output_dim: int = 512
    projection_dim: int = 16
    projector_hidden_dim: int = 2048
    drop_path_rate: float = 0.1

    # LeJEPA/SIGReg loss.
    lambda_sigreg: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_projections: int = 1024
    sigreg_mode: str = "author_ddp_per_view"  # author_ddp_per_view, pooled_views, or author_ddp_pooled_views
    sigreg_normalize_by_n: bool = False
    projection_normalization: str = "sqrt_dim"

    # Optimization
    epochs: int = 300
    batch_size: int = 128  # global batch size under DDP
    learning_rate: float = 1e-4
    weight_decay: float = 5e-2
    eta_min: float = 1e-5
    warmup_epochs: int = 10
    grad_clip_norm: float = 0.0
    grad_accum_steps: int = 1

    # Checkpointing and diagnostics
    checkpoint_every_epochs: int = 50
    diagnostic_every_batches: int = 50
    resume_checkpoint_path: str = ""

    # Timing
    timing_cuda_synchronize: bool = True


def build_config_from_run_config(run_cfg: dict[str, Any], run_config_path: str) -> TrainConfig:
    cfg = TrainConfig()
    cfg.schema_version = str(run_cfg.get("schema_version", cfg.schema_version))
    cfg.run_config_path = str(run_config_path)

    run = run_cfg.get("run", {})
    paths = run_cfg.get("paths", {})
    model = run_cfg.get("model", {})
    training = run_cfg.get("training", {})
    optimizer = run_cfg.get("optimizer", {})
    loss = run_cfg.get("loss", {})
    checkpointing = run_cfg.get("checkpointing", {})
    diagnostics = run_cfg.get("diagnostics", {})
    distributed = run_cfg.get("distributed", {})

    cfg.run_name = str(run.get("name", cfg.run_name))
    cfg.run_description = str(run.get("description", cfg.run_description))
    cfg.output_dir = str(run.get("output_dir", cfg.output_dir))
    cfg.seed = int(run.get("seed", cfg.seed))
    cfg.save_resolved_config = bool(run.get("save_resolved_config", cfg.save_resolved_config))

    cfg.full_csv_path = str(paths.get("full_csv", cfg.full_csv_path))
    cfg.bin_path = str(paths.get("bin", cfg.bin_path))
    cfg.train_csv_path = str(paths.get("train_csv", cfg.train_csv_path))
    cfg.val_csv_path = str(paths.get("val_csv", cfg.val_csv_path))
    cfg.test_csv_path = str(paths.get("test_csv", cfg.test_csv_path))
    cfg.aug_config_path = str(paths.get("augmentation_config", cfg.aug_config_path))

    cfg.image_height = int(model.get("image_height", cfg.image_height))
    cfg.image_width = int(model.get("image_width", cfg.image_width))
    cfg.memmap_dtype = str(model.get("memmap_dtype", cfg.memmap_dtype))
    cfg.normalize_mode = str(model.get("normalize_mode", cfg.normalize_mode))
    cfg.percentile_low = float(model.get("percentile_low", cfg.percentile_low))
    cfg.percentile_high = float(model.get("percentile_high", cfg.percentile_high))

    cfg.backbone_name = str(model.get("backbone_name", cfg.backbone_name))
    cfg.image_size = int(model.get("image_size", cfg.image_size))
    cfg.backbone_output_dim = int(model.get("backbone_output_dim", cfg.backbone_output_dim))
    # Backward compatible default: if not specified, use backbone_output_dim as timm num_classes,
    # preserving the old v6/v7 behavior.
    cfg.backbone_num_classes = int(model.get("backbone_num_classes", model.get("num_classes", cfg.backbone_output_dim)))
    cfg.projection_dim = int(model.get("projection_dim", cfg.projection_dim))
    cfg.projector_hidden_dim = int(model.get("projector_hidden_dim", cfg.projector_hidden_dim))
    cfg.drop_path_rate = float(model.get("drop_path_rate", cfg.drop_path_rate))

    cfg.epochs = int(training.get("epochs", cfg.epochs))
    cfg.batch_size = int(training.get("batch_size", cfg.batch_size))
    cfg.num_views = int(training.get("num_views", cfg.num_views))
    cfg.num_workers = int(training.get("num_workers", cfg.num_workers))
    cfg.grad_accum_steps = max(1, int(training.get("grad_accum_steps", cfg.grad_accum_steps)))

    cfg.learning_rate = float(optimizer.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(optimizer.get("weight_decay", cfg.weight_decay))
    cfg.warmup_epochs = int(optimizer.get("warmup_epochs", cfg.warmup_epochs))
    cfg.eta_min = float(optimizer.get("min_learning_rate", optimizer.get("eta_min", cfg.eta_min)))
    cfg.grad_clip_norm = float(optimizer.get("grad_clip_norm", cfg.grad_clip_norm))

    cfg.lambda_sigreg = float(loss.get("lambda_sigreg", cfg.lambda_sigreg))
    cfg.sigreg_knots = int(loss.get("sigreg_knots", cfg.sigreg_knots))
    cfg.sigreg_num_projections = int(loss.get("sigreg_num_projections", cfg.sigreg_num_projections))
    cfg.sigreg_mode = str(loss.get("sigreg_mode", cfg.sigreg_mode))
    cfg.sigreg_normalize_by_n = bool(loss.get("sigreg_normalize_by_n", cfg.sigreg_normalize_by_n))
    cfg.projection_normalization = str(loss.get("projection_normalization", cfg.projection_normalization))

    if cfg.sigreg_mode not in {"author_ddp_per_view", "pooled_views", "author_ddp_pooled_views"}:
        raise ValueError(f"Unknown loss.sigreg_mode: {cfg.sigreg_mode}")
    if cfg.projection_normalization not in {"none", "unit", "sqrt_dim"}:
        raise ValueError(f"Unknown loss.projection_normalization: {cfg.projection_normalization}")

    cfg.checkpoint_every_epochs = int(checkpointing.get("checkpoint_every_epochs", cfg.checkpoint_every_epochs))
    cfg.resume_checkpoint_path = str(checkpointing.get("resume_checkpoint", cfg.resume_checkpoint_path) or "")
    cfg.diagnostic_every_batches = max(
        1, int(diagnostics.get("diagnostic_every_batches", cfg.diagnostic_every_batches))
    )
    cfg.timing_cuda_synchronize = not bool(distributed.get("no_cuda_timing_sync", not cfg.timing_cuda_synchronize))

    required_paths = {
        "paths.full_csv": cfg.full_csv_path,
        "paths.bin": cfg.bin_path,
        "paths.train_csv": cfg.train_csv_path,
        "paths.val_csv": cfg.val_csv_path,
        "paths.test_csv": cfg.test_csv_path,
        "paths.augmentation_config": cfg.aug_config_path,
        "run.output_dir": cfg.output_dir,
    }
    missing = [k for k, v in required_paths.items() if not v]
    if missing:
        raise ValueError(f"Run config is missing required fields: {missing}")

    if cfg.epochs < 1 or cfg.batch_size < 2 or cfg.num_views < 2 or cfg.num_workers < 0:
        raise ValueError("Training requires epochs >= 1, batch_size >= 2, num_views >= 2 and num_workers >= 0")
    if cfg.sigreg_knots < 2 or cfg.sigreg_num_projections < 1:
        raise ValueError("SIGReg requires at least two knots and one projection")
    if not 0 <= cfg.lambda_sigreg <= 1:
        raise ValueError("lambda_sigreg must be between zero and one")

    return cfg


@dataclass
class ExperimentConfig:
    checkpoint: str
    full_csv: str
    train_csv: str
    val_csv: str
    test_csv: str
    bin_path: str
    aug_config: str
    output_dir: str

    backbone: str = "vit_small_patch8_224"
    image_size: int = 224
    embedding_dim: int = 512
    backbone_num_classes: Optional[int] = None
    auto_model_from_checkpoint: bool = True
    drop_path_rate: float = 0.1
    image_height: int = 512
    image_width: int = 512
    memmap_dtype: str = "uint16"
    normalize_mode: str = "uint16"
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    subset_sizes: tuple[str, ...] = ("2000", "5000", "10000", "25000", "50000", "full")
    subset_strategy: str = "progressive"
    balance_curve: str = "log"
    modes: tuple[str, ...] = ("random", "frozen_jepa", "finetune_jepa")

    epochs: int = 100
    patience: int = 15
    head_warmup_epochs: int = 5
    random_min_epoch_fraction: float = 0.5
    batch_size: int = 64
    eval_batch_size: int = 256
    num_workers: int = 4

    random_learning_rate: float = 3e-4
    random_warmup_epochs: int = 5
    random_warmup_start_factor: float = 0.1
    head_learning_rate: float = 3e-4
    finetune_head_learning_rate: float = 1e-4
    backbone_learning_rate: float = 3e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.0
    label_smoothing: float = 0.0
    use_class_weights: bool = True
    selection_metric: str = "balanced_accuracy"

    supervised_aug_mode: str = "mild"
    hflip_p: float = 0.5
    max_rotation_deg: float = 5.0

    seed: int = 42
    num_seeds: int = 1
    device: str = "cuda"
    amp: bool = True
    save_train_used_csv: bool = True
    save_run_plots: bool = True


def deep_get(dct: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_path(value: str, config_path: str | Path) -> str:
    if not value:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (Path(config_path).resolve().parent / path).resolve())


def load_run_config(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    config["paths"] = {
        key: resolve_path(value, path) if value else value for key, value in config.get("paths", {}).items()
    }
    run = config.setdefault("run", {})
    if run.get("output_dir"):
        run["output_dir"] = resolve_path(run["output_dir"], path)
    checkpointing = config.get("checkpointing", {})
    if checkpointing.get("resume_checkpoint"):
        checkpointing["resume_checkpoint"] = resolve_path(checkpointing["resume_checkpoint"], path)
    return config


@dataclass
class PCAConfig:
    enabled: bool = True
    split: str = "test"
    max_samples: int = 5000
    sampling: str = "balanced_collapsed"
    max_categories: int = 12
    include_raw_columns: bool = False
    color_by: Optional[list[str]] = None


@dataclass
class ProbeConfig:
    enabled: bool = True
    targets: tuple[str, ...] = ("dataset", "machine", "view", "laterality", "collapsed_birads")
    probe_epochs: int = 50
    probe_batch_size: int = 1024
    probe_learning_rate: float = 1e-3
    probe_weight_decay: float = 1e-7
    probe_train_max_samples: int = 60000
    no_class_weights: bool = False
    select_best_by: str = "val_macro_f1"
    machine_min_train_count: int = 20
    machine_top_k: int = 0


@dataclass
class FeatureConfig:
    batch_size: int = 256
    num_workers: int = 4
    save_features: bool = False


@dataclass
class AnalysisConfig:
    seed: int
    feature_extraction: FeatureConfig
    pca: PCAConfig
    probe: ProbeConfig


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    raw = read_json(path)
    unknown = set(raw) - {"seed", "feature_extraction", "pca", "probe"}
    if unknown:
        raise ValueError(f"Unknown analysis settings: {sorted(unknown)}")
    cfg = AnalysisConfig(
        int(raw.get("seed", 42)),
        FeatureConfig(**raw.get("feature_extraction", {})),
        PCAConfig(**raw.get("pca", {})),
        ProbeConfig(**raw.get("probe", {})),
    )
    if cfg.feature_extraction.batch_size < 1 or cfg.feature_extraction.num_workers < 0:
        raise ValueError("Feature batch_size must be positive and num_workers nonnegative")
    if cfg.pca.split not in {"train", "val", "test", "all"}:
        raise ValueError("PCA split must be train, val, test or all")
    if cfg.pca.sampling not in {"random", "balanced_collapsed"}:
        raise ValueError("Unknown PCA sampling policy")
    if cfg.pca.max_categories < 1 or (0 < cfg.pca.max_samples < 3):
        raise ValueError("PCA requires at least three samples and a positive category limit")
    if cfg.probe.probe_epochs < 1 or cfg.probe.probe_batch_size < 1:
        raise ValueError("Probe epochs and batch size must be positive")
    if cfg.probe.select_best_by not in {"val_macro_f1", "val_balanced_accuracy", "last"}:
        raise ValueError("Unknown probe selection metric")
    return cfg


def parse_transfer_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedJEPA supervised transfer / label-efficiency experiment V2.")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--full-csv", required=True, type=str)
    p.add_argument("--train-csv", required=True, type=str)
    p.add_argument("--val-csv", required=True, type=str)
    p.add_argument("--test-csv", required=True, type=str)
    p.add_argument("--bin", required=True, dest="bin_path", type=str)
    p.add_argument("--aug-config", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)

    p.add_argument("--backbone", default="vit_small_patch8_224", type=str)
    p.add_argument("--image-size", default=224, type=int)
    p.add_argument("--embedding-dim", default=512, type=int)
    p.add_argument("--backbone-num-classes", default=None, type=int)
    p.add_argument(
        "--auto-model-from-checkpoint",
        dest="auto_model_from_checkpoint",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-auto-model-from-checkpoint",
        dest="auto_model_from_checkpoint",
        action="store_false",
    )
    p.add_argument("--drop-path-rate", default=0.1, type=float)
    p.add_argument("--image-height", default=512, type=int)
    p.add_argument("--image-width", default=512, type=int)
    p.add_argument("--memmap-dtype", default="uint16", type=str)
    p.add_argument(
        "--normalize-mode",
        default="uint16",
        choices=["uint16", "per_image_percentile"],
    )
    p.add_argument("--percentile-low", default=1.0, type=float)
    p.add_argument("--percentile-high", default=99.0, type=float)

    p.add_argument(
        "--subset-sizes",
        nargs="+",
        required=True,
        help="Label budgets, e.g. 2000 5000 10k 50k full.",
    )
    p.add_argument(
        "--subset-strategy",
        default="progressive",
        choices=["progressive", "balanced", "natural"],
        help=(
            "progressive: V2 nested schedule from balanced small subsets to natural full data; "
            "balanced: as balanced as uniquely feasible; natural: natural class proportions."
        ),
    )
    p.add_argument(
        "--balance-curve",
        default="log",
        choices=["log", "linear"],
        help="How progressive balance degree falls from 1 to 0 with growing budget.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["random", "frozen_jepa", "finetune_jepa"],
        choices=["random", "frozen_jepa", "finetune_jepa"],
    )

    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--patience", default=15, type=int)
    p.add_argument("--head-warmup-epochs", default=5, type=int)
    p.add_argument(
        "--random-min-epoch-fraction",
        default=0.5,
        type=float,
        help="Random mode cannot early-stop before ceil(epochs * this fraction).",
    )
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--eval-batch-size", default=256, type=int)
    p.add_argument("--num-workers", default=4, type=int)

    p.add_argument("--random-learning-rate", default=3e-4, type=float)
    p.add_argument("--random-warmup-epochs", default=5, type=int)
    p.add_argument("--random-warmup-start-factor", default=0.1, type=float)
    p.add_argument("--head-learning-rate", default=3e-4, type=float)
    p.add_argument("--finetune-head-learning-rate", default=1e-4, type=float)
    p.add_argument("--backbone-learning-rate", default=3e-5, type=float)
    p.add_argument("--min-learning-rate", default=1e-6, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--grad-clip-norm", default=0.0, type=float)
    p.add_argument("--label-smoothing", default=0.0, type=float)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument(
        "--selection-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "macro_f1"],
        help="Validation metric used to select/checkpoint the best epoch.",
    )

    p.add_argument(
        "--supervised-aug-mode",
        default="mild",
        choices=["config", "mild", "none"],
    )
    p.add_argument("--hflip-p", default=0.5, type=float)
    p.add_argument("--max-rotation-deg", default=5.0, type=float)

    p.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Base seed. With --num-seeds N, V2 uses seed, seed+1, ..., seed+N-1.",
    )
    p.add_argument(
        "--num-seeds",
        default=1,
        type=int,
        help="Number of repeated seeds. Default 1 keeps V2 runtime comparable to V1.",
    )
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-save-train-used-csv", action="store_true")
    p.add_argument("--no-run-plots", action="store_true")
    p.add_argument("--run-config", help="Transfer JSON; explicit CLI flags override its values")
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--run-config")
    known, _ = preliminary.parse_known_args()
    if known.run_config:
        values = read_json(known.run_config)
        valid = {field.name for field in fields(ExperimentConfig)}
        if set(values) - valid:
            raise ValueError(f"Unknown transfer settings: {sorted(set(values) - valid)}")
        for key in (
            "checkpoint",
            "full_csv",
            "train_csv",
            "val_csv",
            "test_csv",
            "bin_path",
            "aug_config",
            "output_dir",
        ):
            if values.get(key):
                values[key] = resolve_path(values[key], known.run_config)
        for positive, negative in TRANSFER_FLAGS.items():
            if positive in values:
                values[negative] = not values.pop(positive)
        p.set_defaults(**values)
        for action in p._actions:
            if action.dest in values:
                action.required = False
    return p.parse_args()


TRANSFER_FLAGS = {
    "use_class_weights": "no_class_weights",
    "amp": "no_amp",
    "save_train_used_csv": "no_save_train_used_csv",
    "save_run_plots": "no_run_plots",
}


def parse_transfer_config() -> ExperimentConfig:
    values = vars(parse_transfer_args())
    values.pop("run_config")
    for positive, negative in TRANSFER_FLAGS.items():
        values[positive] = not values.pop(negative)
    for key in ("subset_sizes", "modes"):
        values[key] = tuple(values[key])
    return ExperimentConfig(**values)
