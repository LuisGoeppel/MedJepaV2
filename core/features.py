"""Extract ordered backbone embeddings once for PCA and linear probes."""

from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .config import load_analysis_config, load_run_config, save_json
from .data import EvalDataset, collate_batch, load_splits, set_seed, validate_bin
from .models import ViTEncoder, load_encoder
from .transforms import evaluation_transform


@torch.inference_mode()
def extract_features(df, bin_path, n_full, model_cfg, aug_cfg, model, device, settings, split_name):
    if df.empty:
        raise ValueError(f"Cannot extract features from empty {split_name} split")
    dataset = EvalDataset(
        df,
        bin_path,
        n_full,
        (model_cfg.image_height, model_cfg.image_width),
        model_cfg.memmap_dtype,
        evaluation_transform(aug_cfg, model_cfg.image_size),
        model_cfg.normalize_mode,
        model_cfg.percentile_low,
        model_cfg.percentile_high,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    embeddings, indices = [], []
    model.eval()
    for views, index in tqdm(loader, desc=f"Extract features [{split_name}]"):
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # Reports need the backbone only; avoid evaluating the unused projector.
            emb = model.backbone(views.to(device, non_blocking=True).flatten(0, 1))
        embeddings.append(emb.float().cpu())
        indices.append(index)
    ordered = df.iloc[torch.cat(indices).numpy()].reset_index(drop=True)
    return torch.cat(embeddings), ordered


def run_analysis(
    checkpoint: str | Path,
    run_config_path: str | Path,
    analysis_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    model: ViTEncoder | None = None,
) -> dict[str, Any]:
    """Shared steps 3–5 for the training and analysis entry points."""
    run = load_run_config(run_config_path)
    paths = run["paths"]
    config_path = analysis_config_path or paths.get("analysis_config")

    if not config_path:
        raise ValueError("Set paths.analysis_config in the training JSON or pass --analysis-config")

    settings = load_analysis_config(config_path)
    out = Path(output_dir) if output_dir else Path(run["run"]["output_dir"]) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    save_json(asdict(settings), out / "analysis_config_used.json")

    if not settings.pca.enabled and not settings.probe.enabled:
        return {"status": "disabled"}

    set_seed(settings.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, aug_cfg = load_encoder(checkpoint, device, model=model)
    full_raw, train, val, test = load_splits(paths["full_csv"], paths["train_csv"], paths["val_csv"], paths["test_csv"])

    validate_bin(paths["bin"], len(full_raw), model_cfg.image_height, model_cfg.image_width, model_cfg.memmap_dtype)
    frames = {"train": train, "val": val, "test": test}
    names = set(frames) if settings.probe.enabled else set()

    if settings.pca.enabled:
        names.update(frames if settings.pca.split == "all" else [settings.pca.split])

    features, timing = {}, {}
    for name in ("train", "val", "test"):
        if name not in names:
            continue
        frame = frames[name]
        # PCA-only runs need not extract the entire selected dataset.
        if not settings.probe.enabled and settings.pca.split != "all":
            from argparse import Namespace
            from .pca import sample_dataframe

            frame = sample_dataframe(frame, Namespace(**asdict(settings.pca), seed=settings.seed))
        start = time.perf_counter()
        features[name] = extract_features(
            frame, paths["bin"], len(full_raw), model_cfg, aug_cfg, model, device, settings.feature_extraction, name
        )
        timing[name] = time.perf_counter() - start

    model.cpu()  # Release GPU storage even if the training entry point retains the object.
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if settings.feature_extraction.save_features:
        torch.save(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "model_config": asdict(model_cfg),
                "augmentation_config": aug_cfg,
                "splits": {
                    name: {"embeddings": x, "metadata": df.to_dict("list")} for name, (x, df) in features.items()
                },
            },
            out / "features.pt",
        )

    if settings.pca.enabled:
        from .pca import run_pca_report

        run_pca_report(features, settings.pca, settings.seed, str(checkpoint), out / "pca_report.pdf")

    if settings.probe.enabled:
        from .probe import run_probe_report

        set_seed(settings.seed)
        run_probe_report(
            features, settings.probe, settings.seed, str(checkpoint), out / "linear_probe_report.json", device
        )

    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "model_config": asdict(model_cfg),
        "augmentation_config": aug_cfg,
        "data_paths": paths,
        "analysis_settings": asdict(settings),
        "feature_extraction_seconds": timing,
        "num_rows": {name: len(frame) for name, (_, frame) in features.items()},
    }

    save_json(metadata, out / "analysis_metadata.json")
    return metadata
