import json

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_run(tmp_path):
    """Real uint16 BIN, disjoint patients and an unlabeled physical row."""
    count = 25
    rng = np.random.default_rng(12)
    images = rng.integers(0, 65535, (count, 32, 32), dtype=np.uint16)
    images.tofile(tmp_path / "images.bin")
    frame = pd.DataFrame(
        {
            "id": [f"image-{i}" for i in range(count)],
            "patient": [f"patient-{i}" for i in range(count)],
            "original_birads": [1, 3, 5] * 8 + [0],
            "original_index": np.arange(count),
            "dataset": ["a", "b", "a"] * 8 + ["b"],
            "machine": "Hologic",
            "context": json.dumps({"exam": {"view": "CC", "laterality": "L"}, "patient": {"age": 52}}),
        }
    )
    frame.to_csv(tmp_path / "full.csv", index=False)
    for name, indices in {"train": slice(0, 12), "val": slice(12, 18), "test": slice(18, 24)}.items():
        frame.iloc[indices].to_csv(tmp_path / f"{name}.csv", index=False)
    aug = {"preprocessing": {"foreground_crop": {"enabled": False}}, "image": {"output_size": 32}}
    (tmp_path / "augmentation.json").write_text(json.dumps(aug))
    analysis = {
        "seed": 7,
        "feature_extraction": {"batch_size": 3, "num_workers": 0, "save_features": True},
        "pca": {"max_samples": 6, "color_by": ["collapsed_birads"]},
        "probe": {"targets": ["collapsed_birads"], "probe_epochs": 1, "probe_batch_size": 6},
    }
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    config = {
        "run": {"output_dir": "run", "seed": 7},
        "paths": {
            "full_csv": "full.csv",
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "test_csv": "test.csv",
            "bin": "images.bin",
            "augmentation_config": "augmentation.json",
            "analysis_config": "analysis.json",
        },
        "model": {
            "backbone_name": "vit_tiny_patch16_224",
            "backbone_num_classes": 0,
            "backbone_output_dim": 192,
            "image_size": 32,
            "image_height": 32,
            "image_width": 32,
            "projection_dim": 4,
            "projector_hidden_dim": 8,
            "drop_path_rate": 0,
        },
        "training": {"epochs": 1, "batch_size": 6, "num_views": 2, "num_workers": 0},
        "optimizer": {"warmup_epochs": 0},
        "loss": {"sigreg_num_projections": 4, "sigreg_knots": 3},
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(config))
    return path
