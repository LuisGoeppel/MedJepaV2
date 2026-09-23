from collections import Counter
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
import torch

from core.config import ExperimentConfig, build_config_from_run_config, load_run_config
from core.models import ViTEncoder
from run_transfer_experiment import build_subset_plan, run_transfer


def test_oversampling_exact_budgets_and_nested_repeats():
    frame = pd.DataFrame({"original_index": range(36), "target_collapsed": [0] * 30 + [1] * 4 + [2] * 2})
    budgets = ("5", "12", "25", "full")
    plan = build_subset_plan(frame, budgets, "oversampling", "log", 17)
    again = build_subset_plan(frame, budgets, "oversampling", "linear", 17)
    previous = Counter()
    for key, size in zip(budgets, (5, 12, 25, 36)):
        spec = plan[key]
        selected = spec["df"].original_index.tolist()
        assert len(selected) == size
        assert selected == again[key]["df"].original_index.tolist()
        counts = list(spec["actual_counts"].values())
        assert max(counts) - min(counts) <= 1
        current = Counter(selected)
        assert all(current[k] >= v for k, v in previous.items())
        assert spec["unique_rows"] == len(current)
        assert spec["repeated_rows"] == size - len(current)
        assert sum(spec["unique_class_counts"].values()) == len(current)
        for cls, capacity in zip(("routine", "follow_up", "biopsy"), (30, 4, 2)):
            assert spec["unique_class_counts"][cls] == min(spec["actual_counts"][cls], capacity)
        previous = current
    assert plan["full"]["unique_class_counts"] == {"routine": 12, "follow_up": 4, "biopsy": 2}
    assert plan["full"]["repeated_rows"] == 18
    # Original policies retain unique images and the natural full distribution.
    for strategy in ("progressive", "balanced", "natural"):
        old = build_subset_plan(frame, budgets, strategy, "log", 17)
        assert all(s["repeated_rows"] == 0 for s in old.values())
        assert old["full"]["actual_counts"] == {"routine": 30, "follow_up": 4, "biopsy": 2}


def test_oversampling_missing_class_fails():
    frame = pd.DataFrame({"original_index": range(4), "target_collapsed": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="every class"):
        build_subset_plan(frame, ("full",), "oversampling", "log", 1)


def test_oversampling_transfer_all_modes(tiny_run):
    root = tiny_run.parent
    raw = load_run_config(tiny_run)
    cfg = build_config_from_run_config(raw, str(tiny_run))
    model = ViTEncoder(cfg)
    checkpoint = root / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "augmentation_config": {}}, checkpoint)
    train = pd.read_csv(root / "train.csv").iloc[[0, 3, 6, 9, 1, 2]]
    train.to_csv(root / "train.csv", index=False)
    paths = raw["paths"]
    experiment = ExperimentConfig(
        checkpoint=str(checkpoint), full_csv=paths["full_csv"], train_csv=paths["train_csv"],
        val_csv=paths["val_csv"], test_csv=paths["test_csv"], bin_path=paths["bin"],
        aug_config=paths["augmentation_config"], output_dir=str(root / "oversampling"),
        subset_strategy="oversampling", subset_sizes=("full",), image_height=32, image_width=32,
        epochs=1, patience=1, head_warmup_epochs=0, random_warmup_epochs=0,
        num_workers=0, batch_size=3, eval_batch_size=3, device="cpu", amp=False, seed=7,
    )
    output = root / "oversampling"
    output.mkdir()
    for name in ("summary_balance_degree.png", "summary_subset_class_composition.png"):
        (output / name).write_bytes(b"obsolete plot")
    run_transfer(experiment)
    summary = pd.read_csv(output / "summary_table.csv")
    assert set(summary["mode"]) == {"random", "frozen_jepa", "finetune_jepa"}
    assert (summary.train_used_rows == 6).all()
    assert (summary.train_unique_rows == 4).all()
    assert (summary.train_repeated_rows == 2).all()
    np.testing.assert_array_equal(summary[["train_unique_routine", "train_unique_follow_up", "train_unique_biopsy"]], [[2, 1, 1]] * 3)
    assert (output / "summary_sampling_unique_images.png").is_file()
    assert not (output / "summary_balance_degree.png").exists()
    assert not (output / "summary_subset_class_composition.png").exists()
    for result in json.loads((output / "summary_results.json").read_text()):
        assert result["train_unique_rows"] == 4
