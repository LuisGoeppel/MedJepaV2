import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

import analyze_medjepa
import train_medjepa
import run_transfer_experiment
from core import features
from core.config import ExperimentConfig, load_run_config
from core.training import run_training


def test_train_analysis_resume_and_transfer(tiny_run, monkeypatch):
    """Exercise all three public workflows with a real tiny timm ViT on CPU."""
    calls = []
    from core import models

    constructions = []
    original_build = models.build_backbone

    def counted_build(*args, **kwargs):
        constructions.append(args[0])
        return original_build(*args, **kwargs)

    monkeypatch.setattr(models, "build_backbone", counted_build)
    original_extract = features.extract_features

    def counted_extract(*args, **kwargs):
        calls.append(args[-1])
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(features, "extract_features", counted_extract)
    monkeypatch.setattr(sys, "argv", ["train_medjepa.py", "--run-config", str(tiny_run)])
    train_medjepa.main()
    assert len(constructions) == 1  # Post-training analysis reuses the trained encoder.
    assert calls == ["train", "val", "test"]  # Both reports reuse these features.
    run_dir = tiny_run.parent / "run"
    checkpoint = run_dir / "models" / "final_lejepa_checkpoint.pt"
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["epoch"] == 1
    assert len(payload["history"]["lejepa"]) == 1
    assert payload["config"]["backbone_num_classes"] == 0
    assert (run_dir / "analysis" / "pca_report.pdf").read_bytes().startswith(b"%PDF")
    report = json.loads((run_dir / "analysis" / "linear_probe_report.json").read_text())
    assert report["probes"]["collapsed_birads"]["status"] == "ok"
    assert report["data"]["num_train_rows"] == 12
    saved = torch.load(run_dir / "analysis" / "features.pt", weights_only=False)
    assert saved["splits"]["test"]["metadata"]["original_index"] == list(range(18, 24))

    # Standalone analysis must leave training/checkpoint state untouched.
    before = checkpoint.read_bytes()
    separate = tiny_run.parent / "separate_analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_medjepa.py",
            "--run-config",
            str(tiny_run),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(separate),
        ],
    )
    analyze_medjepa.main()
    assert checkpoint.read_bytes() == before
    assert calls == ["train", "val", "test"] * 2
    second_report = json.loads((separate / "linear_probe_report.json").read_text())
    assert report["probes"]["collapsed_birads"]["metrics"] == second_report["probes"]["collapsed_birads"]["metrics"]

    # Resume uses optimizer/scheduler/history state, rather than starting over.
    raw = json.loads(tiny_run.read_text())
    raw["training"]["epochs"] = 2
    raw["checkpointing"] = {"resume_checkpoint": str(checkpoint)}
    tiny_run.write_text(json.dumps(raw))
    run_training(str(tiny_run))
    resumed = torch.load(checkpoint, weights_only=False)
    assert resumed["epoch"] == 2
    assert len(resumed["history"]["lejepa"]) == 2
    assert resumed["history"]["lejepa"][0] == payload["history"]["lejepa"][0]
    assert resumed["scheduler_state_dict"]["last_epoch"] > payload["scheduler_state_dict"]["last_epoch"]

    paths = load_run_config(tiny_run)["paths"]
    transfer_cfg = ExperimentConfig(
        checkpoint=str(checkpoint),
        full_csv=paths["full_csv"],
        train_csv=paths["train_csv"],
        val_csv=paths["val_csv"],
        test_csv=paths["test_csv"],
        bin_path=paths["bin"],
        aug_config=paths["augmentation_config"],
        output_dir=str(tiny_run.parent / "transfer"),
        image_height=32,
        image_width=32,
        subset_sizes=("6",),
        epochs=2,
        patience=2,
        head_warmup_epochs=1,
        random_warmup_epochs=0,
        num_workers=0,
        batch_size=3,
        eval_batch_size=3,
        device="cpu",
        amp=False,
        save_run_plots=True,
        seed=7,
    )
    transfer_path = tiny_run.parent / "transfer_config.json"
    transfer_path.write_text(json.dumps(asdict(transfer_cfg)))
    monkeypatch.setattr(sys, "argv", ["run_transfer_experiment.py", "--run-config", str(transfer_path)])
    run_transfer_experiment.main()
    summary = pd.read_csv(Path(transfer_cfg.output_dir) / "summary_table.csv")
    assert set(summary["mode"]) == {"random", "frozen_jepa", "finetune_jepa"}
    history = json.loads(
        (Path(transfer_cfg.output_dir) / "budget_6__finetune_jepa" / "training_history.json").read_text()
    )
    assert history["stage"] == ["head_warmup", "finetune"]
    expected_plots = {
        "training_history.png",
        "training_diagnostics.png",
        "projection_embedding_diagnostics.png",
        "collapse_diagnostics.png",
        "epoch_times.png",
        "epoch_timing_components.png",
        "epoch_timing_fractions.png",
    }
    assert expected_plots <= {path.name for path in (run_dir / "plots").glob("*.png")}
    expected_summary_plots = {
        "summary_test_balanced_accuracy.png",
        "summary_test_macro_f1.png",
        "summary_test_min_class_recall.png",
        "summary_test_per_class_recall.png",
        "summary_best_epoch.png",
        "summary_balance_degree.png",
        "summary_subset_class_composition.png",
        "summary_predicted_class_fractions.png",
    }
    assert expected_summary_plots <= {path.name for path in Path(transfer_cfg.output_dir).glob("*.png")}
    for mode in transfer_cfg.modes:
        directory = Path(transfer_cfg.output_dir) / f"budget_6__{mode}"
        for filename in (
            "class_distribution.png",
            "training_curves.png",
            "confusion_matrix_val_best.png",
            "confusion_matrix_test.png",
        ):
            assert (directory / filename).read_bytes().startswith(b"\x89PNG")


def test_nonzero_training_rank_does_not_run_analysis(monkeypatch):
    monkeypatch.setattr(train_medjepa, "run_training", lambda path: None)

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("Only rank zero may run post-training analysis")

    monkeypatch.setattr(features, "run_analysis", unexpected_analysis)
    monkeypatch.setattr(sys, "argv", ["train_medjepa.py", "--run-config", "unused.json"])
    train_medjepa.main()


def test_pca_only_extracts_only_requested_split(tiny_run, monkeypatch):
    # Build a checkpoint without running training; all report work remains real.
    from core.models import ViTEncoder
    from core.config import build_config_from_run_config

    run = load_run_config(tiny_run)
    cfg = build_config_from_run_config(run, str(tiny_run))
    model = ViTEncoder(cfg)
    checkpoint = tiny_run.parent / "initial.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "augmentation_config": {"image": {"output_size": 32}},
        },
        checkpoint,
    )
    path = tiny_run.parent / "analysis.json"
    analysis = json.loads(path.read_text())
    analysis["probe"]["enabled"] = False
    analysis["pca"].update(max_samples=3, sampling="random")
    path.write_text(json.dumps(analysis))
    result = features.run_analysis(checkpoint, tiny_run)
    assert result["num_rows"] == {"test": 3}
    assert not (tiny_run.parent / "run" / "analysis" / "linear_probe_report.json").exists()
