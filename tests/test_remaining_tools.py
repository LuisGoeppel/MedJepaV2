"""Exercise notebook setup and the two auxiliary root workflows on small data."""

import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

import plot_medjepa_training_history as history_cli
import run_medjepa_representation_comparison as comparison


@pytest.mark.parametrize("head", [0, 8])
def test_representation_comparison(tiny_run, monkeypatch, head):
    root = tiny_run.parent
    cfg = comparison.ModelConfig(
        backbone_name="vit_tiny_patch16_224", image_size=32, image_height=32, image_width=32,
        backbone_num_classes=head, backbone_output_dim=192 if head == 0 else head,
        projection_dim=4, projector_hidden_dim=8, drop_path_rate=0,
    )
    model = comparison.MedJEPAEncoder(cfg)
    checkpoint = root / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "augmentation_config": {}}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "comparison", "--checkpoint", str(checkpoint), "--full-csv", str(root / "full.csv"),
        "--bin", str(root / "images.bin"), "--train-csv", str(root / "train.csv"),
        "--val-csv", str(root / "val.csv"), "--test-csv", str(root / "test.csv"),
        "--output-pdf", str(root / "comparison.pdf"), "--num-workers", "0", "--batch-size", "6",
        "--probe-epochs", "1", "--probe-batch-size", "6", "--patch-epochs", "1",
        "--patch-batch-size", "6", "--mlp-hidden-dim", "8", "--pca-max-samples", "6",
    ])
    comparison.main()
    result = json.loads((root / "comparison.json").read_text())
    assert result["feature_dims"]["head512"] == (192 if head == 0 else head)
    assert set(result["metrics"]) == {"head512", "cls", "patch_cross_attention"}
    assert (root / "comparison.pdf").read_bytes().startswith(b"%PDF")
    for metrics in result["metrics"].values():
        for classifier in metrics.values():
            assert 0 <= classifier["test"]["balanced_accuracy"] <= 1
    # Incomplete checkpoints must fail instead of measuring random missing weights.
    state = model.state_dict()
    del state["proj.0.weight"]
    torch.save({"model_state_dict": state, "config": asdict(cfg)}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        comparison.load_checkpoint_and_model(checkpoint, torch.device("cpu"))


def test_history_cli_partial_history(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "training_history.json").write_text(json.dumps({
        "lejepa": [1, 0.5, 0.2], "lr": [0.01, None], "emb_std": ["bad", 0.1],
    }))
    monkeypatch.setattr(sys, "argv", ["history", "--run-dir", str(tmp_path), "--dpi", "40"])
    history_cli.main()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 7
    assert not list((tmp_path / "plots").glob("*.tmp.png"))
    assert not plt.get_fignums()


@pytest.mark.parametrize("cwd", [".", "notebooks"])
def test_notebook_with_shared_transforms(tiny_run, monkeypatch, cwd):
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "notebooks/visualize_mg_lejepa_aug_v2.ipynb").read_text(encoding="utf-8"))
    root = tiny_run.parent
    overrides = {
        "FULL_CSV": str(root / "full.csv"), "SPLIT_CSV": str(root / "train.csv"),
        "BIN_PATH": str(root / "images.bin"), "AUG_CONFIG": str(root / "augmentation.json"),
        "IMAGE_HEIGHT": 32, "IMAGE_WIDTH": 32,
    }
    monkeypatch.chdir(project / cwd)
    namespace = {}
    for cell in notebook["cells"][1:3]:
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in overrides:
                    node.value = ast.Constant(overrides[name])
        exec(compile(ast.fix_missing_locations(tree), "notebook", "exec"), namespace)
    assert namespace["train_transform"].__class__.__module__ == "core.transforms"
    row = namespace["df"].iloc[5]
    image = namespace["load_tensor_from_row"](row)
    assert image.shape == (1, 32, 32)
    namespace["plot_rows_with_augmentations"](namespace["sample_random_rows"](2), n_augs=1)
    plt.close("all")
    # All remaining cells remain valid Python after relocation.
    for cell in notebook["cells"][3:]:
        compile("".join(cell["source"]), "notebook", "exec")
