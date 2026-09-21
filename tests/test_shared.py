import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from core.config import (
    TrainConfig,
    FeatureConfig,
    build_config_from_run_config,
    load_run_config,
    parse_transfer_config,
)
from core.data import EvalDataset, ensure_original_index, load_splits, prepare_labels, validate_bin
from core.features import extract_features
from core.models import ViTEncoder, load_encoder, ViTBackboneClassifier, load_jepa_backbone
from core.loss import LeJEPALoss, AuthorDDPPooledViewsSIGReg, PooledViewsSIGReg
from run_transfer_experiment import build_subset_plan
from core.transforms import ConfigurableMGAugmentation, evaluation_transform, MildSupervisedMGAugmentation


def test_raw_rows_and_mapping_survive_label_filtering(tiny_run):
    config = load_run_config(tiny_run)
    paths = config["paths"]
    full, train, val, test = load_splits(*(paths[key] for key in ("full_csv", "train_csv", "val_csv", "test_csv")))
    assert len(full) == 25
    assert len(prepare_labels(full)) == 24
    validate_bin(paths["bin"], len(full), 32, 32, "uint16")
    reordered = test.iloc[[5, 0, 3]].reset_index(drop=True)
    dataset = EvalDataset(reordered, paths["bin"], len(full), (32, 32), "uint16", lambda x: x)
    raw = np.memmap(paths["bin"], mode="r", dtype="uint16", shape=(25, 32, 32))
    for i, original in enumerate([23, 18, 21]):
        image, index = dataset[i]
        np.testing.assert_allclose(image.numpy()[0, 0], raw[original].astype(np.float32) / 65535)
        assert index == i
    assert set(train.patient).isdisjoint(val.patient)


def test_ambiguous_or_invalid_indices_fail():
    full = pd.DataFrame({"id": ["duplicate", "duplicate"]})
    with pytest.raises(ValueError, match="duplicate ids"):
        ensure_original_index(full.iloc[:1], full, "train")
    for index in [-1, 2, 0.5, np.nan]:
        with pytest.raises(ValueError, match="original_index"):
            ensure_original_index(pd.DataFrame({"original_index": [index]}), full, "train")
    mapped = ensure_original_index(pd.DataFrame({"original_index": [1]}), full, "train")
    assert mapped.original_index.tolist() == [1]


def test_patient_overlap_rejected(tiny_run):
    paths = load_run_config(tiny_run)["paths"]
    val = pd.read_csv(paths["val_csv"])
    val.loc[0, "patient"] = "patient-0"
    val.to_csv(paths["val_csv"], index=False)
    with pytest.raises(RuntimeError, match="Patient leakage"):
        load_splits(*(paths[key] for key in ("full_csv", "train_csv", "val_csv", "test_csv")))


def test_evaluation_policies_share_mask_and_crop():
    aug = {
        "preprocessing": {
            "foreground_crop": {"enabled": True, "threshold_abs": 0.01, "margin_frac": 0.1},
            "top_right_corner_mask": {"enabled": True, "skip_if_single_component": False, "frac_x": 0.4, "frac_y": 0.3},
        }
    }
    image = torch.zeros(1, 32, 32)
    image[:, 2:30, 10:30] = 0.8
    image[:, 2:6, 2:5] = 1.0
    training_eval = ConfigurableMGAugmentation(aug, 24, train=False)
    report_eval = evaluation_transform(aug, 24)
    transfer_eval = MildSupervisedMGAugmentation(aug, 24, train=False)
    torch.testing.assert_close(training_eval(image), report_eval(image), rtol=0, atol=0)
    torch.testing.assert_close(training_eval(image), transfer_eval(image), rtol=0, atol=0)
    torch.testing.assert_close(report_eval(image), report_eval(image), rtol=0, atol=0)
    report_eval.eval()  # nn.Module.train must remain callable.


@pytest.mark.parametrize("head_dim", [0, 7])
def test_checkpoint_roundtrip_and_transfer(tmp_path, head_dim):
    cfg = TrainConfig(
        backbone_name="vit_tiny_patch16_224",
        image_size=32,
        backbone_num_classes=head_dim,
        backbone_output_dim=7,
        projector_hidden_dim=8,
        projection_dim=4,
        drop_path_rate=0,
    )
    model = ViTEncoder(cfg).eval()
    checkpoint = tmp_path / "model.pt"
    payload = {
        "config": asdict(cfg),
        "augmentation_config": {"image": {"output_size": 32}},
        "model_state_dict": {"module." + key: value for key, value in model.state_dict().items()},
    }
    # Old checkpoints did not store backbone_num_classes; infer it from their tensors.
    payload["config"].pop("backbone_num_classes")
    torch.save(payload, checkpoint)
    loaded, recovered, _ = load_encoder(checkpoint, torch.device("cpu"))
    image = torch.rand(2, 1, 1, 32, 32)
    with torch.no_grad():
        for expected, actual in zip(model(image), loaded(image)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert recovered.backbone_num_classes == head_dim
    classifier = ViTBackboneClassifier(cfg.backbone_name, 32, cfg.backbone_output_dim, 0, backbone_num_classes=head_dim)
    load_jepa_backbone(classifier, checkpoint, torch.device("cpu"))
    classifier.eval()
    with torch.no_grad():
        torch.testing.assert_close(classifier.backbone(image[:, 0]), model.backbone(image[:, 0]))
    payload["model_state_dict"].pop("module.proj.0.weight")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_encoder(checkpoint, torch.device("cpu"))


@pytest.mark.parametrize("workers", [0, 1])
def test_feature_rows_remain_aligned(tiny_run, workers):
    paths = load_run_config(tiny_run)["paths"]
    full, train, _, _ = load_splits(*(paths[key] for key in ("full_csv", "train_csv", "val_csv", "test_csv")))
    cfg = TrainConfig(image_size=32, image_height=32, image_width=32)
    model = torch.nn.Module()
    model.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1024, 4))
    selected = train.iloc[[7, 2, 11]]
    features, metadata = extract_features(
        selected,
        paths["bin"],
        len(full),
        cfg,
        {},
        model,
        torch.device("cpu"),
        FeatureConfig(batch_size=2, num_workers=workers),
        "train",
    )
    assert metadata.original_index.tolist() == [7, 2, 11]
    assert features.shape == (3, 4)
    dataset = EvalDataset(selected, paths["bin"], len(full), (32, 32), "uint16", lambda x: x)
    with torch.no_grad():
        expected = model.backbone(torch.stack([dataset[i][0][0] for i in range(3)]))
    torch.testing.assert_close(features, expected)


@pytest.mark.parametrize("mode", ["pooled_views", "author_ddp_pooled_views", "author_ddp_per_view"])
def test_loss_shapes_normalization_and_gradients(mode):
    cfg = TrainConfig(sigreg_mode=mode, sigreg_knots=5, sigreg_num_projections=8)
    values = torch.randn(6, 2, 4, requires_grad=True)
    total, invariance, sigreg = LeJEPALoss(cfg)(values)
    normalized = torch.nn.functional.normalize(values, dim=-1) * 2
    expected_inv = (normalized - normalized.mean(dim=1, keepdim=True)).square().mean()
    torch.testing.assert_close(invariance, expected_inv)
    torch.testing.assert_close(total, (1 - cfg.lambda_sigreg) * invariance + cfg.lambda_sigreg * sigreg)
    total.backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


def test_pooled_sigreg_single_rank_equivalence():
    values = torch.randn(6, 2, 4)
    gradients, losses = [], []
    for cls in (PooledViewsSIGReg, AuthorDDPPooledViewsSIGReg):
        value = values.clone().requires_grad_()
        torch.manual_seed(12)
        loss = cls(knots=5, num_projections=8)(value)
        loss.backward()
        losses.append(loss)
        gradients.append(value.grad)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_budget_subsets_are_nested_reproducible_and_unique():
    frame = pd.DataFrame({"target_collapsed": [0] * 30 + [1] * 8 + [2] * 5, "original_index": range(43)})
    first = build_subset_plan(frame, ("6", "15", "30", "full"), "progressive", "log", 17)
    second = build_subset_plan(frame, ("6", "15", "30", "full"), "progressive", "log", 17)
    previous = set()
    for budget, size in (("6", 6), ("15", 15), ("30", 30), ("full", 43)):
        selected = first[budget]["df"].original_index.tolist()
        assert len(set(selected)) == size
        assert previous.issubset(selected)
        assert selected == second[budget]["df"].original_index.tolist()
        previous = set(selected)


def test_config_relative_paths_and_transfer_overrides(tiny_run, monkeypatch):
    monkeypatch.chdir(tiny_run.parent.parent)
    run = load_run_config(tiny_run)
    cfg = build_config_from_run_config(run, str(tiny_run))
    assert cfg.full_csv_path == str(tiny_run.parent / "full.csv")
    transfer = {
        "checkpoint": "checkpoint.pt",
        "full_csv": "full.csv",
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "test_csv": "test.csv",
        "bin_path": "images.bin",
        "aug_config": "augmentation.json",
        "output_dir": "transfer",
        "subset_sizes": ["6", "full"],
        "epochs": 5,
        "amp": False,
    }
    path = tiny_run.parent / "transfer.json"
    path.write_text(json.dumps(transfer))
    monkeypatch.setattr(sys, "argv", ["run_transfer_experiment.py", "--run-config", str(path), "--epochs", "2"])
    result = parse_transfer_config()
    assert result.epochs == 2
    assert result.amp is False
    assert result.subset_sizes == ("6", "full")
    assert result.full_csv == cfg.full_csv_path
