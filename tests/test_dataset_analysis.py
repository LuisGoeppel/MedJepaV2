import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from core.config import TrainConfig
from core.models import ViTEncoder
from dataset import create_dataset_overview as overview
from dataset import create_mg_machine_family_split as split
from analysis import analyze_mg_factor_birads_baselines as factor
from analysis import create_medjepa_checkpoint_pca_progression as progression
from analysis import pca_outlier_analysis_v3 as outlier


def test_dataset_and_factor_entrypoints(tmp_path, monkeypatch):
    n = 60
    pd.DataFrame(
        {
            "patient": [f"p{i}" for i in range(n)],
            "id": [f"i{i}" for i in range(n)],
            "birads": [1, 3, 5] * 20,
            "collapsed_birads": ["routine", "follow_up", "biopsy"] * 20,
            "machine": ["Hologic", "GE", "Hologic"] * 20,
            "view": "CC",
            "laterality": "L",
        }
    ).to_csv(tmp_path / "mg-only-all.csv", index=False)
    np.random.default_rng(2).integers(0, 65535, (n, 224, 224), dtype=np.uint16).tofile(tmp_path / "mg-only-all.bin")
    monkeypatch.setattr(sys, "argv", ["overview", str(tmp_path)])
    overview.main()
    assert (tmp_path / "mg_dataset_overview.html").stat().st_size > 1000
    assert json.loads((tmp_path / "mg_dataset_overview.json").read_text())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split",
            "--full-csv",
            str(tmp_path / "mg-only-all.csv"),
            "--mode",
            "resplit",
            "--output-dir",
            str(tmp_path / "splits"),
            "--prefix",
            "check",
        ],
    )
    split.main()
    frames = [pd.read_csv(tmp_path / f"splits/check_{name}.csv") for name in ["train", "val", "test"]]
    indices = set(pd.concat(frames).original_index)
    assert indices == {i for i in range(n) if i % 3 != 1}
    assert not any(split.patient_overlap_report(dict(zip(["train", "val", "test"], frames))).values())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split",
            "--mode",
            "filter_existing",
            "--full-csv",
            str(tmp_path / "mg-only-all.csv"),
            "--train-csv",
            str(tmp_path / "splits/check_train.csv"),
            "--val-csv",
            str(tmp_path / "splits/check_val.csv"),
            "--test-csv",
            str(tmp_path / "splits/check_test.csv"),
            "--output-dir",
            str(tmp_path / "filtered"),
            "--prefix",
            "check",
        ],
    )
    split.main()
    pd.testing.assert_frame_equal(frames[0], pd.read_csv(tmp_path / "filtered/check_train.csv"))

    monkeypatch.setattr(
        sys, "argv", ["factor", "--csv", str(tmp_path / "mg-only-all.csv"), "--out-dir", str(tmp_path / "factors")]
    )
    factor.main()
    report = json.loads((tmp_path / "factors/factor_birads_report.json").read_text())
    assert len(report) == 4


def checkpoint_fixture(tiny_run, head):
    root = tiny_run.parent
    cfg = TrainConfig(
        backbone_name="vit_tiny_patch16_224",
        backbone_num_classes=head,
        backbone_output_dim=192 if head == 0 else head,
        image_size=32,
        image_height=32,
        image_width=32,
        projector_hidden_dim=8,
        projection_dim=4,
        drop_path_rate=0.0,
        full_csv_path=str(root / "full.csv"),
        train_csv_path=str(root / "train.csv"),
        val_csv_path=str(root / "val.csv"),
        test_csv_path=str(root / "test.csv"),
        bin_path=str(root / "images.bin"),
    )
    model = ViTEncoder(cfg).eval()
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "augmentation_config": {"preprocessing": {"foreground_crop": {"enabled": False}}},
    }
    path = root / "checkpoint_epoch_0001.pt"
    torch.save(payload, path)
    return root, path, model, payload


@pytest.mark.parametrize("head", [0, 6])
def test_shared_analysis_encoder_and_cache(tiny_run, monkeypatch, head):
    root, path, model, payload = checkpoint_fixture(tiny_run, head)
    restored, arch, _ = outlier.build_model(payload, torch.device("cpu"))
    frame = pd.read_csv(root / "test.csv")
    _, full_frame = outlier.load_analysis_df(SimpleNamespace(split="full"), {"full_csv": root / "full.csv"})
    assert full_frame.columns.is_unique
    assert full_frame.original_index.tolist() == list(range(24))
    ds = outlier.make_dataset(frame, 25, root / "images.bin", arch, payload["augmentation_config"], False)
    emb, proj = outlier.extract_representations(
        restored, DataLoader(ds, batch_size=3), torch.device("cpu"), False, "test"
    )
    with torch.no_grad():
        expected_emb, expected_proj = model.encode_one(torch.stack([ds[i][0] for i in range(len(ds))]))
    np.testing.assert_allclose(emb, expected_emb.numpy(), atol=1e-6)
    np.testing.assert_allclose(proj, expected_proj.numpy(), atol=1e-6)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "progression",
            "--models-dir",
            str(root),
            "--full-csv",
            str(root / "full.csv"),
            "--bin",
            str(root / "images.bin"),
            "--test-csv",
            str(root / "test.csv"),
            "--output-pdf",
            str(root / "progression.pdf"),
            "--image-height",
            "32",
            "--image-width",
            "32",
            "--num-workers",
            "0",
            "--batch-size",
            "3",
            "--use-cache",
        ],
    )
    args = progression.parse_args()
    features, cfg = progression.extract_or_load_features(
        "one", path, frame, args, payload["augmentation_config"], torch.device("cpu"), 25, None
    )
    np.testing.assert_allclose(features, emb, atol=1e-6)
    assert cfg.backbone_num_classes == head
    original = progression.extract_features_for_checkpoint
    calls = []

    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(progression, "extract_features_for_checkpoint", counted)
    cached, cached_cfg = progression.extract_or_load_features(
        "one", path, frame, args, payload["augmentation_config"], torch.device("cpu"), 25, None
    )
    assert not calls and cached_cfg == cfg
    np.testing.assert_array_equal(cached, features)
    reversed_features, _ = progression.extract_or_load_features(
        "one", path, frame.iloc[::-1], args, payload["augmentation_config"], torch.device("cpu"), 25, None
    )
    assert len(calls) == 1
    np.testing.assert_allclose(reversed_features, features[::-1], atol=1e-6)
    broken = {**payload, "model_state_dict": dict(payload["model_state_dict"])}
    del broken["model_state_dict"]["proj.0.weight"]
    torch.save(broken, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        progression.extract_features_for_checkpoint(
            "bad", path, frame, args, payload["augmentation_config"], torch.device("cpu"), 25
        )


def test_pca_report_entrypoints(tiny_run, monkeypatch):
    root, path, _, _ = checkpoint_fixture(tiny_run, 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "progression",
            "--models-dir",
            str(root),
            "--full-csv",
            str(root / "full.csv"),
            "--bin",
            str(root / "images.bin"),
            "--test-csv",
            str(root / "test.csv"),
            "--output-pdf",
            str(root / "progression.pdf"),
            "--image-height",
            "32",
            "--image-width",
            "32",
            "--num-workers",
            "0",
            "--batch-size",
            "3",
        ],
    )
    progression.main()
    assert (root / "progression.pdf").stat().st_size > 1000
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "outlier",
            "--checkpoint",
            str(path),
            "--output-dir",
            str(root / "outliers"),
            "--split",
            "train",
            "--num-workers",
            "0",
            "--batch-size",
            "3",
            "--max-details",
            "1",
            "--num-aug-views",
            "1",
            "--no-amp",
        ],
    )
    outlier.main()
    report = json.loads((root / "outliers/analysis_summary.json").read_text())
    assert report["primary_architecture"]["backbone_num_classes"] == 0
    assert (root / "outliers/pca_outlier_analysis.html").stat().st_size > 1000


def test_analysis_index_validation_and_legacy_mask():
    full = pd.DataFrame({"id": ["duplicate", "duplicate"], "exam": ["a", "b"]})
    mapped = outlier.ensure_original_index(full.iloc[[1]], full, "test")
    assert mapped.original_index.tolist() == [1]
    for module in (outlier, progression):
        with pytest.raises(ValueError, match="invalid"):
            module.ensure_original_index(pd.DataFrame({"original_index": [0.5]}), full, "test")
    transform = progression.EvalMammographyTransform(
        {
            "preprocessing": {
                "top_corner_mask": {"enabled": True, "side": "left", "height_frac": 0.5, "width_frac": 0.5}
            }
        },
        8,
    )
    result = transform(torch.ones(1, 8, 8))
    assert torch.all(result[:, :4, :4] == 0)
    assert torch.all(result[:, :4, 4:] == 1)
