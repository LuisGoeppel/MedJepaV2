import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from core.data import (
    BaselineDataset,
    BinSpec,
    collapse_label_from_row,
    make_patient_disjoint_pools,
    prepare_baseline_metadata,
    prepare_baseline_split,
    validate_split_frames,
)
from core import supervised as training
from supervised import run_baseline, run_data_ablation, run_backbone_resolution


def test_standalone_entrypoint(tiny_run, monkeypatch):
    root = tiny_run.parent
    args = ["baseline"]
    for option, name in [
        ("full-csv", "full.csv"),
        ("bin", "images.bin"),
        ("train-csv", "train.csv"),
        ("val-csv", "val.csv"),
        ("test-csv", "test.csv"),
        ("output-dir", "baseline"),
    ]:
        args += ["--" + option, str(root / name)]
    args += [
        "--backbone",
        "vit_tiny_patch16_224",
        "--image-size",
        "32",
        "--image-height",
        "32",
        "--image-width",
        "32",
        "--epochs",
        "2",
        "--batch-size",
        "6",
        "--num-workers",
        "0",
        "--no-amp",
    ]
    monkeypatch.setattr(sys, "argv", args)
    run_baseline.main()
    result = json.loads((root / "baseline/metrics.json").read_text())
    assert result["final_epoch"] == 2
    assert sum(map(sum, result["test"]["confusion_matrix"])) == 6
    assert len(list((root / "baseline").glob("*.png"))) == 4
    checkpoint = torch.load(root / "baseline/best_by_val_balanced_accuracy.pt", weights_only=True)
    assert checkpoint["args"]["percentile_high"] == 99.0


@pytest.mark.parametrize("module", [run_data_ablation, run_backbone_resolution])
def test_sweep_entrypoints(tmp_path, monkeypatch, module):
    pytest.importorskip("cv2")
    # Real supported BIN format and enough patients for stratified 80/10/10 splits.
    count = 90
    images = np.random.default_rng(1).integers(0, 65535, (count, 224, 224), dtype=np.uint16)
    images.tofile(tmp_path / "mg-only-all.bin")
    frame = pd.DataFrame(
        {"patient": [f"p{i}" for i in range(count)], "birads": [1, 3, 5] * 30, "machine": "Hologic", "view": "CC"}
    )
    frame.to_csv(tmp_path / "mg-only-all.csv", index=False)
    output = tmp_path / "out"
    args = [
        "sweep",
        "--mg-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--epochs",
        "1",
        "--batch-size",
        "6",
        "--num-workers",
        "0",
        "--no-amp",
    ]
    if module is run_data_ablation:
        args += ["--train-sizes", "12", "--image-size", "32", "--backbone", "vit_tiny_patch16_224"]
        expected = 3
    else:
        args += ["--train-size", "60", "--image-sizes", "32", "--backbones", "vit_tiny_patch16_224", "resnet18"]
        expected = 2
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    results = json.loads((output / "summary.json").read_text())
    assert len(results) == expected
    assert all(result["status"] == "completed" for result in results), results
    assert (output / "summary_report.pdf").stat().st_size > 1000
    assert len(list(output.glob("*/training_curves.png"))) == expected
    if module is run_backbone_resolution:
        for result in results:
            assert "_60_" in result["experiment"]["name"]
            assert len(result["test_by_checkpoint"]) == 3
            assert all(checkpoint is not None for checkpoint in result["test_by_checkpoint"].values())


def test_preprocessing_and_label_policies(tmp_path):
    cv2 = pytest.importorskip("cv2")
    images = np.zeros((2, 32, 32), dtype=np.uint16)
    images[0, 0, 0] = 65535  # Equal 1st/99th percentiles: sweeps use min/max fallback.
    images[1] = np.arange(1024).reshape(32, 32)
    path = tmp_path / "images.bin"
    images.tofile(path)
    frame = pd.DataFrame({"source_index": [0, 1], "target": [0, 2]})
    ds = BaselineDataset(
        frame,
        path,
        2,
        (32, 32),
        "uint16",
        16,
        False,
        "per_image_percentile",
        interpolation="area",
        percentile_fallback=True,
    )
    x, y = ds[0]
    expected = cv2.resize(images[0].astype(np.float32) / 65535, (16, 16), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(x[0], expected)
    arr = images[1].astype(np.float32)
    lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
    expected = cv2.resize(np.clip((arr - lo) / (hi - lo), 0, 1), (16, 16), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(ds[1][0][0], expected)
    assert y.item() == 0
    simple = BaselineDataset(frame, path, 2, (32, 32), "uint16", 16, False, "per_image_percentile")
    assert torch.count_nonzero(simple[0][0]) == 0
    labels = pd.DataFrame({"birads": [0, 1, 3, 6], "birads_numeric": [1, 1, 1, 1]})
    assert prepare_baseline_metadata(labels, policy="resolution")["target"].tolist() == [1, 0, 1, 2]
    assert prepare_baseline_metadata(labels, policy="data_ablation")["target"].tolist() == [0, 0, 0, 0]
    assert collapse_label_from_row(pd.Series({"birads": 0})) == "unknown"
    assert collapse_label_from_row(pd.Series({"birads": 6})) == "biopsy"


def test_baseline_split_validation():
    full = pd.DataFrame({"exam": ["a", "b", "c"], "patient": ["p0", "p1", "p2"], "birads": [1, 3, 5]})
    frames = [prepare_baseline_split(full.iloc[[i]], full, str(i)) for i in range(3)]
    validate_split_frames(*frames)
    assert [df.original_index.iloc[0] for df in frames] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="Patient leakage"):
        validate_split_frames(frames[0], frames[0], frames[2])
    invalid = full.iloc[[0]].assign(original_index=99)
    with pytest.raises(ValueError, match="invalid"):
        prepare_baseline_split(invalid, full, "train")
    with pytest.raises(ValueError, match="Patient identifiers"):
        make_patient_disjoint_pools(full.drop(columns="patient"), seed=1)


@pytest.mark.parametrize("multi_metric, expected_epochs", [(False, 2), (True, 4)])
def test_checkpoint_selection_and_patience(tmp_path, monkeypatch, multi_metric, expected_epochs):
    model = torch.nn.Linear(1, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = iter(
        [
            {"balanced_accuracy": 0.5, "macro_f1": 0.4, "loss": 1.0},
            {"balanced_accuracy": 0.5, "macro_f1": 0.6, "loss": 1.0},
            {"balanced_accuracy": 0.5, "macro_f1": 0.6, "loss": 0.8},
            {"balanced_accuracy": 0.5, "macro_f1": 0.6, "loss": 0.8},
        ]
    )
    monkeypatch.setattr(training, "train_supervised_epoch", lambda *a, **kw: (1.0, 0.5))
    monkeypatch.setattr(training, "evaluate", lambda *a, **kw: {"accuracy": 0.5, **next(metrics)})
    saved = []
    names = {"balanced_accuracy": "balanced.pt"}
    if multi_metric:
        names.update(macro_f1="f1.pt", loss="loss.pt")
    result = training.fit_supervised(
        model,
        [1],
        [1],
        optimizer,
        torch.device("cpu"),
        epochs=6,
        amp=False,
        criterion=torch.nn.CrossEntropyLoss(),
        output_dir=tmp_path,
        checkpoint_names=names,
        save_checkpoint=lambda path, epoch, val: saved.append((path.name, epoch)),
        patience=1,
        min_delta=0.01,
    )
    assert len(result["history"]) == expected_epochs
    assert result["best_epochs"]["balanced_accuracy"] == 1
    if multi_metric:
        assert ("f1.pt", 2) in saved and ("loss.pt", 3) in saved


def test_worker_selects_assigned_gpu(tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(run_backbone_resolution, "open_memmap", lambda *a: None)
    monkeypatch.setattr(run_backbone_resolution.pd, "read_csv", lambda *a: pd.DataFrame())

    def train(**kwargs):
        assert kwargs["device"] == torch.device("cuda:2")
        return {"status": "completed"}

    monkeypatch.setattr(run_backbone_resolution, "train_one_experiment", train)
    result = run_backbone_resolution._worker_run_one_experiment(
        {
            "spec": {"name": "test", "backbone": "resnet18", "image_size": 32},
            "args": {"output_dir": str(tmp_path)},
            "gpu_id": 2,
            "bin_spec": asdict(BinSpec("uint16", 32, 32, 1, True)),
            "n_raw_rows": 3,
            "bin_path": "images.bin",
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "test_csv": "test.csv",
        }
    )
    assert result["status"] == "completed", result
    assert selected == [2]


@pytest.mark.parametrize("sweep", [False, True])
def test_shared_fit_preserves_baseline_shuffle_and_updates(tmp_path, sweep):
    """Compare two epochs against the original loop's iterator/RNG behavior."""
    from tqdm import tqdm as standard_progress
    from tqdm.auto import tqdm as automatic_progress
    from torch.utils.data import DataLoader, TensorDataset

    progress = standard_progress if sweep else automatic_progress
    inputs = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 30
    dataset = TensorDataset(inputs, torch.arange(10) % 3)

    def setup():
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(0.2), torch.nn.Linear(6, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return model, optimizer, DataLoader(dataset, batch_size=4, shuffle=True), DataLoader(dataset, batch_size=5)

    reference, optimizer, train, val = setup()
    for _ in range(2):
        reference.train()
        for x, y in progress(train, disable=True):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(reference(x), y).backward()
            optimizer.step()
        reference.eval()
        with torch.inference_mode():
            for x, y in val if sweep else automatic_progress(val, disable=True):
                reference(x)
    expected_rng = torch.get_rng_state()

    model, optimizer, train, val = setup()
    training.fit_supervised(
        model,
        train,
        val,
        optimizer,
        torch.device("cpu"),
        epochs=2,
        amp=False,
        criterion=torch.nn.CrossEntropyLoss(),
        output_dir=tmp_path,
        checkpoint_names={"balanced_accuracy": "best.pt"},
        save_checkpoint=lambda *args: None,
        progress_factory=progress,
        eval_progress=not sweep,
    )
    assert torch.equal(expected_rng, torch.get_rng_state())
    for key, tensor in reference.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[key]), key
