import copy
import json
from pathlib import Path

import numpy as np
import pytest

import run_training_sweep as sweep
from core.config import read_json


def sweep_config(tiny_run):
    raw = read_json(tiny_run)
    raw["paths"].pop("analysis_config")
    raw["model"]["projection_dim"] = [4, 8]
    raw["loss"]["lambda_sigreg"] = [0.02]
    raw["analysis"] = {
        "seed": 42,
        "feature_extraction": {"batch_size": 6, "num_workers": 0},
        "pca": {"split": "val", "max_samples": 6, "output_format": "compact", "color_by": ["collapsed_birads"]},
        "probe": {"targets": ["collapsed_birads"], "probe_epochs": 1, "probe_batch_size": 6, "evaluate_test": False},
    }
    path = tiny_run.parent / "sweep.json"
    path.write_text(json.dumps(raw))
    return path, raw


def test_expansion_and_validation(tiny_run):
    path, raw = sweep_config(tiny_run)
    raw["model"]["projection_dim"] = [16, 64, 128]
    raw["loss"]["lambda_sigreg"] = [0.01, 0.02, 0.05]
    path.write_text(json.dumps(raw))
    output, runs, analysis, axes = sweep.expand_config(path)
    assert len(runs) == 9
    assert axes == ["model.projection_dim", "loss.lambda_sigreg"]
    assert analysis["pca"]["color_by"] == ["collapsed_birads"]
    assert not output.exists()
    assert runs[0]["config"]["paths"]["augmentation_config"] == str(path.parent / "augmentation.json")
    assert len({r["config"]["run"]["output_dir"] for r in runs}) == 9
    raw["run"]["seed"] = [42, 43]
    path.write_text(json.dumps(raw))
    assert len(sweep.expand_config(path)[1]) == 18
    for field, value in [
        ("projection_dim", []),
        ("projection_dim", [4, 4]),
        ("projection_dim", [4, -1]),
        ("projection_dim", [4, 2.5]),
        ("projection_dim", [4, True]),
        ("drop_path_rate", [float("nan")]),
        ("unknown_parameter", [1, 2]),
    ]:
        bad = copy.deepcopy(raw)
        bad["model"][field] = value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            sweep.expand_config(path)
    raw["model"]["projection_dim"] = 4
    raw["loss"]["lambda_sigreg"] = 0.02
    raw["run"]["seed"] = 7
    path.write_text(json.dumps(raw))
    assert len(sweep.expand_config(path)[1]) == 1


def test_real_sequential_sweep_and_resume(tiny_run, monkeypatch):
    """Real subprocess training, compact analysis, failed-run retry and report regeneration."""
    path, raw = sweep_config(tiny_run)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output, runs, analysis, axes = sweep.expand_config(path)
    launch = sweep.launch
    calls = []

    def interrupted_analysis(run, processes, log):
        calls.append(run["id"])
        code = launch(run, processes, log)
        # Mimic a worker failure after final checkpoint creation.
        return 7 if run["id"] == "run_001" and len(calls) == 1 and code == 0 else code

    monkeypatch.setattr(sweep, "launch", interrupted_analysis)
    assert sweep.execute(output, runs, analysis, axes) == 1
    state = read_json(output / "sweep_manifest.json")
    assert [r["status"] for r in state["runs"]] == ["failed", "completed"]
    checkpoint = output / "run_001/models/final_lejepa_checkpoint.pt"
    before = checkpoint.stat().st_mtime_ns
    assert sweep.execute(output, runs, analysis, axes, resume=True) == 0
    assert calls == ["run_001", "run_002", "run_001"]
    assert checkpoint.stat().st_mtime_ns == before  # Analysis retry did not retrain.
    assert not (output / ".sweep.lock").exists()
    rows = json.loads((output / "comparison/results.json").read_text())
    assert len(rows) == 2 and all("test_balanced_accuracy" not in row for row in rows)
    first = None
    for run in runs:
        folder = Path(run["config"]["run"]["output_dir"]) / "analysis"
        assert (folder / "pca.png").exists() and not (folder / "pca_report.pdf").exists()
        metadata = read_json(folder / "analysis_metadata.json")
        assert set(metadata["num_rows"]) == {"train", "val"}
        with np.load(folder / "pca_coordinates.npz") as data:
            if first is not None:
                np.testing.assert_array_equal(first, data["original_index"])
            first = data["original_index"].copy()
    for name in [
        "pca_comparison_01.png",
        "heatmap_collapsed_birads.png",
        "probe_collapsed_birads.png",
        "training_diagnostics.png",
    ]:
        assert (output / "comparison" / name).stat().st_size > 1000
    assert sweep.execute(output, runs, analysis, axes, resume=True) == 0
    assert len(calls) == 3
    with pytest.raises(ValueError, match="already exists"):
        sweep.execute(output, runs, analysis, axes)
    changed = copy.deepcopy(runs)
    changed[0]["config"]["training"]["epochs"] = 2
    with pytest.raises(ValueError, match="changed"):
        sweep.execute(output, changed, analysis, axes, resume=True)
    # Reporting needs saved artifacts, not access to the dataset or another worker.
    monkeypatch.setattr(sweep, "preflight", lambda *args: pytest.fail("reports-only accessed the dataset"))
    assert sweep.execute(output, runs, analysis, axes, reports_only=True) == 0


def test_worker_commands_and_snapshot_guard(tiny_run, monkeypatch):
    path, _ = sweep_config(tiny_run)
    output, runs, analysis, axes = sweep.expand_config(path)
    commands = []

    class Result:
        returncode = 0

    monkeypatch.setattr(sweep.subprocess, "run", lambda command, **kwargs: commands.append(command) or Result())
    from io import StringIO

    sweep.launch(runs[0], 2, StringIO())
    assert "torch.distributed.run" in commands[0] and "--nproc_per_node=2" in commands[0]
    assert commands[0][-3].endswith("train_medjepa.py")
    # Malformed outputs are failures even if a worker returned zero.
    assert sweep.execute(output, runs, analysis, axes) == 1
    state = read_json(output / "sweep_manifest.json")
    assert all(r["status"] == "failed" for r in state["runs"])
    (output / "analysis_config.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot was modified"):
        sweep.execute(output, runs, analysis, axes, resume=True)
