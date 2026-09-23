"""Sequential LeJEPA experiments from scalar-or-array training settings."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict

from core.config import (
    TrainConfig,
    analysis_config_from_dict,
    build_config_from_run_config,
    load_analysis_config,
    read_json,
    resolve_path,
)


# Ordinary scalar fields supported by the existing training-config parser.
# Arrays elsewhere (analysis targets, for example) are not sweep axes.
FIELDS = {
    "run": "name description output_dir seed save_resolved_config",
    "model": "backbone_name image_size backbone_output_dim backbone_num_classes num_classes projection_dim projector_hidden_dim drop_path_rate image_height image_width memmap_dtype normalize_mode percentile_low percentile_high",
    "training": "epochs batch_size num_views num_workers grad_accum_steps",
    "optimizer": "learning_rate weight_decay warmup_epochs min_learning_rate eta_min grad_clip_norm",
    "loss": "lambda_sigreg sigreg_knots sigreg_num_projections sigreg_mode sigreg_normalize_by_n projection_normalization",
    "distributed": "no_cuda_timing_sync",
    "checkpointing": "checkpoint_every_epochs resume_checkpoint",
    "diagnostics": "diagnostic_every_batches",
}
ALIASES = {
    "name": "run_name",
    "description": "run_description",
    "num_classes": "backbone_num_classes",
    "min_learning_rate": "eta_min",
    "no_cuda_timing_sync": "timing_cuda_synchronize",
    "resume_checkpoint": "resume_checkpoint_path",
}
PATHS = {"full_csv", "bin", "train_csv", "val_csv", "test_csv", "augmentation_config", "analysis_config"}


def save_state(value, path):
    """Replace manifests atomically so an interruption cannot leave partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def expand_config(path):
    """Validate all combinations before writing anything or starting a worker."""
    raw = read_json(path)
    if "schema_version" in raw and not isinstance(raw["schema_version"], str):
        raise ValueError("schema_version must be a scalar string")
    unknown = set(raw) - set(FIELDS) - {"schema_version", "paths", "analysis"}
    if unknown:
        raise ValueError(f"Unknown training sections: {sorted(unknown)}")
    base = copy.deepcopy(raw)
    axes = {}
    defaults = TrainConfig()
    for section, fields in FIELDS.items():
        values = base.get(section, {})
        if not isinstance(values, dict) or set(values) - set(fields.split()):
            raise ValueError(f"Unknown fields or invalid object in {section}")
        for key, value in values.items():
            choices = value if isinstance(value, list) else [value]
            if not choices or len({json.dumps(v, sort_keys=True) for v in choices}) != len(choices):
                raise ValueError(f"{section}.{key}: empty or duplicate choices")
            if isinstance(value, list):
                if section == "run" and key != "seed":
                    raise ValueError(f"Only run.seed can vary within run; {key} must be scalar")
                axes[f"{section}.{key}"] = choices
            expected = type(getattr(defaults, ALIASES.get(key, key)))
            for choice in choices:
                if key == "resume_checkpoint" and choice is None:
                    continue
                valid = type(choice) is expected
                if expected is float:
                    valid = type(choice) in (float, int) and math.isfinite(choice)
                if not valid:
                    raise ValueError(f"{section}.{key}: expected {expected.__name__}, got {choice!r}")
    paths = base.get("paths", {})
    if not isinstance(paths, dict) or set(paths) - PATHS:
        raise ValueError("Unknown or invalid paths")
    for key, value in paths.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a nonempty scalar path")
        paths[key] = resolve_path(value, path)
    if "analysis" in base and "analysis_config" in paths:
        raise ValueError("Use inline analysis OR paths.analysis_config, not both")
    if "analysis" in base:
        settings = analysis_config_from_dict(base.pop("analysis"))
    elif "analysis_config" in paths:
        settings = load_analysis_config(paths["analysis_config"])
    else:
        settings = analysis_config_from_dict(
            {
                "pca": {"output_format": "compact", "split": "val", "color_by": ["collapsed_birads"]},
                "probe": {
                    "targets": ["collapsed_birads"],
                    "evaluate_test": False,
                    "select_best_by": "val_balanced_accuracy",
                },
            }
        )
    if not settings.pca.enabled or not settings.probe.enabled or settings.pca.output_format != "compact":
        raise ValueError("Sweep reports require enabled probes and compact PCA")
    if settings.pca.split not in {"train", "val"}:
        raise ValueError("Use train or val PCA for model comparison; reserve test for final evaluation")
    if not settings.probe.targets or len(set(settings.probe.targets)) != len(settings.probe.targets):
        raise ValueError("Probe targets must be nonempty and unique")
    if base.get("checkpointing", {}).get("resume_checkpoint"):
        raise ValueError("Start sweeps without a shared resume_checkpoint; use --resume for interrupted sweeps")
    base.setdefault("run", {})
    if not base["run"].get("output_dir"):
        raise ValueError("run.output_dir is required")
    output = Path(resolve_path(base["run"]["output_dir"], path))
    base["run"]["output_dir"] = str(output)
    runs = []
    for number, values in enumerate(itertools.product(*axes.values()), 1):
        config = copy.deepcopy(base)
        parameters = dict(zip(axes, values))
        for field, value in parameters.items():
            section, key = field.split(".")
            config[section][key] = value
        run_id = f"run_{number:03d}"
        config["run"]["name"] = f"{base['run'].get('name', 'sweep')}_{run_id}"
        config["run"]["output_dir"] = str(output / run_id)
        config["paths"]["analysis_config"] = str(output / "analysis_config.json")
        cfg = build_config_from_run_config(config, str(path))
        for field in ("projection_dim", "projector_hidden_dim", "image_size", "image_height", "image_width"):
            if getattr(cfg, field) < 1:
                raise ValueError(f"{field} must be positive")
        if cfg.backbone_num_classes < 0 or cfg.backbone_output_dim < 1 or not 0 <= cfg.drop_path_rate < 1:
            raise ValueError("Invalid backbone dimensions or drop_path_rate")
        if (
            cfg.normalize_mode not in {"uint16", "per_image_percentile"}
            or not 0 <= cfg.percentile_low < cfg.percentile_high <= 100
        ):
            raise ValueError("Invalid image normalization settings")
        if cfg.learning_rate <= 0 or cfg.weight_decay < 0 or not 0 <= cfg.eta_min <= cfg.learning_rate:
            raise ValueError("Invalid learning rate, minimum learning rate or weight decay")
        if config.get("training", {}).get("grad_accum_steps", 1) < 1 or cfg.warmup_epochs < 0:
            raise ValueError("Invalid gradient accumulation or warmup")
        runs.append({"id": run_id, "parameters": parameters, "config": config})
    return output, runs, asdict(settings), list(axes)


def preflight(runs, analysis):
    """Check shared data and every requested binary layout before expensive jobs."""
    from core.data import load_splits, validate_bin
    import timm

    paths = runs[0]["config"]["paths"]
    full, train, val, _ = load_splits(paths["full_csv"], paths["train_csv"], paths["val_csv"], paths["test_csv"])
    for target in analysis["probe"]["targets"]:
        if target not in train or target not in val:
            raise ValueError(f"Unknown probe target: {target}")
    color = (analysis["pca"]["color_by"] or ["collapsed_birads"])[0]
    if color not in train or color not in val:
        raise ValueError(f"Unknown PCA color column: {color}")
    augmentation = read_json(paths["augmentation_config"])
    for run in runs:
        cfg = build_config_from_run_config(run["config"], "")
        if not timm.is_model(cfg.backbone_name):
            raise ValueError(f"Unknown timm backbone: {cfg.backbone_name}")
        validate_bin(paths["bin"], len(full), cfg.image_height, cfg.image_width, cfg.memmap_dtype)
    # Record file identities to prevent silently mixing datasets on resume.
    inputs = {
        k: {"path": v, "size": Path(v).stat().st_size, "mtime_ns": Path(v).stat().st_mtime_ns}
        for k, v in paths.items()
        if k not in {"analysis_config", "augmentation_config"}
    }
    return {"data_files": inputs, "augmentation": augmentation}


def launch(run, processes, log):
    """One subprocess per experiment; a finished checkpoint needs analysis only."""
    root = Path(__file__).resolve().parent
    output = Path(run["config"]["run"]["output_dir"])
    config = output / "run_config.json"
    checkpoint = output / "models" / "final_lejepa_checkpoint.pt"
    if checkpoint.exists() and (output / "metrics" / "summary.json").exists():
        command = [
            sys.executable,
            str(root / "analyze_medjepa.py"),
            "--checkpoint",
            str(checkpoint),
            "--run-config",
            str(config),
        ]
    else:
        command = [sys.executable]
        if processes > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={processes}"]
        command += [str(root / "train_medjepa.py"), "--run-config", str(config)]
    log.write("\n" + json.dumps(command) + "\n")
    log.flush()
    return subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode


def collect_results(manifest):
    rows = []
    for run in manifest["runs"]:
        if run["status"] != "completed":
            continue
        output = Path(run["config"]["run"]["output_dir"])
        report = read_json(output / "analysis" / "linear_probe_report.json")
        for target, result in report["probes"].items():
            if result["status"] != "ok":
                continue
            row = {
                "run": run["id"],
                **run["parameters"],
                "seed": run["config"]["run"].get("seed", 42),
                "target": target,
                "elapsed_seconds": run.get("elapsed_seconds", 0),
            }
            for split, metrics in result["metrics"].items():
                if split in {"val", "test"} and metrics:
                    row.update({f"{split}_{key}": value for key, value in metrics.items()})
            rows.append(row)
    return rows


def comparison_reports(output, manifest):
    """Sweep-only tables, metric plots, PCA panels and training diagnostics."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from core.pca import plot_pca2
    from core.config import slugify

    destination = output / "comparison"
    destination.mkdir(exist_ok=True)
    rows = collect_results(manifest)
    save_state(rows, destination / "results.json")
    pd.DataFrame(rows).to_csv(destination / "results.csv", index=False)
    pd.DataFrame(
        [
            {"run": r["id"], "status": r["status"], **r["parameters"], "error": r.get("error", "")}
            for r in manifest["runs"]
        ]
    ).to_csv(destination / "status.csv", index=False)
    axes = [axis for axis in manifest["axes"] if axis != "run.seed"]
    winners = {}
    if rows:
        frame = pd.DataFrame(rows)
        for target, group in frame.groupby("target", sort=False):
            winners[target] = group.loc[group.val_balanced_accuracy.idxmax()].to_dict()
            fig, panels = plt.subplots(1, 2, figsize=(max(10, len(group) * 0.7), 4))
            for ax, metric in zip(panels, ["val_balanced_accuracy", "val_macro_f1"]):
                ax.bar(group.run, group[metric])
                ax.set(title=metric, ylim=(0, 1))
                ax.tick_params(axis="x", rotation=90)
            fig.suptitle(f"{target}: individual runs (mapping in results.csv)")
            fig.tight_layout()
            fig.savefig(destination / f"probe_{slugify(target)}.png", dpi=150)
            plt.close(fig)
            if len(axes) == 2:
                fig, panels = plt.subplots(1, 2, figsize=(11, 5))
                for ax, metric in zip(panels, ["val_balanced_accuracy", "val_macro_f1"]):
                    index_values = list(dict.fromkeys(r["parameters"][axes[0]] for r in manifest["runs"]))
                    column_values = list(dict.fromkeys(r["parameters"][axes[1]] for r in manifest["runs"]))
                    table = group.pivot_table(index=axes[0], columns=axes[1], values=metric, aggfunc="mean").reindex(
                        index=index_values, columns=column_values
                    )
                    std = group.pivot_table(index=axes[0], columns=axes[1], values=metric, aggfunc="std").reindex(
                        index=table.index, columns=table.columns
                    )
                    counts = group.pivot_table(index=axes[0], columns=axes[1], values=metric, aggfunc="count").reindex(
                        index=table.index, columns=table.columns
                    )
                    ax.imshow(table, vmin=0, vmax=1, cmap="viridis")
                    ax.set(
                        xticks=range(len(table.columns)),
                        xticklabels=table.columns,
                        yticks=range(len(table.index)),
                        yticklabels=table.index,
                        xlabel=axes[1],
                        ylabel=axes[0],
                        title=metric,
                    )
                    for i in range(len(table)):
                        for j in range(len(table.columns)):
                            mean = table.iloc[i, j]
                            label = "pending" if np.isnan(mean) else f"{mean:.3f}\nn={int(counts.iloc[i, j])}"
                            if counts.iloc[i, j] > 1:
                                label += f"\nSD={std.iloc[i, j]:.3f}"
                            ax.text(j, i, label, ha="center", va="center", color="white")
                fig.suptitle(f"{target}: means over completed seeds")
                fig.tight_layout()
                fig.savefig(destination / f"heatmap_{slugify(target)}.png", dpi=150)
                plt.close(fig)
    save_state(
        {"selection_metric": "val_balanced_accuracy", "best_individual_runs": winners}, destination / "selection.json"
    )
    # Each panel fits PCA independently; compare class structure, not absolute axes.
    completed = [r for r in manifest["runs"] if r["status"] == "completed"]
    reference_ids = None
    for offset in range(0, len(completed), 12):
        page = completed[offset : offset + 12]
        fig, panels = plt.subplots(
            math.ceil(len(page) / 3), 3, figsize=(15, 4 * math.ceil(len(page) / 3)), squeeze=False
        )
        for ax in panels.flat:
            ax.set_visible(False)
        for ax, run in zip(panels.flat, page):
            ax.set_visible(True)
            file = Path(run["config"]["run"]["output_dir"]) / "analysis" / "pca_coordinates.npz"
            with np.load(file, allow_pickle=False) as data:
                ids = data["original_index"]
                if reference_ids is not None and not np.array_equal(reference_ids, ids):
                    raise ValueError("PCA rows differ between experiments; comparisons require identical samples")
                reference_ids = ids.copy()
                title = run["id"] + "\n" + ", ".join(f"{k.split('.')[-1]}={v}" for k, v in run["parameters"].items())
                plot_pca2(ax, data["coordinates"], pd.Series(data["labels"]), title, str(data["color_by"]))
                variance = data["explained_variance_ratio"]
                ax.set_xlabel(f"PC1 ({variance[0]:.1%})")
                ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
        fig.suptitle("Backbone PCA: identical images; independently fitted axes")
        fig.tight_layout()
        fig.savefig(destination / f"pca_comparison_{offset // 12 + 1:02d}.png", dpi=150)
        plt.close(fig)
    if completed:
        fig, panels = plt.subplots(2, 3, figsize=(15, 8))
        for ax, key in zip(
            panels.flat, ["invariance", "sigreg", "emb_std", "raw_proj_std", "emb_effective_rank", "epoch_time_sec"]
        ):
            for run in completed:
                history = read_json(Path(run["config"]["run"]["output_dir"]) / "metrics" / "training_history.json")
                values = history.get(key, [])
                ax.plot(range(1, len(values) + 1), values, label=run["id"])
            ax.set(title=key, xlabel="Epoch")
            ax.grid(alpha=0.2)
        panels.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / "training_diagnostics.png", dpi=150)
        plt.close(fig)


def execute(output, runs, analysis, axes, *, resume=False, processes=1, reports_only=False):
    if processes < 1:
        raise ValueError("processes must be positive")
    for run in runs:
        cfg = build_config_from_run_config(run["config"], "")
        if cfg.batch_size % processes or cfg.batch_size // processes < 2:
            raise ValueError(
                "Global batch_size must be divisible by process count, with at least two images per process"
            )
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".sweep.lock"
    # Never run two controllers against the same outputs.
    with lock.open("x") as file:
        file.write(f"pid={os.getpid()}\n")
    try:
        state_path = output / "sweep_manifest.json"
        plan_hash = hashlib.sha256(
            json.dumps({"runs": runs, "analysis": analysis, "axes": axes}, sort_keys=True).encode()
        ).hexdigest()
        if reports_only:
            manifest = read_json(state_path)
            if manifest["plan_hash"] != plan_hash:
                raise ValueError("Config changed; use the original config to regenerate reports")
            comparison_reports(output, manifest)
            return 0
        identity = {
            "runs": runs,
            "analysis": analysis,
            "axes": axes,
            "processes": processes,
            "inputs": preflight(runs, analysis),
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if state_path.exists():
            if not resume and not reports_only:
                raise ValueError("Sweep already exists. Use --resume or choose a new output_dir")
            manifest = read_json(state_path)
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("Config, process count or input files changed; choose a new output_dir")
        else:
            if reports_only:
                raise ValueError("No existing sweep to report")
            if any((output / r["id"]).exists() for r in runs):
                raise ValueError("Run directories already exist without a manifest; use a new output_dir")
            manifest = {
                "fingerprint": fingerprint,
                "plan_hash": plan_hash,
                "axes": axes,
                "inputs": identity["inputs"],
                "runs": [{**copy.deepcopy(r), "status": "pending", "elapsed_seconds": 0} for r in runs],
            }
            save_state(analysis, output / "analysis_config.json")
            save_state(identity["inputs"]["augmentation"], output / "augmentation_config.json")
            # Freeze augmentation content while preserving source paths in identity.
            for run in manifest["runs"]:
                run["config"]["paths"]["augmentation_config"] = str(output / "augmentation_config.json")
                save_state(run["config"], Path(run["config"]["run"]["output_dir"]) / "run_config.json")
            save_state(manifest, state_path)
        if (
            read_json(output / "analysis_config.json") != analysis
            or read_json(output / "augmentation_config.json") != identity["inputs"]["augmentation"]
        ):
            raise ValueError("Generated analysis or augmentation snapshot was modified")
        for run in manifest["runs"]:
            if read_json(Path(run["config"]["run"]["output_dir"]) / "run_config.json") != run["config"]:
                raise ValueError(f"Generated config for {run['id']} was modified")
        for run in manifest["runs"]:
            if run["status"] == "completed":
                folder = Path(run["config"]["run"]["output_dir"])
                for name in ("linear_probe_report.json", "pca.png", "pca_coordinates.npz", "analysis_metadata.json"):
                    if not (folder / "analysis" / name).is_file():
                        raise ValueError(
                            f"Completed run {run['id']} is missing {name}; restore its outputs before resuming"
                        )
                continue
            run.update(status="running", error="")
            save_state(manifest, state_path)
            print(f"{run['id']}: {run['parameters']}", flush=True)
            start = time.monotonic()
            try:
                folder = Path(run["config"]["run"]["output_dir"])
                with (folder / "process.log").open("a", encoding="utf-8") as log:
                    code = launch(run, processes, log)
                if code:
                    raise RuntimeError(f"Worker exited with code {code}; see {folder / 'process.log'}")
                for file in ("linear_probe_report.json", "pca.png", "pca_coordinates.npz", "analysis_metadata.json"):
                    if not (folder / "analysis" / file).is_file():
                        raise RuntimeError(f"Worker did not produce {file}")
                run["status"] = "completed"
            except KeyboardInterrupt:
                run.update(status="interrupted", error="Controller interrupted")
                raise
            except Exception as exc:
                run.update(status="failed", error=str(exc))
                print(run["error"], file=sys.stderr, flush=True)
            finally:
                run["elapsed_seconds"] += time.monotonic() - start
                save_state(manifest, state_path)
            comparison_reports(output, manifest)
        comparison_reports(output, manifest)
        return int(any(r["status"] != "completed" for r in manifest["runs"]))
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and list combinations without data access or writes"
    )
    parser.add_argument("--resume", action="store_true", help="Skip completed runs; retry failed/interrupted runs")
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument(
        "--nproc-per-node", type=int, default=1, help="Sequential experiments, each optionally using torchrun DDP"
    )
    args = parser.parse_args()
    if args.nproc_per_node < 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Launch one controller with python; use --nproc-per-node for multi-GPU training")
    output, runs, analysis, axes = expand_config(args.run_config)
    print(f"{len(runs)} experiments; output: {output}")
    if args.dry_run:
        for run in runs:
            print(run["id"], json.dumps(run["parameters"]))
        return 0
    return execute(
        output, runs, analysis, axes, resume=args.resume, processes=args.nproc_per_node, reports_only=args.reports_only
    )


if __name__ == "__main__":
    raise SystemExit(main())
