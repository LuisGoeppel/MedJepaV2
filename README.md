# MedJEPA

Mammography LeJEPA pretraining, representation analysis and transfer label-efficiency experiments.

## Main workflows

Run these commands from the project folder in your existing cluster environment.

**Train, then generate PCA and linear-probe reports:**

```bash
python train_medjepa.py --run-config config/mg_v7_hologic_lorad_config.json
```

For multiple GPUs, use the same entry point with `torchrun`, for example:

```bash
torchrun --standalone --nproc_per_node=2 train_medjepa.py --run-config config/mg_v7_hologic_lorad_config.json
```

The configured training batch size remains the **global** batch size, and must be divisible by the number of ranks. All ranks finish training and close the process group before rank zero runs analysis on one GPU. The final checkpoint is saved before analysis starts.

**Analyze an existing checkpoint without training:**

```bash
python analyze_medjepa.py --run-config config/mg_v7_hologic_lorad_config.json --checkpoint /path/to/run/models/final_lejepa_checkpoint.pt
```

Optional `--analysis-config` and `--output-dir` flags override the referenced analysis configuration and destination. This is a single-process command; launch it with `python`. If post-training analysis fails, use this command to rerun it without retraining.

**Run transfer label-efficiency experiments:**

```bash
python run_transfer_experiment.py --run-config config/transfer.json
```

Edit the checkpoint, dataset and output paths in the example configurations for your environment. The transfer command also accepts the previous experiment's CLI flags; explicitly supplied flags override JSON settings.

## Configuration

- `config/mg_v7_hologic_lorad_config.json` keeps the existing v7 training schema. Its `paths.analysis_config` references `analysis.json`, and `paths.augmentation_config` references `mg_lejepa_aug_v4.json`.
- `config/analysis.json` controls extraction, PCA and probes. It does not repeat model architecture or image preprocessing: those come from the checkpoint. Data locations come from the training JSON passed with `--run-config`.
- `config/transfer.json` uses the fields of `ExperimentConfig`. It is a separate experiment configuration with its own budgets, modes and training settings.
- Paths inside JSON files resolve relative to the file containing them. Absolute cluster paths remain absolute on the cluster. Explicit CLI path overrides resolve from the working directory.
- Set `pca.enabled` or `probe.enabled` to `false` to disable an analysis. Older training JSONs without `paths.analysis_config` still run training without analysis.

The training run records resolved settings and the augmentation configuration. Analysis records its resolved settings, checkpoint path, model/preprocessing settings, data paths, row counts and extraction timing. It does not automatically reuse an old feature cache.

## Shared code

| Module | Responsibility |
| --- | --- |
| `core/config.py` | Settings, JSON loading and CLI parsing |
| `core/data.py` | Labels, metadata, split validation, raw image reading and dataset wrappers |
| `core/transforms.py` | Shared deterministic preprocessing, SSL augmentation and supervised augmentation |
| `core/models.py` | Encoder/projector, transfer classifier, checkpoint reconstruction |
| `core/loss.py` | Invariance loss, projection normalization and all SIGReg variants |
| `core/training.py` | LeJEPA optimization, epoch loop and training workflow |
| `core/training_utils.py` | DDP lifecycle, runtime metadata, checkpoints, diagnostics and SSL loaders |
| `core/supervised.py` | Supervised loaders, a weighted cross-entropy epoch and evaluation |
| `core/plotting.py` | History grids, diagnostics, confusion matrices, class distributions and summary plots |
| `core/features.py` | Shared analysis workflow and ordered embedding extraction |
| `core/pca.py` | Sampling and PCA PDF generation from features |
| `core/probe.py` | Frozen probes and JSON reporting from features |

`core/` contains reusable implementation rather than naming the whole project again. It is kept flat so each responsibility is one import away. Plotting uses functions rather than a stateful class; PCA-specific report layout remains in `core/pca.py`.

`train_medjepa.py` and `analyze_medjepa.py` are small entry points. `run_transfer_experiment.py` owns the experiment-specific protocol: nested budgets, mode selection, head warm-up/unfreezing, optimizer schedules, early stopping, seed/budget loops and result aggregation. It calls shared batch-training and plotting functions; shared modules do not import this experiment script.

Training and standalone analysis call the same `run_analysis` function. Post-training analysis reuses the trained encoder; standalone analysis reconstructs it from the checkpoint. PCA and probing share one extraction pass per required split; PCA-only runs extract only the requested split(s). Feature tensors stay on CPU between extraction and probing. If `feature_extraction.save_features` is enabled, the run also saves tensors and ordered metadata in `features.pt`.

Typical training outputs retain `models/`, `metrics/` and `plots/`. Analysis writes:

```text
<run output>/analysis/
    analysis_config_used.json
    analysis_metadata.json
    pca_report.pdf
    linear_probe_report.json
    features.pt                 # optional
```

Probe results retain per-target class mappings, counts, selection details and train/validation/test metrics. Shared model/preprocessing information and extraction timings now live in `analysis_metadata.json`.

## Refactoring compatibility notes

The four original standalone implementations were replaced as follows:

| Previous command | Replacement |
| --- | --- |
| `train_medjepa_mg_v7.py` | `train_medjepa.py` |
| `run_medjepa_linear_probe_report.py` | `analyze_medjepa.py`, with probes enabled |
| `create_medjepa_pca_report.py` | `analyze_medjepa.py`, with PCA enabled |
| `run_jepa_transfer_label_efficiency_v2.py` | `run_transfer_experiment.py` |

The initial source snapshot is Git commit `463d46b`. Other top-level experiments, the notebook and `old/` remain outside this refactoring scope and retain their original standalone implementations.

The training objective, tensor layout, schedules and transfer subset strategies are retained. Transfer also retains model reuse between runs and nonpersistent workers. With multiple seed repetitions in one process, initial model weights are shared, as before; use separate processes with different seeds for independent initializations.

These corrections are intentional and can affect comparisons with old reports:

- Analysis reconstructs both headless and learned-head backbones and loads model weights strictly. It will not continue with randomly initialized missing weights. Supply a full training checkpoint containing `model_state_dict`, `config` and `augmentation_config`; bare state dictionaries are not sufficient for standalone analysis.
- Analysis now applies the same deterministic foreground crop and corner mask as training and transfer. Old PCA/probe scripts omitted the corner mask.
- Metadata uses one canonical view/laterality/machine-family mapping. Training keeps its numeric BI-RADS label policy; transfer retains its preference for an existing valid `collapsed_birads` column.
- Binary shape uses the raw CSV row count, before label filtering. Split indices must be in bounds and unambiguous; duplicate IDs require explicit `original_index`. Missing patient identifiers, repeated image rows and overlapping splits fail validation.

## Environment and checks

Upload the entry points, **entire `core/` directory** and referenced configurations together. Launch jobs from a fixed snapshot so subsequent uploads do not change queued jobs. No package installation is needed to import `core` when the scripts and package are adjacent. The distribution name in `pyproject.toml` remains `medjepa`, since that names the project, while its Python package is `core`.

`pyproject.toml` declares dependencies. In a prepared environment, optional editable installation is:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m ruff check core tests train_medjepa.py analyze_medjepa.py run_transfer_experiment.py
```

Keep the cluster's working PyTorch/torchvision/CUDA combination. The local `.venv/` is ignored by Git and uses CPU PyTorch for verification; it should not be uploaded as the cluster environment. OpenCV is optional but affects connected-component corner-mask behavior and CLAHE; use the same OpenCV availability when comparing runs.

Tests use synthetic uint16 images and a small real timm ViT, with no dataset or pretrained-weight downloads. They cover checkpoint compatibility, row alignment, preprocessing, losses/gradients, nested budgets, training/resume, shared extraction, PCA/probes and all transfer modes. GPU/DDP execution still requires a cluster smoke run.
