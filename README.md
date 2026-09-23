# MedJEPA

MedJEPA is a research framework for self-supervised mammography representation learning with LeJEPA, combining a vision transformer encoder with invariance and SIGReg objectives. It supports pretraining, representation analysis, supervised baselines and transfer experiments across training-set sizes.

## Capabilities

- Configurable image preprocessing, augmentation, encoder/projector architectures and regularization, with single-GPU and distributed training.
- Sequential hyperparameter and seed sweeps with resume support, validation-based comparisons and diagnostic reports.
- Shared embedding extraction for PCA and frozen linear probes, plus checkpoint progression, outlier analysis and representation comparisons.
- Transfer evaluation across initialization and fine-tuning modes, nested data budgets and sampling strategies, including balanced oversampling.
- Supervised backbone/resolution comparisons, dataset summaries and machine-family split preparation.

Inputs are mammography images stored in a memory-mapped BIN file, CSV metadata and train/validation/test splits. Outputs include checkpoints, resolved configurations, metrics, plots and analysis reports. With oversampling, budgets count training entries including repeats; reports separately record unique images.

## Runnable workflows

Root scripts use `python <file>`. Scripts in `supervised/`, `dataset/` and `analysis/` use module invocation, for example `python -m supervised.run_baseline`. Their `--help` options describe inputs and available settings.

| File | Purpose |
| --- | --- |
| `train_medjepa.py` | Train LeJEPA, save checkpoints and run configured PCA/probe analysis after training. Supports `torchrun` for distributed training. |
| `analyze_medjepa.py` | Generate PCA and linear-probe reports from an existing checkpoint without retraining. |
| `run_training_sweep.py` | Expand arrays in training JSON into sequential experiments; compare probe metrics, PCA and training diagnostics. Supports dry runs, resume and report regeneration. |
| `run_transfer_experiment.py` | Evaluate transfer label efficiency across seeds, budgets and training modes; aggregate metrics and sampling statistics. |
| `run_medjepa_representation_comparison.py` | Compare head embeddings, CLS tokens and supervised cross-attention pooling over frozen patch tokens using linear/MLP classifiers and PCA. |
| `plot_medjepa_training_history.py` | Regenerate diagnostic plots from saved training history, optionally updating them while training runs. |
| `supervised/run_baseline.py` | Train one weighted-cross-entropy classifier on predefined splits, with optional training subsampling. |
| `supervised/run_data_ablation.py` | Compare balanced training sizes across all data, one machine family and one view. |
| `supervised/run_backbone_resolution.py` | Compare backbones and image resolutions using shared training subsets and evaluation pools. |
| `dataset/create_dataset_overview.py` | Produce HTML/JSON summaries of metadata, label distributions, image examples and intensity statistics. |
| `dataset/create_mg_machine_family_split.py` | Filter existing splits by machine family or create new grouped splits. |
| `analysis/create_medjepa_checkpoint_pca_progression.py` | Produce a PDF showing representation changes across training checkpoints. |
| `analysis/pca_outlier_analysis_v3.py` | Screen embedding/projector spaces for outliers and generate reports with image galleries and optional reference-checkpoint comparisons. |
| `analysis/analyze_mg_factor_birads_baselines.py` | Measure associations between metadata factors and collapsed BI-RADS, including factor-only prediction baselines. |
| `notebooks/visualize_mg_lejepa_aug_v2.ipynb` | Interactively inspect image augmentation using the shared training transforms. |

## Shared implementation

Experiment scripts define protocols and orchestration; `core/` provides reusable components.

| File | Responsibility |
| --- | --- |
| `core/config.py` | Training, transfer and analysis settings; JSON loading, path resolution and transfer CLI parsing. |
| `core/data.py` | Label/metadata processing, physical image-row mapping, split validation, memory-mapped image reading and dataset wrappers. |
| `core/transforms.py` | Deterministic preprocessing and self-supervised/supervised augmentation. |
| `core/models.py` | ViT encoders, projectors, classifiers and model reconstruction from checkpoints. |
| `core/loss.py` | Invariance loss, projection normalization and SIGReg variants, including distributed objectives. |
| `core/training.py` | LeJEPA optimizers, schedules, epoch loop and training orchestration. |
| `core/training_utils.py` | Distributed lifecycle, device selection, loaders, checkpoints, runtime metadata and representation diagnostics. |
| `core/supervised.py` | Shared supervised loaders, training/evaluation loops, class weighting and checkpoint selection. |
| `core/features.py` | Ordered embedding extraction and the shared PCA/probe analysis workflow. |
| `core/pca.py` | Sampling, PCA projections, PDF reports and compact PNG/coordinate outputs. |
| `core/probe.py` | Frozen linear probes, target mappings, balanced sampling and metric reports. |
| `core/plotting.py` | Training diagnostics, confusion matrices, class distributions and experiment comparison plots. |

The `__init__.py` files in `core/`, `supervised/`, `dataset/` and `analysis/` define the Python packages. Training and standalone analysis share the same analysis workflow; PCA and probes reuse extracted features.

## Configuration and supporting files

| File | Role |
| --- | --- |
| `config/mg_v7_hologic_lorad_config.json` | v7 training configuration referencing augmentation and analysis settings. |
| `config/mg_v6_base_config.json`, `config/mg_v6_base_hologic_lorad_config.json` | v6 base configurations for full-data and Hologic/Lorad experiments. |
| `config/mg_v6_hologic_lorad_pooled_config.json`, `config/mg_v6_hologic_lorad_sigreg_perview_config.json` | Pooled-view and per-view SIGReg configurations. |
| `config/mg_v6_positive_pairs_config.json`, `config/mg_v6_weighted_infinite_config.json` | Positive-pair and weighted/infinite-sampling training variants. |
| `config/mg_projection_sigreg_sweep.json` | Projection-dimension/SIGReg-weight sweep with embedded analysis settings. |
| `config/mg_lejepa_aug_v4.json` | Mammography augmentation policy. |
| `config/analysis.json` | Feature extraction, PCA and linear-probe settings. |
| `config/transfer.json` | Transfer modes, budgets, sampling and optimization settings. |
| `analysis/factor_report_dataset_full_mg.txt`, `analysis/factor_report_dataset_hologic_lorad.txt` | Saved factor-analysis reports. |
| `pyproject.toml` | Package metadata, dependencies, optional test/OpenCV extras and pytest/Ruff settings. |
| `.gitignore` | Excludes local environments, caches, build artifacts and output directories. |

JSON paths resolve relative to the containing configuration file; CLI path overrides resolve from the working directory. Analysis obtains model/preprocessing settings from the checkpoint and data locations from the training configuration. Example configurations contain environment-specific paths.

## Tests and runtime

Python 3.10+ and dependencies are declared in `pyproject.toml`. OpenCV supports the supervised sweeps' resizing and optional preprocessing. Tests run with `python -m pytest -q` and use synthetic data; CUDA/distributed execution requires GPU validation.

| File | Coverage |
| --- | --- |
| `tests/conftest.py` | Shared test setup and fixtures. |
| `tests/test_shared.py` | Shared data, transforms, models and losses. |
| `tests/test_workflows.py` | Training, checkpoint analysis and transfer workflows. |
| `tests/test_training_sweep.py` | Sweep configuration, execution, resume and reporting. |
| `tests/test_training_devices.py` | Training device selection and distributed setup. |
| `tests/test_transfer_oversampling.py` | Balanced oversampling, nested budgets and sampling reports. |
| `tests/test_image_row_mapping.py` | CSV-to-BIN physical image alignment. |
| `tests/test_supervised_baselines.py` | Supervised experiment drivers and shared training behavior. |
| `tests/test_dataset_analysis.py` | Dataset preparation and analysis tools. |
| `tests/test_remaining_tools.py` | Auxiliary scripts and notebook integration. |
