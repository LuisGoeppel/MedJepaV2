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

For balanced transfer training with a fixed number of **total entries**, set `"subset_strategy": "oversampling"` in the transfer JSON, or pass `--subset-strategy oversampling`. The existing `progressive`, `balanced` and `natural` strategies retain their behavior and defaults. `balance_curve` is ignored for oversampling.

Oversampling divides each budget equally among routine, follow-up and biopsy (counts differ by at most one). Each class uses unique shuffled images first, then repeats images drawn from that class when its supply is exhausted. The subsets and repeated-entry multiplicities are nested across budgets and identical across transfer modes for a given seed. An absent class is an error. No validation or test images are oversampled. Existing class weighting is retained and computed from the constructed training-entry counts, so it is approximately uniform with balanced oversampling.

A `10k` point contains 10,000 entries, including repeats, not 10,000 unique labels plus extra entries. As with existing strategies, budgets exceeding the full training size are capped at that size. `full` uses the original training split's row count as its entry budget but is balanced; it does **not** include every unique majority-class image. Equal budgets preserve entries/batches per epoch at the same batch size. Early stopping and other training settings remain unchanged, so total optimizer steps across complete runs can still differ. Treat this as a fixed-training-entry comparison, and use reported unique counts when discussing annotation efficiency.

Subset manifests, per-run results and summary CSVs record total entries, unique counts per class and repeated entries. For oversampling, `summary_sampling_unique_images.png` replaces the balance-degree and subset-composition summaries with unique images per class and unique-versus-repeated totals. Metric plot axes refer to training entries. Use a separate output directory when comparing strategies so results and old plots are not overwritten or mixed.

## Sequential training sweeps

`run_training_sweep.py` accepts one training JSON. Scalar training fields are fixed; arrays form a Cartesian product. The example `config/mg_projection_sigreg_sweep.json` runs nine 50-epoch experiments: projection dimensions `[16, 64, 128]` and SIGReg weights `[0.01, 0.02, 0.05]`, with a ViT-base learned 512-dimensional backbone head. Set `run.seed` to an array to repeat every combination across seeds. With no arrays, it runs one experiment.

```bash
# Validate configuration and inspect all combinations, without accessing cluster data or writing outputs:
python run_training_sweep.py --run-config config/mg_projection_sigreg_sweep.json --dry-run

# Run sequentially (training followed by analysis for each experiment):
python run_training_sweep.py --run-config config/mg_projection_sigreg_sweep.json

# Optional: each experiment uses two GPUs; experiments are still sequential:
python run_training_sweep.py --run-config config/mg_projection_sigreg_sweep.json --nproc-per-node 2

# Continue an interrupted sweep using the original config and GPU process count:
python run_training_sweep.py --run-config config/mg_projection_sigreg_sweep.json --resume

# Rebuild comparisons from saved results, without training or reading the dataset:
python run_training_sweep.py --run-config config/mg_projection_sigreg_sweep.json --reports-only
```

Launch the controller with `python`, not `torchrun`; it launches DDP workers when requested. Global batch sizes must divide evenly across workers. The sweep CLI uses a whitelist of supported scalar training fields and rejects empty/duplicate arrays, unknown fields and invalid values before launching jobs. Dataset paths, run name/output directory, and analysis settings are shared across runs; only `run.seed` varies within the `run` section. Arrays such as `analysis.probe.targets` retain their usual meaning.

The example embeds an `analysis` section, so no second user-maintained config is needed. Alternatively, use `paths.analysis_config` instead of that section. A fixed analysis seed ensures identical probe sampling and PCA image identities across experiments, even when training seeds vary. PCA and probes reuse backbone embeddings from the same extraction pass. Sweep reports require compact PCA on train or validation data. Other analysis entry points also support `pca.output_format: "compact"`; their default remains the existing PDF report.

The example probes collapsed BI-RADS and ranks individual runs by **validation balanced accuracy**, with macro-F1 alongside it. It sets `probe.evaluate_test: false`, so test embeddings are not extracted and test metrics are omitted. Test CSVs are still read for split-integrity validation. Enable test evaluation only for a separate final evaluation of selected models. The 50-epoch budget is an experimental setting, not a guarantee of convergence. Heatmaps average completed training seeds and show counts and standard deviations; partially completed seed groups should not be treated as a final ranking. `selection.json` identifies the best individual runs, not a seed-aggregated hyperparameter winner.

```text
<sweep output>/
    sweep_manifest.json           # parameters, status, errors, elapsed time, input identities
    analysis_config.json          # generated snapshot
    augmentation_config.json      # generated snapshot
    run_001/                      # unique directory per combination
        run_config.json           # generated scalar training config
        process.log
        models/, metrics/, plots/
        analysis/
            linear_probe_report.json
            pca.png
            pca_coordinates.npz
            analysis_metadata.json
    comparison/
        results.csv, results.json, status.csv, selection.json
        probe_collapsed_birads.png
        heatmap_collapsed_birads.png
        pca_comparison_01.png
        training_diagnostics.png
```

Comparison reports update after each attempt. Two non-seed sweep axes produce metric heatmaps; other grids still produce per-run metric plots and PCA panels. PCA panels use identical samples but independently fitted axes, so absolute coordinates are not comparable. Training diagnostics show loss components, embedding/projection standard deviations, effective rank and epoch timing; total weighted SSL loss is not used to select a winner.

Completed runs are skipped on `--resume`. Failed runs do not stop subsequent experiments, but the controller exits with a nonzero status if any remain failed. If training finished and wrote its final checkpoint and summary, a retry runs analysis only; an interrupted training run restarts from epoch one. Automatic continuation from intermediate optimizer checkpoints is not implemented. Config/input identity changes require a new output directory, and generated config snapshots must remain unmodified. Normal interruption releases `.sweep.lock`; after a forcibly killed controller, remove a stale lock only after confirming its workers have stopped. Upload this script and the updated `core/` together with the config.

## Supervised baselines

The three supervised experiments live in `supervised/`. Run them as modules from this folder (or install the project first). Their existing CLI flags are retained:

```bash
python -m supervised.run_baseline --full-csv /data/mg-only-all.csv --bin /data/mg-only-all.bin --train-csv /data/train.csv --val-csv /data/val.csv --test-csv /data/test.csv --output-dir /runs/baseline
python -m supervised.run_data_ablation --mg-dir /data --output-dir /runs/data_ablation
python -m supervised.run_backbone_resolution --mg-dir /data --output-dir /runs/resolution --parallel-gpus 0 1
```

| Previous script | New module | Protocol |
| --- | --- | --- |
| `run_mg_supervised_weighted_ce_simple.py` | `supervised.run_baseline` | One weighted-CE model using predefined splits; optional training subsampling |
| `run_mg_supervised_baselines.py` | `supervised.run_data_ablation` | Balanced training sizes across all data, one machine family and one view |
| `run_mg_supervised_resolution_backbone_baselines.py` | `supervised.run_backbone_resolution` | Backbones/resolutions sharing one minority-inclusive training subset and evaluation pools |

The drivers own sampling, schedules, checkpoint payloads and result aggregation. `core.supervised.fit_supervised` owns the common epoch loop and checkpoint-selection mechanism; `core.plotting` renders the reports. No experiment imports another experiment.

Historical defaults remain explicit: sweeps use percentile normalization with min/max fallback, OpenCV area resizing and FP16 with gradient scaling; the single baseline uses uint16 normalization and bilinear resizing by default, BF16, warmup and dropped incomplete training batches. The resolution sweep uses weighted validation/test loss and monitors balanced accuracy, macro F1 and loss; improvement in any resets its patience. The single baseline uses unweighted evaluation loss and selects by balanced accuracy. History/metric files retain their previous keys, with additional shared diagnostics; plot layouts are standardized.

Baseline label policies are separate from LeJEPA's numeric 1–5 policy. Sweeps accept numeric BI-RADS 0 and 6; the single baseline accepts 6 but excludes numeric 0. Existing label-column precedence is retained. The data ablation's evaluation pools remain fixed across sizes **within** each dataset setting, but differ between settings.

Intentional corrections: predefined splits now reject patient/image overlap, invalid indices and duplicate images; generated splits require patient identifiers rather than falling back to row splitting. Unknown BIN layouts fail instead of guessing 512×512 uint16. Resolution workers select their assigned CUDA device, and directory names include the actual requested training size. GPU numbers are indices within the job's visible CUDA devices. Do not combine `--parallel-gpus` and `--data-parallel`.

## Dataset preparation and additional analysis

These workflows now live in packages and reuse `core/`. Run them from the project folder with their existing arguments:

```bash
python -m dataset.create_dataset_overview /path/to/mg
python -m dataset.create_mg_machine_family_split --mode filter_existing --full-csv /data/mg-only-all.csv --train-csv /data/train.csv --val-csv /data/val.csv --test-csv /data/test.csv --output-dir /data/hologic_splits
python -m analysis.create_medjepa_checkpoint_pca_progression --models-dir /runs/model/models --full-csv /data/mg-only-all.csv --bin /data/mg-only-all.bin --test-csv /data/test.csv --output-pdf /reports/progression.pdf
python -m analysis.pca_outlier_analysis_v3 --checkpoint /runs/model/models/final_lejepa_checkpoint.pt --output-dir /reports/outliers
python -m analysis.analyze_mg_factor_birads_baselines --csv /data/test.csv --out-dir /reports/factors
```

The original basenames are retained; only their folders and invocation change. The existing text reports in `analysis/` are unchanged. Dataset overview reuses label parsing, machine-family inference, BIN layout detection and HTML figure serialization. The split tool reuses naming, metadata summaries and JSON writing, while retaining its proportional split ratios and historical small-group fallback. It now adds physical `original_index` values before filtering the full CSV in `resplit` mode. Missing group identifiers and nonpositive split ratios are rejected.

The PCA tools reuse the shared encoder/projector, memmap reader, numeric-label helpers, index validation and ordered batch extraction. Outlier analysis also uses the shared augmentation implementation. Its outlier algorithms, detailed galleries and reference-checkpoint model reuse remain local. Progression keeps its historical fixed-corner mask and sampling policy; these differ from training's adaptive corner mask and regular PCA sampling. Its checkpoint loads now fail on missing weights, and both PCA tools support headless v7 encoders. Its feature cache verifies checkpoint/BIN identity, selected row order and preprocessing settings; old caches are recomputed. The overview rejects unknown BIN layouts instead of guessing.

Factor analysis retains its own factor-label rules, metadata fallback and factor-only metrics; replacing those with training policies would change its results. It shares CSV/JSON utilities and robust context parsing. Outlier histograms now handle nearly constant scores without failing to construct bins, and full-dataset analysis preserves one unambiguous physical row-index column.

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
| `core/supervised.py` | Supervised loaders, training/evaluation epochs and single-baseline fitting |
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

The initial source snapshot is Git commit `463d46b`. The supervised, dataset and analysis scripts now reuse `core/`. Historical files in `old/` retain their standalone implementations.

`notebooks/visualize_mg_lejepa_aug_v2.ipynb` imports the training augmentation implementation from `core/transforms.py`, plus shared labels, row mapping and image reading. Start Jupyter from this directory or `notebooks/`, with `core/` available beside `notebooks/`. Edit the notebook's cluster paths before running it; its selected augmentation JSON determines the policy despite the retained v2 filename. The notebook keeps its existing display categories and sampling controls.

The two auxiliary scripts remain at the root:

- `plot_medjepa_training_history.py` retains its CLI, JSON-read retries and watch mode, but calls shared training-history plotting. Training and this command write the same seven plot files, with atomic replacement and support for partial histories.
- `run_medjepa_representation_comparison.py` shares encoder construction, checkpoint hints, labels, image reading and deterministic preprocessing. Its CLS/patch-token extraction, probe protocols, composite-key row matching and report layout remain local. Its existing metadata category conventions are retained. Checkpoints now load strictly and support headless encoders; the historical `head512` result key denotes the backbone output even when its dimension differs from 512. Normalization must be `uint16` or `per_image_percentile`, matching core; invalid modes now fail instead of silently using per-image maximum scaling.

The training objective, tensor layout, schedules and transfer subset strategies are retained. Transfer also retains model reuse between runs and nonpersistent workers. With multiple seed repetitions in one process, initial model weights are shared, as before; use separate processes with different seeds for independent initializations.

These corrections are intentional and can affect comparisons with old reports:

- Analysis reconstructs both headless and learned-head backbones and loads model weights strictly. It will not continue with randomly initialized missing weights. Supply a full training checkpoint containing `model_state_dict`, `config` and `augmentation_config`; bare state dictionaries are not sufficient for standalone analysis.
- Analysis now applies the same deterministic foreground crop and corner mask as training and transfer. Old PCA/probe scripts omitted the corner mask.
- Metadata uses one canonical view/laterality/machine-family mapping. Training keeps its numeric BI-RADS label policy; transfer retains its preference for an existing valid `collapsed_birads` column.
- Binary shape uses the raw CSV row count, before label filtering. Split indices must be in bounds and unambiguous. Without `original_index`, unique IDs map directly to physical rows; duplicate IDs are matched using all shared raw identity metadata (patient, dataset, modality, machine, context, findings and source labels where available). Every requested row must have exactly one match. Remaining ambiguous or unmatched duplicate IDs require explicit `original_index` from the original full CSV; never deduplicate the full CSV or assign split-relative row numbers, because that breaks BIN alignment. Missing patient identifiers, repeated image rows and overlapping splits fail validation.

## Environment and checks

Upload the entry points, **entire `core/`, `supervised/`, `dataset/` and `analysis/` directories** and referenced configurations together. Launch jobs from a fixed snapshot so subsequent uploads do not change queued jobs. No package installation is needed when launching the documented commands from this folder. The distribution name in `pyproject.toml` remains `medjepa`; it includes all four Python packages.

`pyproject.toml` declares dependencies. In a prepared environment, optional editable installation is:

```bash
python -m pip install -e ".[test,opencv]"
python -m pytest -q
python -m ruff check core supervised dataset analysis tests train_medjepa.py analyze_medjepa.py run_transfer_experiment.py plot_medjepa_training_history.py run_medjepa_representation_comparison.py
```

Keep the cluster's working PyTorch/torchvision/CUDA combination. The local `.venv/` is ignored by Git and uses CPU PyTorch for verification; it should not be uploaded as the cluster environment. OpenCV is required for resizing in the two supervised sweeps. It also affects connected-component corner-mask behavior and CLAHE; use the same OpenCV availability when comparing runs.

Tests use synthetic uint16 images and a small real timm ViT, with no dataset or pretrained-weight downloads. They cover checkpoint compatibility, row alignment, preprocessing, losses/gradients, nested budgets, training/resume, shared extraction, PCA/probes and all transfer modes. GPU/DDP execution still requires a cluster smoke run.

Supervised tests also exercise all three baseline entry points, ViT/ResNet construction, reports, label/preprocessing policies, split validation, multi-metric early stopping and assigned-device selection. GPU assignment is mocked locally; actual FP16/BF16 and concurrent GPU execution require a cluster run.
