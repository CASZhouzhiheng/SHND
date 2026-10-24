# SHND

**Selective Hypergraph Neural Diffusion for Reliable Brain Disorder Diagnosis from fMRI**

SHND learns disease predictions and prediction reliability from subject-specific resting-state fMRI hypergraphs. It combines higher-order functional interaction modeling, adaptive hypergraph diffusion, and dual contrastive learning to support selective diagnosis: the model predicts on accepted subjects and abstains on subjects with lower reliability scores.

## Overview

The framework consists of four components:

1. **Hypergraph construction.** Normalized temporal derivatives of ROI signals define node features. Pairwise and three-ROI interactions are evaluated using signed co-fluctuation statistics and temporal permutation tests.
2. **Hypergraph neural diffusion.** A learnable incidence-level modulation operator controls information propagation between nodes and hyperedges.
3. **Prediction and rejection.** A predictive head estimates disease probabilities, while a hypergraph-aware rejection head estimates prediction reliability.
4. **Dual contrastive learning.** Confidence-aware supervised contrastive learning (CSCL) shapes predictive representations, and reliability-aware contrastive learning (RCL) shapes rejection representations.

The training objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\lambda_c\mathcal{L}_{\mathrm{conf}}+\lambda_f\mathcal{L}_{\mathrm{CSCL}}+\lambda_g\mathcal{L}_{\mathrm{RCL}}.
$$

During warm-up, only classification and confidence-estimation losses are optimized. The two contrastive objectives are activated afterward.

## Repository Structure

```text
SHND/
├── src/shnd/
│   ├── construction.py       # Signed co-fluctuation statistics and permutation tests
│   ├── incidence.py          # Hypergraph gradient and adjoint operators
│   ├── model.py              # Diffusion encoder, predictive head, and rejection head
│   ├── losses.py             # CSCL and RCL objectives
│   ├── data.py               # Graph datasets, manifests, and stratified splits
│   ├── engine.py             # Training, checkpoint selection, and evaluation
│   ├── experiment.py         # Experiment orchestration and staged parameter search
│   └── metrics.py            # Coverage-dependent selective risk
├── scripts/
│   ├── 00_merge_manifests.py
│   ├── 01_build_hypergraphs.py
│   ├── 02_train_shnd.py
│   ├── 03_export_attention.py
│   └── 04_smoke_test.py
├── configs/
│   └── paper_search.json
├── tests/
│   └── test_paper_alignment.py
├── data/
│   └── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

Requirements: Python 3.10 or later, PyTorch 2.0 or later, and NumPy 1.24 or later.

Run the following commands from the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

For GPU execution, install a PyTorch build compatible with your CUDA environment. Both CPU and CUDA execution are supported through `--device`.

Run the test suite:

```bash
python scripts/04_smoke_test.py
```

The tests cover interaction statistics, diffusion operators, contrastive objectives, data splits, and synthetic training. No external dataset is required.

## Data Preparation

The construction pipeline expects **preprocessed ROI time series**. Image preprocessing and atlas-based ROI extraction should be completed before running these scripts.

Organize each dataset as follows:

```text
data/raw/ABIDE_I/
├── control/
│   └── timeseries/
│       ├── subject_001.txt
│       └── ...
└── patient/
    └── timeseries/
        ├── subject_002.txt
        └── ...
```

Each file should contain a time-by-ROI matrix. The default configuration uses 90 non-cerebellar AAL ROIs in a consistent order across subjects. Whitespace-delimited text, CSV, and TSV files are supported. Include only ROI signal columns; remove subject identifiers and time-index columns from the matrix. File stems serve as subject identifiers and must be unique within each dataset.

Labels are assigned as `control = 0` and `patient = 1`. Raw data, processed subject graphs, and trained checkpoints are not included in the repository.

## Hypergraph Construction

For each subject, the pipeline:

1. Computes first-order temporal derivatives and divides each ROI derivative sequence by its temporal standard deviation.
2. Enumerates all two-node and three-node candidate hyperedges.
3. Standardizes the temporal product of the normalized derivatives and applies a sign-consistency gate. Mixed-sign interactions are assigned zero at the corresponding time point.
4. Computes interaction strength as the mean absolute signed co-fluctuation.
5. Independently permutes each ROI sequence to construct the null distribution and retains hyperedges with empirical `p < 0.05`.

The default number of permutations is 10,000.

```bash
python scripts/01_build_hypergraphs.py --dataset ABIDE_I --input-root data/raw/ABIDE_I --output-root data/processed/ABIDE_I --num-rois 90 --permutations 10000 --alpha 0.05 --device cuda
```

Use `--device cpu` for CPU execution. Reduce `--edge-batch-size` or `--permutation-chunk` to lower memory usage; their defaults are 256 and 16, respectively.

The pipeline preserves the full node-feature sequence by default. Training requires a common feature width. If recordings have different lengths, explicitly select a common length using `--target-time`, for example:

```bash
python scripts/01_build_hypergraphs.py --dataset ABIDE_I --input-root data/raw/ABIDE_I --output-root data/processed/ABIDE_I_T128 --target-time 128 --device cuda
```

This option truncates or zero-pads node features to the requested length. Hyperedge statistics are still computed from the complete derivative sequences. Apply the same feature-length convention to source and target cohorts in cross-dataset experiments.

Each graph stores node features, retained hyperedges, empirical p-values, signed co-fluctuation sequences, interaction strengths, and construction metadata. A `manifest.csv` indexes the generated graphs. Construction failures are recorded in `errors.csv` and must be resolved before training or merging manifests. Graphs with no significant hyperedges raise an error. Use a fresh construction output directory for each run.

### Merge Dataset Manifests

Construct each dataset separately, then merge the manifests for multi-cohort experiments:

```bash
python scripts/00_merge_manifests.py --manifests data/processed/ABIDE_I/manifest.csv data/processed/ABIDE_II/manifest.csv --output data/processed/all_datasets.csv
```

The merger rebases graph paths and checks for duplicate dataset–subject identifiers.

## Training and Evaluation

### Within-Dataset Evaluation

Within-dataset experiments use stratified tenfold cross-validation. All methods share the same folds for a given seed. One fold is held out for testing, and a stratified validation subset is drawn from the remaining nine folds. The default validation fraction is `1/9` of the development set and can be changed with `--validation-fraction`.

```bash
python scripts/02_train_shnd.py --manifest data/processed/ABIDE_I/manifest.csv --dataset ABIDE_I --methods shnd sr --output-dir outputs/tenfold
```

Staged validation-set search is enabled by default. Each outer fold performs its own parameter search, followed by final training with the selected configuration. Test subjects are excluded from parameter and checkpoint selection.

### Training Defaults

| Setting | Default |
|---|---|
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Batch size | `64` |
| Training epochs | `150` |
| Warm-up epochs | `50` |
| Hidden dimension | `64` before search |
| Diffusion layers | `3` before search |
| Reliability-loss weight, `lambda_c` | `0.5` before search |
| Contrastive-loss weights, `lambda_f`, `lambda_g` | `0.05`, `0.05` before search |
| Temperatures, `tau_f`, `tau_g` | `0.2`, `0.2` before search |
| Reliability margin, `gamma` | `0.1` before search |
| Weight decay | `1e-4` before search |
| Contrastive sampling | Up to 3 positives and 3 negatives per anchor |
| Validation frequency | Every epoch |
| Checkpoint selection | Mean selective risk over the evaluated coverage levels |
| Random seed | `2024` |

Contrastive candidates are constructed within the current mini-batch. Sampling is uniform without replacement; smaller candidate sets use all available members. Anchors without positive candidates contribute zero while remaining in the batch-average denominator. Set `--contrastive-k 0` to use all candidates. Confidence weights are differentiable by default; `--detach-confidence` enables stop-gradient weighting.

### Hyperparameter Search

The default search configuration is defined in `configs/paper_search.json`:

| Stage | Parameters | Candidate values |
|---|---|---|
| 1 | Hidden dimension; diffusion depth | `{16, 32, 64}`; `{2, 3}` |
| 2 | Weight decay | `{0, 1e-4, 1e-3}` |
| 3 | `lambda_c` | `{0.1, 0.5, 1, 2}` |
| 4 | `lambda_f`; `lambda_g` | `{0.01, 0.05, 0.1}` each |
| 5 | `tau_f`; `tau_g` | `{0.1, 0.2, 0.5}` each |
| 6 | `gamma` | `{0.05, 0.1, 0.2}` |

Parameters selected at earlier stages are carried forward. Full SHND search uses 34 trials per fold, followed by one final training run. Use `--search-config path/to/config.json` to supply another staged search configuration.

To train with fixed hyperparameters, pass `--skip-search`:

```bash
python scripts/02_train_shnd.py --manifest data/processed/ABIDE_I/manifest.csv --dataset ABIDE_I --methods shnd --skip-search --hidden-dim 64 --layers 3 --lambda-c 0.5 --lambda-f 0.05 --lambda-g 0.05 --output-dir outputs/fixed_config
```

### Cross-Dataset Evaluation

Use the source cohort for training and validation, and the full target cohort for independent testing:

```bash
python scripts/02_train_shnd.py --manifest data/processed/all_datasets.csv --train-dataset ABIDE_I --test-dataset ABIDE_II --methods shnd sr --output-dir outputs/transfer
```

The source cohort is split into training and validation sets at a stratified ratio of 7:1. Hyperparameter selection uses only source validation data. Source and target node-feature dimensions must match.

### Methods and Ablations

| CLI name | Description |
|---|---|
| `shnd` | Full model with both contrastive objectives |
| `sr` | Diffusion encoder and predictive head trained with classification loss; MSP-based selection |
| `wo_cscl` | Without CSCL |
| `wo_rcl` | Without RCL |
| `wo_dual_cl` | Without either contrastive objective |
| `wo_warmup` | Activates all objectives from the first epoch |
| `pairwise_only` | Uses only retained two-node hyperedges |
| `uniform_readout` | Uses uniform hyperedge weights in the rejection readout |

Specify multiple methods after `--methods` to evaluate them on shared splits.

## Selective-Risk Metrics

Subjects are ranked by descending reliability score. SR uses maximum softmax probability (MSP). Evaluation reports classification error among accepted subjects at coverage levels:

```text
10%, 30%, 50%, 70%, 75%, 80%, 85%, 90%, 95%, 100%
```

Selective risk is reported as a **percentage**. The accepted-sample count is rounded upward, and output files include the realized coverage and accepted count. At 100% coverage, all subjects are evaluated.

SHND also exports MSP-ranked results using the same trained model and predictions. This differs from the separately trained `sr` baseline. SHND and its MSP-ranked evaluation therefore have identical risk at 100% coverage.

## Experiment Outputs

Runs are organized as:

```text
<output-dir>/<task>/<feature>/<method>/seed_<seed>/fold_<fold>/
```

Each final run includes:

| File | Contents |
|---|---|
| `split.csv` | Training, validation, and test subject assignments |
| `selected_config.json` | Selected training configuration |
| `training_history.csv` | Training losses and validation metrics |
| `validation_selection.json` | Best epoch and validation selection score |
| `best_model.pt` | Selected model checkpoint and configuration |
| `predictions.csv` | Subject-level predictions, probabilities, and selection scores |
| `coverage_risk.csv` | Selective risk, realized coverage, and accepted counts |
| `predictions_msp.csv` | SHND predictions with MSP selection scores |
| `coverage_risk_msp.csv` | MSP-ranked evaluation of the SHND model |

The last two files are generated for `shnd`. Search records are stored under `search/`, including `search_trace.json` and per-trial results. Task-level raw results and summaries are written to the output root. Within-dataset summaries report fold-wise means and population standard deviations (`ddof=0`) for each seed. Use a fresh output directory to avoid mixing experiments.

## Export Hyperedge Attention

Export rejection-head hyperedge attention for held-out test subjects using a saved fold assignment:

```bash
python scripts/03_export_attention.py --checkpoint outputs/fixed_config/ABIDE_I/x_derivative/shnd/seed_2024/fold_01/best_model.pt --manifest outputs/fixed_config/ABIDE_I/x_derivative/shnd/seed_2024/fold_01/split.csv --output outputs/attention/fold01_test.csv
```

The exporter uses `--split test` by default. Each row contains the subject identifier, diagnostic label, prediction, reliability score, hyperedge membership, and normalized rejection-readout weight `beta`. Both zero-based and one-based ROI indices are exported.

Repeat this procedure for all folds to obtain out-of-fold attention results. For inference on an ordinary graph manifest, explicitly use `--split all`. SR has no rejection head and does not support rejection-attention export.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details. Dataset access and redistribution are governed by the terms of the respective data providers.
