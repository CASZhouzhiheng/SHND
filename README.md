# SHND

Official-style research code for **Selective Hypergraph Neural Diffusion (SHND)** on resting-state fMRI hypergraphs.

SHND combines an adaptive hypergraph neural diffusion encoder, a predictive head, and a hypergraph-aware rejection head. The repository includes MTD-based hypergraph construction, single-dataset and cross-dataset training, selective-risk evaluation, ablation settings, and subject-level hyperedge-attention export.

> Raw neuroimaging data, generated hypergraphs, checkpoints, and experimental outputs are intentionally not included.

## Repository layout

```text
SHND-GitHub/
  src/shnd/
    construction.py       # MTD permutation testing and edge completion
    incidence.py          # Matrix-free G_H and G_H^T operators
    model.py              # Adaptive HND encoder and dual heads
    losses.py             # CSCL and RCL objectives
    data.py               # Manifest datasets and data splits
    metrics.py            # Coverage-risk evaluation
    engine.py             # Training and checkpoint selection
  scripts/
    00_merge_manifests.py
    01_build_hypergraphs.py
    02_train_shnd.py
    03_export_attention.py
    04_smoke_test.py
  configs/example_commands.md
  data/README.md
  tests/test_model.py
  requirements.txt
  pyproject.toml
```

## Environment

Python 3.10 or newer and PyTorch 2.0 or newer are recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the code-only smoke test:

```bash
python scripts/04_smoke_test.py
```

## Expected raw-data layout

Each dataset uses the same binary-group organization. Files may be whitespace-delimited text, CSV, or TSV matrices. The expected orientation is time points by ROIs; ROI-by-time matrices are automatically transposed when unambiguous.

```text
data/raw/ABIDE_I/
  control/timeseries/<subject files>
  patient/timeseries/<subject files>
data/raw/ABIDE_II/
  control/timeseries/<subject files>
  patient/timeseries/<subject files>
```

The first 90 columns are retained when an AAL116 time series is supplied. See [data/README.md](data/README.md) for the processed graph format.

## 1. Construct subject-level hypergraphs

The construction script computes first-order temporal derivatives, tests all second- and third-order MTD candidates using permutation tests, retains edges with `p < 0.1`, adds the three lowest-p incident candidates for any isolated ROI, fills the graph in ascending p-value order when fewer than 6,000 edges remain, and caps the graph at 12,000 edges.

```bash
python scripts/01_build_hypergraphs.py \
  --dataset ABIDE_I \
  --input-root data/raw/ABIDE_I \
  --output-root data/processed/ABIDE_I \
  --num-rois 90 \
  --target-time 128 \
  --permutations 500 \
  --alpha 0.1 \
  --fallback-per-isolate 3 \
  --min-edges 6000 \
  --max-edges 12000 \
  --device cuda
```

Repeat the command for each cohort, then merge the generated manifests while safely rebasing their graph paths:

```bash
python scripts/00_merge_manifests.py \
  --manifests \
    data/processed/ABIDE_I/manifest.csv \
    data/processed/ABIDE_II/manifest.csv \
    data/processed/ADHD200/manifest.csv \
    data/processed/MDD/manifest.csv \
    data/processed/ADNI/manifest.csv \
  --output data/processed/all_datasets_manifest.csv
```

## 2. Train and evaluate SHND

Single-dataset experiments use a stratified `7:1:2` train/validation/test split. Checkpoint selection is performed on validation selective risk, never on the test set. The default coverage levels are `{10, 30, 50, 70, 75, 80, 85, 90, 95, 99}%`.

```bash
python scripts/02_train_shnd.py \
  --manifest data/processed/all_datasets_manifest.csv \
  --dataset ABIDE_I \
  --feature x_derivative \
  --methods sr shnd \
  --seeds 2024 2025 2026 2027 2028 \
  --hidden-dim 64 \
  --layers 3 \
  --epochs 150 \
  --warmup-epochs 50 \
  --batch-size 16 \
  --lambda-c 0.5 \
  --lambda-f 0.05 \
  --lambda-g 0.05 \
  --tau-f 0.2 \
  --tau-g 0.2 \
  --gamma 0.1 \
  --contrastive-k 3 \
  --output-dir outputs/shnd
```

Available methods are:

- `sr`: softmax response with the shared diffusion backbone
- `shnd`: complete model
- `wo_cscl`: without classification-space contrastive learning
- `wo_rcl`: without rejection-space contrastive learning
- `wo_dual_cl`: without both contrastive objectives
- `wo_warmup`: starts all objectives at epoch 1
- `pairwise_only`: removes hyperedges containing more than two ROIs
- `uniform_readout`: replaces learned rejection readout weights with uniform weights

### Cross-dataset evaluation

Train on ABIDE-I and evaluate directly on ABIDE-II:

```bash
python scripts/02_train_shnd.py \
  --manifest data/processed/all_datasets_manifest.csv \
  --train-dataset ABIDE_I \
  --test-dataset ABIDE_II \
  --feature x_derivative \
  --methods sr shnd \
  --seeds 2024 2025 2026 2027 2028
```

## 3. Export rejection-head hyperedge attention

The exporter saves the reliability score, prediction, hyperedge membership, and normalized rejection-head attention coefficient `beta` for every subject-hyperedge pair.

```bash
python scripts/03_export_attention.py \
  --checkpoint outputs/shnd/ABIDE_I/x_derivative/shnd/seed_2024/best_model.pt \
  --manifest data/processed/all_datasets_manifest.csv \
  --dataset ABIDE_I \
  --output outputs/attention/ABIDE_I_seed2024.csv
```

These exported values can be used for rejected-vs-accepted, incorrect-vs-correct, and disease-related high-order interaction analyses.

## Model overview

For node features `X` with shape `|V| x T`, SHND first obtains `Z^(0) = ReLU(X W)`. Incidence-level compatibility scores between nodes and their incident hyperedges are normalized over the hyperedges attached to each node. The encoder then performs adaptive hypergraph diffusion without explicitly materializing a dense incidence matrix. Mean node pooling feeds the predictive head, while an attention-weighted hyperedge readout feeds the rejection head.

The training objective is

```text
L = L_CE + lambda_c L_rejection + lambda_f L_CSCL + lambda_g L_RCL.
```

During warm-up, only `L_CE` and `L_rejection` are optimized. CSCL and RCL are enabled after the warm-up period.

## Reproducibility notes

- Generated `.pt` graphs retain the original hyperedge list and edge-level permutation p-values.
- Model checkpoints contain both architecture and training configurations.
- All reported selective risks rank subjects from high to low reliability. SR uses maximum softmax probability; SHND uses the rejection-head output.
- Hyperparameter selection should be completed using the validation split only.
- Acquisition-site information should be retained as metadata and not used as an input feature unless explicitly studying site-aware models.

## License

This code is released under the MIT License. Dataset access and redistribution remain subject to the terms of the original data providers.
