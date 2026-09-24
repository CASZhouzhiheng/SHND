# Example commands

## Build five datasets

```powershell
$datasets = "ABIDE_I", "ABIDE_II", "ADHD200", "MDD", "ADNI"
foreach ($dataset in $datasets) {
  python scripts/01_build_hypergraphs.py `
    --dataset $dataset `
    --input-root "data/raw/$dataset" `
    --output-root "data/processed/$dataset" `
    --device cuda
}
```

## Run the complete ablation study

```powershell
python scripts/02_train_shnd.py `
  --manifest data/processed/all_datasets_manifest.csv `
  --dataset ABIDE_I `
  --methods shnd wo_cscl wo_rcl wo_dual_cl wo_warmup pairwise_only uniform_readout `
  --seeds 2024 2025 2026 2027 2028 `
  --epochs 150 `
  --warmup-epochs 50
```

## Use standardized raw time series

```powershell
python scripts/02_train_shnd.py `
  --manifest data/processed/all_datasets_manifest.csv `
  --dataset ABIDE_I `
  --feature x_raw_zscore `
  --methods sr shnd
```

