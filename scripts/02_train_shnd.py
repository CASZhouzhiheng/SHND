from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shnd.data import cross_dataset_split, read_manifest, stratified_split  # noqa: E402
from shnd.engine import train_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SHND.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--train-dataset")
    parser.add_argument("--test-dataset")
    parser.add_argument("--feature", choices=["x_derivative", "x_raw_zscore"], default="x_derivative")
    parser.add_argument("--methods", nargs="+", default=["sr", "shnd"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-c", type=float, default=0.5)
    parser.add_argument("--lambda-f", type=float, default=0.05)
    parser.add_argument("--lambda-g", type=float, default=0.05)
    parser.add_argument("--tau-f", type=float, default=0.2)
    parser.add_argument("--tau-g", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--contrastive-k", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["lowcov", "aurc", "risk95"], default="lowcov")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shnd"))
    args = parser.parse_args()

    if bool(args.dataset) == bool(args.train_dataset or args.test_dataset):
        parser.error("use either --dataset or both --train-dataset and --test-dataset")
    if (args.train_dataset is None) != (args.test_dataset is None):
        parser.error("--train-dataset and --test-dataset must be used together")

    required = {args.dataset} if args.dataset else {args.train_dataset, args.test_dataset}
    rows = read_manifest(args.manifest, required)
    config = {
        "feature": args.feature,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.layers,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda_c": args.lambda_c,
        "lambda_f": args.lambda_f,
        "lambda_g": args.lambda_g,
        "tau_f": args.tau_f,
        "tau_g": args.tau_g,
        "gamma": args.gamma,
        "contrastive_k": args.contrastive_k,
        "eval_every": args.eval_every,
        "selection_metric": args.selection_metric,
        "gradient_clip": 5.0,
        "device": args.device,
    }
    result_rows = []
    task = args.dataset or f"{args.train_dataset}_to_{args.test_dataset}"
    for method in args.methods:
        for seed in args.seeds:
            if args.dataset:
                task_rows = [row for row in rows if row["dataset"] == args.dataset]
                train, validation, test = stratified_split(task_rows, seed)
            else:
                train, validation, test = cross_dataset_split(
                    rows, args.train_dataset, args.test_dataset, seed
                )
            run_dir = args.output_dir / task / args.feature / method / f"seed_{seed}"
            print(
                f"task={task} method={method} seed={seed} "
                f"train={len(train)} val={len(validation)} test={len(test)}",
                flush=True,
            )
            risks = train_run(train, validation, test, config, method, seed, run_dir)
            result_rows.append(
                {"task": task, "feature": args.feature, "method": method, "seed": seed,
                 **{f"risk_{coverage}": risk for coverage, risk in risks.items()}}
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{task}_raw_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = []
    for method in args.methods:
        subset = [row for row in result_rows if row["method"] == method]
        row = {"task": task, "feature": args.feature, "method": method}
        for coverage in (10, 30, 50, 70, 75, 80, 85, 90, 95, 99):
            values = np.asarray([item[f"risk_{coverage}"] for item in subset])
            row[f"risk_{coverage}_mean"] = values.mean()
            row[f"risk_{coverage}_std"] = values.std(ddof=0)
        summary.append(row)
    with (args.output_dir / f"{task}_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
