from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .data import read_manifest, stratified_tenfold, cross_dataset_split, HypergraphDataset
from .engine import train_run, ABLATIONS, _write_csv
from .metrics import DEFAULT_COVERAGES


def staged_search(train, validation, config, method, seed, directory, stages):
    current = dict(config)
    trace = []
    for index, stage in enumerate(stages):
        keys = list(stage)
        if not keys or any(k not in current for k in keys) or any(not stage[k] for k in keys):
            raise ValueError(f"Invalid search stage: {stage}")
        if method == "sr" and any(k.startswith("lambda_") or k in ("tau_f", "tau_g", "gamma") for k in keys):
            continue
        if method in ("wo_cscl", "wo_dual_cl") and any(k in ("lambda_f", "tau_f") for k in keys):
            stage = {k: ([current[k]] if k in ("lambda_f", "tau_f") else v) for k, v in stage.items()}
        if method in ("wo_rcl", "wo_dual_cl") and any(k in ("lambda_g", "tau_g", "gamma") for k in keys):
            stage = {k: ([current[k]] if k in ("lambda_g", "tau_g", "gamma") else v) for k, v in stage.items()}
        best_score, best = float("inf"), None
        for trial, values in enumerate(itertools.product(*(stage[k] for k in keys))):
            candidate = dict(current, **dict(zip(keys, values)))
            result = train_run(train, validation, None, candidate, method, seed,
                               directory / f"stage_{index+1}" / f"trial_{trial+1}")
            score = result["validation_score"]
            trace.append({"stage": index+1, "trial": trial+1, "parameters": dict(zip(keys, values)),
                          "validation_score": score})
            if score < best_score:
                best_score, best = score, candidate
        current = best
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "search_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return current


def main():
    parser = argparse.ArgumentParser(description="SHND.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--train-dataset")
    parser.add_argument("--test-dataset")
    parser.add_argument("--methods", nargs="+", choices=sorted(ABLATIONS), default=["shnd"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--feature", choices=["x_derivative", "x_raw_zscore"], default="x_derivative")
    parser.add_argument("--input-dim", type=int, help="Inferred from training features if omitted")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-c", type=float, default=0.5)
    parser.add_argument("--lambda-f", type=float, default=0.05)
    parser.add_argument("--lambda-g", type=float, default=0.05)
    parser.add_argument("--tau-f", type=float, default=0.2)
    parser.add_argument("--tau-g", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--contrastive-k", type=int, default=3, help="0: full candidate-set formula")
    parser.add_argument("--detach-confidence", action="store_true", help="Alternative to differentiating Eq. (45)")
    parser.add_argument("--gradient-clip", type=float, default=0")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--selection-metric", choices=["mean_risk", "lowcov", "risk95"], default="mean_risk")
    parser.add_argument("--validation-fraction", type=float, default=1/9)
    parser.add_argument("--search-config", type=Path,
                        default=Path(__file__).resolve().parents[2]/"configs/paper_search.json")
    parser.add_argument("--skip-search", action="store_true", help="Fixed configuration experiment, not tuned replication")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shnd_paper"))
    args = parser.parse_args()
    if bool(args.dataset) == bool(args.train_dataset or args.test_dataset):
        parser.error("use --dataset OR both --train-dataset and --test-dataset")
    if (args.train_dataset is None) != (args.test_dataset is None):
        parser.error("both cross-dataset arguments are required")
    if args.train_dataset and args.train_dataset == args.test_dataset:
        parser.error("source and independent target must differ")
    config = {k: getattr(args, k) for k in (
        "feature", "input_dim", "hidden_dim", "num_classes", "epochs", "warmup_epochs", "batch_size",
        "learning_rate", "weight_decay", "lambda_c", "lambda_f", "lambda_g", "tau_f", "tau_g", "gamma",
        "contrastive_k", "eval_every", "selection_metric", "gradient_clip", "device", "detach_confidence")}
    config["num_layers"] = args.layers
    required = {args.dataset} if args.dataset else {args.train_dataset, args.test_dataset}
    rows = read_manifest(args.manifest, required)
    if set(r["dataset"] for r in rows) != required:
        parser.error("Missing requested cohort in manifest")
    data = HypergraphDataset(rows, args.feature)
    widths = {data[i]["x"].shape[1] for i in range(len(data))}
    if len(widths) != 1:
        parser.error("Variable feature lengths: specify and document a common --target-time when building graphs")
    width = widths.pop()
    if args.input_dim is not None and args.input_dim != width:
        parser.error("--input-dim differs from cached node feature width")
    config["input_dim"] = width
    del data
    stages = [] if args.skip_search else json.loads(args.search_config.read_text(encoding="utf-8"))["stages"]
    task = args.dataset or f"{args.train_dataset}_to_{args.test_dataset}"
    results = []
    for seed in args.seeds:
        splits = list(stratified_tenfold(rows, seed, args.validation_fraction)) if args.dataset else [
            (0, *cross_dataset_split(rows, args.train_dataset, args.test_dataset, seed))]
        for fold, train, validation, test in splits:
            for method in args.methods:
                run = args.output_dir/task/args.feature/method/f"seed_{seed}"/f"fold_{fold+1:02d}"
                if run.exists():
                    raise FileExistsError(f"Use a fresh output directory: {run}")
                selected = staged_search(train, validation, config, method, seed, run/"search", stages)
                (run/"selected_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
                risks = train_run(train, validation, test, selected, method, seed, run)
                results.append(dict(task=task, method=method, seed=seed, fold=fold+1,
                                    **{f"risk_{c}": v for c, v in risks.items()}))
                _write_csv(args.output_dir/f"{task}_raw_results.csv", results)
    summary = []
    for seed in args.seeds:
        for method in args.methods:
            subset = [r for r in results if r["method"] == method and r["seed"] == seed]
            row = dict(task=task, method=method, seed=seed, folds=len(subset), risk_unit="percent")
            for c in DEFAULT_COVERAGES:
                values = np.array([r[f"risk_{c}"] for r in subset])
                row[f"risk_{c}_mean"] = float(values.mean())
                row[f"risk_{c}_std"] = float(values.std(ddof=0))
            summary.append(row)
    _write_csv(args.output_dir/f"{task}_summary.csv", summary)
