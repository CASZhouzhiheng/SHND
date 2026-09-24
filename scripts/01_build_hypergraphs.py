from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shnd.construction import (  # noqa: E402
    finalize_edges,
    make_features,
    test_candidate_edges,
    zscore_columns,
)


def read_timeseries(path: Path, num_rois: int) -> np.ndarray:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    array = np.genfromtxt(path, delimiter=delimiter, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    array = array[~np.all(~np.isfinite(array), axis=1)]
    array = array[:, ~np.all(~np.isfinite(array), axis=0)]
    if array.shape[1] == num_rois + 1:
        array = array[:, 1:]
    if array.shape[0] == num_rois and array.shape[1] != num_rois:
        array = array.T
    if array.shape[1] < num_rois:
        raise ValueError(f"{path}: expected at least {num_rois} ROI columns, got {array.shape}")
    array = array[:, :num_rois]
    if array.shape[0] < 3 or not np.isfinite(array).all():
        raise ValueError(f"{path}: invalid or too-short time series with shape {array.shape}")
    return array


def discover(root: Path, group: str, patterns):
    directory = root / group / "timeseries"
    files = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Construct subject-level MTD hypergraphs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patterns", nargs="+", default=["*.txt", "*.csv", "*.tsv"])
    parser.add_argument("--num-rois", type=int, default=90)
    parser.add_argument("--target-time", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--edge-batch-size", type=int, default=4096)
    parser.add_argument("--permutation-chunk", type=int, default=128)
    parser.add_argument("--fallback-per-isolate", type=int, default=3)
    parser.add_argument("--min-edges", type=int, default=6000)
    parser.add_argument("--max-edges", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-group", type=int)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    errors_path = args.output_root / "errors.csv"
    manifest_rows = []
    error_rows = []
    subject_index = 0

    for group, label in (("control", 0), ("patient", 1)):
        files = discover(args.input_root, group, args.patterns)
        if args.limit_per_group is not None:
            files = files[: args.limit_per_group]
        for path in files:
            print(f"processing {args.dataset}/{group}/{path.name}", flush=True)
            try:
                raw = read_timeseries(path, args.num_rois)
                derivative_features, raw_features = make_features(raw, args.target_time)
                derivatives = zscore_columns(np.diff(raw, axis=0))
                significant = {}
                tested = []
                for order in (2, 3):
                    selected_order, tested_order = test_candidate_edges(
                        derivatives,
                        order=order,
                        permutations=args.permutations,
                        alpha=args.alpha,
                        edge_batch_size=args.edge_batch_size,
                        permutation_chunk=args.permutation_chunk,
                        seed=args.seed + 2 * subject_index + order,
                        device=args.device,
                    )
                    significant.update(selected_order)
                    tested.extend(tested_order)
                tested.sort(key=lambda item: (item[1], len(item[0]), item[0]))
                finalized = finalize_edges(
                    significant,
                    tested,
                    num_nodes=args.num_rois,
                    fallback_per_isolate=args.fallback_per_isolate,
                    min_edges=args.min_edges,
                    max_edges=args.max_edges,
                )
                edge_list = [edge for edge, _ in finalized]
                p_values = [p_value for _, p_value in finalized]
                subject_id = path.stem
                graph_path = args.output_root / "graphs" / group / f"{subject_id}.pt"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "x_derivative": torch.from_numpy(derivative_features),
                        "x_raw_zscore": torch.from_numpy(raw_features),
                        "edge_list": edge_list,
                        "edge_p_values": torch.tensor(p_values, dtype=torch.float32),
                        "label": label,
                        "dataset": args.dataset,
                        "group": group,
                        "subject_id": subject_id,
                    },
                    graph_path,
                )
                degree = np.zeros(args.num_rois, dtype=int)
                for edge in edge_list:
                    np.add.at(degree, np.asarray(edge), 1)
                manifest_rows.append(
                    {
                        "status": "ok",
                        "dataset": args.dataset,
                        "subject_id": subject_id,
                        "group": group,
                        "label": label,
                        "file": os.path.relpath(path, args.input_root),
                        "pt_path": os.path.relpath(graph_path, args.output_root),
                        "num_rois": args.num_rois,
                        "raw_time": raw.shape[0],
                        "significant_edges": len(significant),
                        "final_edges": len(edge_list),
                        "isolated_nodes": " ".join(map(str, np.flatnonzero(degree == 0))),
                    }
                )
            except Exception as error:
                error_rows.append(
                    {
                        "dataset": args.dataset,
                        "group": group,
                        "file": str(path),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                print(f"ERROR {path}: {error}", flush=True)
            subject_index += 1

    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    if error_rows:
        with errors_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(error_rows[0]))
            writer.writeheader()
            writer.writerows(error_rows)
    print(f"saved {len(manifest_rows)} graphs to {args.output_root}")


if __name__ == "__main__":
    main()

