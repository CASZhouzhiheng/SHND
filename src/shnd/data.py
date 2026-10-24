from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .incidence import IncidenceOperator
from .construction import GRAPH_SCHEMA


def read_manifest(path: str | Path, datasets: set[str] | None = None) -> list[dict]:
    path = Path(path)
    if (path.parent / "errors.csv").exists():
        raise ValueError("Construction errors.csv exists; resolve all failed subjects before using this manifest")
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "ok") != "ok":
                continue
            if datasets and row["dataset"] not in datasets:
                continue
            pt_path = Path(row["pt_path"])
            if not pt_path.is_absolute():
                pt_path = path.parent / pt_path
            row["pt_path"] = str(pt_path)
            rows.append(row)
    keys = [(r["dataset"], r["subject_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset/subject_id in manifest")
    return rows


class HypergraphDataset(Dataset):
    def __init__(self, rows, feature_key: str = "x_derivative", pairwise_only: bool = False):
        self.rows = list(rows)
        self.feature_key = feature_key
        self.pairwise_only = pairwise_only
        self._cache: dict[tuple[str, bool], dict] = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        cache_key = (row["pt_path"], self.pairwise_only)
        if cache_key not in self._cache:
            graph = torch.load(row["pt_path"], map_location="cpu", weights_only=False)
            if graph.get("schema") != GRAPH_SCHEMA:
                raise ValueError(f"{row['pt_path']}: legacy graph; rebuild using corrected Eqs. (8)-(17)")
            edge_list = [tuple(map(int, edge)) for edge in graph["edge_list"]]
            if self.pairwise_only:
                edge_list = [edge for edge in edge_list if len(set(edge)) == 2]
            features = graph[self.feature_key]
            if features.ndim != 2 or min(features.shape) < 1 or not torch.isfinite(features).all():
                raise ValueError(f"{row['pt_path']}: invalid node features")
            if "label" in graph and int(graph["label"]) != int(row["label"]):
                raise ValueError(f"{row['pt_path']}: manifest/cache label mismatch")
            num_nodes = int(features.size(0))
            self._cache[cache_key] = {
                "features": graph[self.feature_key].float(),
                "edge_list": edge_list,
                "incidence": IncidenceOperator.from_edge_list(edge_list, num_nodes),
            }
        graph = self._cache[cache_key]
        return {
            "x": graph["features"],
            "edge_list": graph["edge_list"],
            "incidence": graph["incidence"],
            "label": int(row["label"]),
            "dataset": row["dataset"],
            "group": row.get("group", ""),
            "subject_id": row.get("subject_id", Path(row.get("file", row["pt_path"])).stem),
            "source_file": row.get("file", ""),
        }


def collate_graphs(batch):
    return batch


def stratified_split(rows, seed: int, ratios=(0.7, 0.1, 0.2)):
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to one")
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    splits = [[], [], []]
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        n_train = round(len(label_rows) * ratios[0])
        n_val = round(len(label_rows) * ratios[1])
        splits[0].extend(label_rows[:n_train])
        splits[1].extend(label_rows[n_train : n_train + n_val])
        splits[2].extend(label_rows[n_train + n_val :])
    for split in splits:
        rng.shuffle(split)
    return tuple(splits)


def cross_dataset_split(rows, train_dataset: str, test_dataset: str, seed: int):
    source = [row for row in rows if row["dataset"] == train_dataset]
    target = [row for row in rows if row["dataset"] == test_dataset]
    train, val, _ = stratified_split(source, seed, ratios=(0.875, 0.125, 0.0))
    return train, val, target



def stratified_tenfold(rows, seed=2024, validation_fraction=1/9):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0,1)")
    keys = [(r["dataset"], r["subject_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate subjects in cohort")
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["label"]), []).append(row)
    if len(grouped) < 2 or min(map(len, grouped.values())) < 10:
        raise ValueError("Stratified tenfold requires at least 10 subjects in each class")
    rng = random.Random(seed)
    folds = [[] for _ in range(10)]
    offset = 0
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda r: (r["dataset"], r["subject_id"]))
        rng.shuffle(ordered)
        for j, row in enumerate(ordered):
            folds[(offset+j) % 10].append(row)
        offset = (offset+len(ordered)) % 10
    for index in range(10):
        development = [r for j, fold in enumerate(folds) if j != index for r in fold]
        train, val, _ = stratified_split(development, seed+index,
                                        (1-validation_fraction, validation_fraction, 0))
        if not train or not val:
            raise ValueError("Inner split is empty")
        yield index, train, val, folds[index]
