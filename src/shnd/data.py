from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .incidence import IncidenceOperator


def read_manifest(path: str | Path, datasets: set[str] | None = None) -> list[dict]:
    path = Path(path)
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
            edge_list = [tuple(map(int, edge)) for edge in graph["edge_list"]]
            if self.pairwise_only:
                edge_list = [edge for edge in edge_list if len(set(edge)) == 2]
            num_nodes = int(graph[self.feature_key].size(0))
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

