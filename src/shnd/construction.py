from __future__ import annotations

import itertools

import numpy as np
import torch


def zscore_columns(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    output = np.zeros_like(array)
    np.divide(array - mean, std, out=output, where=std > 1e-8)
    return output


def truncate_or_pad(features: np.ndarray, target_time: int) -> np.ndarray:
    if features.shape[1] >= target_time:
        return features[:, :target_time].astype(np.float32, copy=False)
    output = np.zeros((features.shape[0], target_time), dtype=np.float32)
    output[:, : features.shape[1]] = features
    return output


def make_features(time_by_roi: np.ndarray, target_time: int):
    raw = zscore_columns(time_by_roi).T
    derivative = zscore_columns(np.diff(time_by_roi, axis=0)).T
    return truncate_or_pad(derivative, target_time), truncate_or_pad(raw, target_time)


def _strength(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, unbiased=False, keepdim=True)
    normalized = torch.where(std > 1e-8, (values - mean) / std.clamp_min(1e-8), 0.0)
    return normalized.abs().mean(dim=-1)


def _observed_strength(derivatives: torch.Tensor, edges: torch.Tensor):
    values = derivatives[:, edges[:, 0]].T * derivatives[:, edges[:, 1]].T
    if edges.size(1) == 3:
        values = values * derivatives[:, edges[:, 2]].T
    return _strength(values)


def _null_strength(permuted: torch.Tensor, edges: torch.Tensor):
    values = permuted[:, :, edges[:, 0]].permute(0, 2, 1)
    values = values * permuted[:, :, edges[:, 1]].permute(0, 2, 1)
    if edges.size(1) == 3:
        values = values * permuted[:, :, edges[:, 2]].permute(0, 2, 1)
    return _strength(values)


def test_candidate_edges(
    derivatives: np.ndarray,
    order: int,
    permutations: int,
    alpha: float,
    edge_batch_size: int,
    permutation_chunk: int,
    seed: int,
    device: str,
):
    values = torch.as_tensor(derivatives, dtype=torch.float32, device=device)
    num_time, num_nodes = values.shape
    candidates = itertools.combinations(range(num_nodes), order)
    generator = torch.Generator(device=device).manual_seed(seed)
    tested = []
    while True:
        batch = list(itertools.islice(candidates, edge_batch_size))
        if not batch:
            break
        edges = torch.tensor(batch, dtype=torch.long, device=device)
        observed = _observed_strength(values, edges)
        exceedances = torch.zeros(edges.size(0), dtype=torch.int32, device=device)
        completed = 0
        while completed < permutations:
            current = min(permutation_chunk, permutations - completed)
            random_values = torch.rand(
                (current, num_time, num_nodes), generator=generator, device=device
            )
            indices = torch.argsort(random_values, dim=1)
            shuffled = torch.gather(values.unsqueeze(0).expand(current, -1, -1), 1, indices)
            exceedances += (_null_strength(shuffled, edges) >= observed.unsqueeze(0)).sum(0)
            completed += current
        p_values = (1 + exceedances.float()) / (permutations + 1)
        for edge, p_value in zip(edges.cpu().tolist(), p_values.cpu().tolist()):
            tested.append((tuple(edge), float(p_value)))
    tested.sort(key=lambda item: (item[1], len(item[0]), item[0]))
    significant = {edge: p for edge, p in tested if p < alpha}
    return significant, tested


def finalize_edges(
    significant: dict,
    tested: list[tuple[tuple[int, ...], float]],
    num_nodes: int,
    fallback_per_isolate: int = 3,
    min_edges: int = 6000,
    max_edges: int = 12000,
):
    selected = dict(significant)

    def degrees():
        degree = np.zeros(num_nodes, dtype=np.int64)
        for edge in selected:
            np.add.at(degree, np.asarray(edge), 1)
        return degree

    degree = degrees()
    for node in np.flatnonzero(degree == 0):
        incident = [(edge, p) for edge, p in tested if node in edge and edge not in selected]
        for edge, p_value in incident[:fallback_per_isolate]:
            selected[edge] = p_value
        degree = degrees()

    for edge, p_value in tested:
        if len(selected) >= min_edges:
            break
        selected.setdefault(edge, p_value)

    ordered = sorted(selected.items(), key=lambda item: (item[1], len(item[0]), item[0]))
    if max_edges <= 0 or len(ordered) <= max_edges:
        return ordered

    kept = dict(ordered[:max_edges])

    def kept_degrees():
        degree = np.zeros(num_nodes, dtype=np.int64)
        for edge in kept:
            np.add.at(degree, np.asarray(edge), 1)
        return degree

    degree = kept_degrees()
    isolated = set(np.flatnonzero(degree == 0).tolist())
    for edge, p_value in ordered[max_edges:]:
        if isolated.intersection(edge):
            kept[edge] = p_value
            degree = kept_degrees()
            isolated = set(np.flatnonzero(degree == 0).tolist())
            if not isolated:
                break

    while len(kept) > max_edges:
        removable = None
        for edge, p_value in sorted(
            kept.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True
        ):
            trial = degree.copy()
            np.add.at(trial, np.asarray(edge), -1)
            if np.all(trial > 0):
                removable = edge
                degree = trial
                break
        if removable is None:
            break
        del kept[removable]
    return sorted(kept.items(), key=lambda item: (item[1], len(item[0]), item[0]))
