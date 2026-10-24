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


def normalized_derivatives(time_by_roi):
    raw = np.asarray(time_by_roi, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] < 3 or not np.isfinite(raw).all():
        raise ValueError("Expected finite T-by-ROI data, at least three time points")
    dx = np.diff(raw, axis=0)
    std = dx.std(axis=0, keepdims=True)
    return np.divide(dx, std, out=np.zeros_like(dx), where=std > 1e-8)


def make_features(time_by_roi, target_time=None):
    raw = zscore_columns(time_by_roi).T
    derivative = normalized_derivatives(time_by_roi).T
    if target_time is None:
        return derivative, raw
    if target_time < 1:
        raise ValueError("target_time must be positive")
    return truncate_or_pad(derivative, target_time), truncate_or_pad(raw, target_time)


def signed_cofluctuation(q, edges):
    components = q[..., edges].movedim(-3, -1)
    product = components.prod(dim=-2)
    std = product.std(dim=-1, unbiased=False, keepdim=True)
    xi = (product-product.mean(dim=-1, keepdim=True)) / std.clamp_min(1e-8)
    xi = torch.where(std > 1e-8, xi, 0.0)
    pos = (components >= 0).all(dim=-2)
    neg = (components <= 0).all(dim=-2)
    sign = torch.where(pos, 1.0, torch.where(neg, -1.0, 0.0))
    return sign * xi.abs()


def _observed_strength(derivatives, edges):
    return signed_cofluctuation(derivatives, edges).abs().mean(dim=-1)


def _null_strength(permuted, edges):
    return _observed_strength(permuted, edges)


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
    if order not in (2, 3) or min(permutations, edge_batch_size, permutation_chunk) < 1:
        raise ValueError("Use cardinality 2 or 3 and positive permutation/batch counts")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    values = torch.as_tensor(derivatives, dtype=torch.float32, device=device)
    if values.ndim != 2 or values.shape[0] < 2 or not torch.isfinite(values).all():
        raise ValueError("derivatives must be finite time-by-node data")
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
        generator = torch.Generator(device=device).manual_seed(seed)
        completed = 0
        while completed < permutations:
            current = min(permutation_chunk, permutations - completed)
            random_values = torch.rand(
                (current, num_nodes, num_time), generator=generator, device=device
            )
            indices = torch.argsort(random_values, dim=-1).transpose(1, 2)
            shuffled = torch.gather(values.unsqueeze(0).expand(current, -1, -1), 1, indices)
            exceedances += (_null_strength(shuffled, edges) >= observed.unsqueeze(0)).sum(0)
            completed += current
        p_values = (1 + exceedances.double()) / (permutations + 1)
        for edge, p_value in zip(edges.cpu().tolist(), p_values.cpu().tolist()):
            tested.append((tuple(edge), float(p_value)))
    tested.sort(key=lambda item: (item[1], len(item[0]), item[0]))
    significant = {edge: p for edge, p in tested if p < alpha}
    return significant, tested


def finalize_edges(significant, tested=None, num_nodes=None):
    return sorted(significant.items(), key=lambda item: (item[1], len(item[0]), item[0]))
