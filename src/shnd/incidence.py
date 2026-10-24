from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class IncidenceOperator:
    node_ids: torch.Tensor
    edge_ids: torch.Tensor
    inverse_sqrt_node_degree: torch.Tensor
    node_degree: torch.Tensor
    edge_sizes: torch.Tensor
    num_nodes: int
    num_edges: int

    @classmethod
    def from_edge_list(
        cls,
        edge_list: Iterable[Sequence[int]],
        num_nodes: int,
        device: torch.device | str = "cpu",
    ) -> "IncidenceOperator":
        nodes: list[int] = []
        edges: list[int] = []
        for edge_id, edge in enumerate(edge_list):
            clean = sorted({int(node) for node in edge})
            if len(clean) < 2 or len(clean) != len(edge) or any(v < 0 or v >= num_nodes for v in clean):
                raise ValueError("Invalid hyperedge membership: require distinct in-range nodes and cardinality >=2")
            nodes.extend(clean)
            edges.extend([edge_id] * len(clean))
        if not nodes:
            raise ValueError("No retained hyperedges")
        node_ids = torch.tensor(nodes, dtype=torch.long, device=device)
        original_edge_ids = torch.tensor(edges, dtype=torch.long, device=device)
        _, edge_ids = torch.unique(original_edge_ids, sorted=True, return_inverse=True)
        num_edges = int(edge_ids.max().item()) + 1
        node_degree = torch.bincount(node_ids, minlength=num_nodes).float().clamp_min(1.0)
        edge_sizes = torch.bincount(edge_ids, minlength=num_edges).float().clamp_min(1.0)
        return cls(
            node_ids=node_ids,
            edge_ids=edge_ids,
            inverse_sqrt_node_degree=node_degree.rsqrt(),
            node_degree=node_degree,
            edge_sizes=edge_sizes,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    def to(self, device: torch.device | str) -> "IncidenceOperator":
        device = torch.device(device)
        if self.node_ids.device == device:
            return self
        return IncidenceOperator(
            node_ids=self.node_ids.to(device, non_blocking=True),
            edge_ids=self.edge_ids.to(device, non_blocking=True),
            inverse_sqrt_node_degree=self.inverse_sqrt_node_degree.to(device, non_blocking=True),
            node_degree=self.node_degree.to(device, non_blocking=True),
            edge_sizes=self.edge_sizes.to(device, non_blocking=True),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
        )

    def edge_mean(self, features: torch.Tensor) -> torch.Tensor:
        output = features.new_zeros((self.num_edges, features.size(-1)))
        output.index_add_(0, self.edge_ids, features[self.node_ids])
        return output / self.edge_sizes.unsqueeze(-1)

    def gradient(self, features: torch.Tensor) -> torch.Tensor:
        normalized = features * self.inverse_sqrt_node_degree.unsqueeze(-1)
        edge_means = self.edge_mean(normalized)
        return normalized[self.node_ids] - edge_means[self.edge_ids]

    def divergence(self, incidence_features: torch.Tensor) -> torch.Tensor:
        scale = self.inverse_sqrt_node_degree[self.node_ids].unsqueeze(-1)
        direct = incidence_features.new_zeros((self.num_nodes, incidence_features.size(-1)))
        direct.index_add_(0, self.node_ids, incidence_features * scale)

        edge_sum = incidence_features.new_zeros((self.num_edges, incidence_features.size(-1)))
        edge_sum.index_add_(0, self.edge_ids, incidence_features)
        correction = incidence_features.new_zeros((self.num_nodes, incidence_features.size(-1)))
        correction.index_add_(
            0,
            self.node_ids,
            edge_sum[self.edge_ids] / self.edge_sizes[self.edge_ids].unsqueeze(-1) * scale,
        )
        return direct - correction

