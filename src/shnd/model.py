from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .incidence import IncidenceOperator


def grouped_softmax(scores: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Apply softmax independently to entries sharing the same group id."""

    maxima = scores.new_full((num_groups,), -torch.inf)
    maxima.scatter_reduce_(0, group_ids, scores, reduce="amax", include_self=True)
    exponentials = torch.exp(scores - maxima[group_ids])
    denominators = scores.new_zeros((num_groups,))
    denominators.index_add_(0, group_ids, exponentials)
    return exponentials / denominators[group_ids].clamp_min(1e-12)


class AdaptiveHNDEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.incidence_score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_layers = num_layers
        self.step_logits = nn.Parameter(torch.full((num_layers,), 0.2))

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: IncidenceOperator,
    ) -> tuple[torch.Tensor, torch.Tensor, IncidenceOperator]:
        incidence = incidence.to(node_features.device)
        z0 = F.relu(self.input_projection(node_features))
        edge_z0 = incidence.edge_mean(z0)
        pairs = torch.cat(
            [z0[incidence.node_ids], edge_z0[incidence.edge_ids]],
            dim=-1,
        )
        raw_scores = self.incidence_score(pairs).squeeze(-1)
        incidence_attention = grouped_softmax(raw_scores, incidence.node_ids, incidence.num_nodes)

        z = z0
        for layer in range(self.num_layers):
            gradient = incidence.gradient(z)
            step = torch.sigmoid(self.step_logits[layer])
            z = z - step * incidence.divergence(incidence_attention.unsqueeze(-1) * gradient)
        return z, incidence_attention, incidence


class SHNDModel(nn.Module):
    """Adaptive hypergraph diffusion with predictive and rejection heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        uniform_rejection_readout: bool = False,
    ):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "uniform_rejection_readout": uniform_rejection_readout,
        }
        self.encoder = AdaptiveHNDEncoder(input_dim, hidden_dim, num_layers)
        self.predictive_head = nn.Linear(hidden_dim, num_classes)
        self.rejection_head = nn.Linear(hidden_dim, 1)
        self.uniform_rejection_readout = uniform_rejection_readout

    @staticmethod
    def _batch_incidence(batch, device: torch.device):
        node_chunks = []
        edge_chunks = []
        edge_graph_chunks = []
        node_offset = 0
        edge_offset = 0
        node_counts = []
        for graph_id, item in enumerate(batch):
            incidence = item["incidence"]
            node_chunks.append(incidence.node_ids.to(device) + node_offset)
            edge_chunks.append(incidence.edge_ids.to(device) + edge_offset)
            edge_graph_chunks.append(
                torch.full((incidence.num_edges,), graph_id, dtype=torch.long, device=device)
            )
            node_counts.append(incidence.num_nodes)
            node_offset += incidence.num_nodes
            edge_offset += incidence.num_edges

        node_ids = torch.cat(node_chunks)
        edge_ids = torch.cat(edge_chunks)
        edge_graph_ids = torch.cat(edge_graph_chunks)
        node_degree = torch.bincount(node_ids, minlength=node_offset).float().clamp_min(1.0)
        edge_sizes = torch.bincount(edge_ids, minlength=edge_offset).float().clamp_min(1.0)
        incidence = IncidenceOperator(
            node_ids=node_ids,
            edge_ids=edge_ids,
            inverse_sqrt_node_degree=node_degree.rsqrt(),
            node_degree=node_degree,
            edge_sizes=edge_sizes,
            num_nodes=node_offset,
            num_edges=edge_offset,
        )
        return incidence, edge_graph_ids, node_counts

    def forward(self, batch):
        if not batch:
            raise ValueError("SHNDModel received an empty batch")
        device = next(self.parameters()).device
        node_features = torch.cat([item["x"].to(device) for item in batch], dim=0)
        incidence, edge_graph_ids, node_counts = self._batch_incidence(batch, device)
        node_embeddings, incidence_attention, incidence = self.encoder(node_features, incidence)

        graph_ids = torch.repeat_interleave(
            torch.arange(len(batch), device=device),
            torch.tensor(node_counts, device=device),
        )
        predictive_embedding = node_embeddings.new_zeros((len(batch), node_embeddings.size(-1)))
        predictive_embedding.index_add_(0, graph_ids, node_embeddings)
        predictive_embedding /= torch.tensor(node_counts, device=device).unsqueeze(-1)

        edge_embeddings = incidence.edge_mean(node_embeddings)
        edge_attention_sum = node_embeddings.new_zeros(incidence.num_edges)
        edge_attention_sum.index_add_(0, incidence.edge_ids, incidence_attention)
        edge_importance = edge_attention_sum / incidence.edge_sizes
        if self.uniform_rejection_readout:
            edge_attention = grouped_softmax(
                torch.zeros_like(edge_importance), edge_graph_ids, len(batch)
            )
        else:
            edge_attention = grouped_softmax(edge_importance, edge_graph_ids, len(batch))

        rejection_embedding = edge_embeddings.new_zeros((len(batch), edge_embeddings.size(-1)))
        rejection_embedding.index_add_(
            0,
            edge_graph_ids,
            edge_attention.unsqueeze(-1) * edge_embeddings,
        )
        return {
            "logits": self.predictive_head(predictive_embedding),
            "reliability_logits": self.rejection_head(rejection_embedding).squeeze(-1),
            "predictive_embedding": predictive_embedding,
            "rejection_embedding": rejection_embedding,
            "node_embeddings": node_embeddings,
            "incidence_attention": incidence_attention,
            "edge_attention": edge_attention,
            "edge_graph_ids": edge_graph_ids,
        }

