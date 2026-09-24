from __future__ import annotations

import torch
import torch.nn.functional as F


def _topk_contrastive_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    anchor_weights: torch.Tensor | None,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=-1)
    similarities = normalized @ normalized.T
    losses = []
    for anchor in range(normalized.size(0)):
        positive = torch.where(positive_mask[anchor])[0]
        negative = torch.where(negative_mask[anchor])[0]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        positive_scores = similarities[anchor, positive]
        negative_scores = similarities[anchor, negative]
        if positive.numel() > top_k:
            positive = positive[torch.topk(positive_scores, top_k, largest=False).indices]
        if negative.numel() > top_k:
            negative = negative[torch.topk(negative_scores, top_k, largest=True).indices]
        candidates = torch.cat([positive, negative])
        denominator = torch.logsumexp(similarities[anchor, candidates] / temperature, dim=0)
        loss = -(similarities[anchor, positive] / temperature - denominator).mean()
        if anchor_weights is not None:
            loss = loss * anchor_weights[anchor].detach()
        losses.append(loss)
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


def dual_contrastive_losses(outputs, labels, tau_f: float, tau_g: float, gamma: float, top_k: int):
    probabilities = torch.softmax(outputs["logits"].detach(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    reliability = torch.sigmoid(outputs["reliability_logits"].detach())
    correct = predictions.eq(labels)
    not_self = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)

    cscl_positive = labels[:, None].eq(labels[None, :]) & correct[None, :] & not_self
    cscl_negative = (
        predictions[None, :].eq(labels[:, None])
        & labels[None, :].ne(labels[:, None])
        & not_self
    )
    cscl_weights = probabilities.max(dim=-1).values * reliability
    cscl = _topk_contrastive_loss(
        outputs["predictive_embedding"],
        cscl_positive,
        cscl_negative,
        cscl_weights,
        tau_f,
        top_k,
    )

    reliability_gap = (reliability[:, None] - reliability[None, :]).abs()
    rcl_positive = labels[:, None].eq(labels[None, :]) & (reliability_gap < gamma) & not_self
    rcl_negative = (labels[:, None].ne(labels[None, :]) | (reliability_gap >= gamma)) & not_self
    rcl = _topk_contrastive_loss(
        outputs["rejection_embedding"],
        rcl_positive,
        rcl_negative,
        None,
        tau_g,
        top_k,
    )
    return cscl, rcl

