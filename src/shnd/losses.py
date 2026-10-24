from __future__ import annotations
import torch
import torch.nn.functional as F


def contrastive_loss(embeddings, positive_mask, negative_mask, anchor_weights,
                     temperature, top_k=3):
    if temperature <= 0 or top_k < 0:
        raise ValueError("temperature must be positive; top_k=0 means all candidates")
    normalized = F.normalize(embeddings, dim=-1)
    similarity = normalized @ normalized.T / temperature
    total = embeddings.sum() * 0
    for i in range(len(embeddings)):
        positive = positive_mask[i].nonzero(as_tuple=True)[0]
        negative = negative_mask[i].nonzero(as_tuple=True)[0]
        if positive.numel() == 0:
            continue  
        if top_k:
            if positive.numel() > top_k:
                positive = positive[torch.randperm(len(positive), device=positive.device)[:top_k]]
            if negative.numel() > top_k:
                negative = negative[torch.randperm(len(negative), device=negative.device)[:top_k]]
        candidates = torch.cat((positive, negative))
        loss = torch.logsumexp(similarity[i, candidates], dim=0)-similarity[i, positive].mean()
        if anchor_weights is not None:
            loss = loss * anchor_weights[i]
        total = total + loss
    return total / max(1, len(embeddings))


def contrastive_masks(labels, predictions, reliability, gamma):
    other = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same = labels[:, None].eq(labels[None, :])
    correct = predictions.eq(labels)
    pf = same & correct[None, :] & other
    nf = predictions[None, :].eq(labels[:, None]) & ~same & other
    close = (reliability[:, None]-reliability[None, :]).abs() < gamma
    pg = same & close & other
    ng = (~same | ~close) & other
    return pf, nf, pg, ng


def dual_contrastive_losses(outputs, labels, tau_f=0.2, tau_g=0.2, gamma=0.1,
                            top_k=3, detach_confidence=False):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    probabilities = outputs["logits"].softmax(-1)
    reliability = outputs["reliability_logits"].sigmoid()
    pf, nf, pg, ng = contrastive_masks(labels, probabilities.detach().argmax(-1),
                                      reliability.detach(), gamma)
    weights = probabilities.max(-1).values * reliability
    if detach_confidence:
        weights = weights.detach()
    return (contrastive_loss(outputs["predictive_embedding"], pf, nf, weights, tau_f, top_k),
            contrastive_loss(outputs["rejection_embedding"], pg, ng, None, tau_g, top_k))
