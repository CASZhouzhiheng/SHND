from __future__ import annotations

import csv
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import HypergraphDataset, collate_graphs
from .losses import dual_contrastive_losses
from .metrics import risk_at_coverages, selection_score
from .model import SHNDModel


ABLATIONS = {
    "shnd",
    "sr",
    "wo_cscl",
    "wo_rcl",
    "wo_dual_cl",
    "wo_warmup",
    "pairwise_only",
    "uniform_readout",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_from_batch(batch, device):
    return torch.tensor([item["label"] for item in batch], dtype=torch.long, device=device)


def method_settings(method: str, config: dict):
    if method not in ABLATIONS:
        raise ValueError(f"unknown method: {method}")
    return {
        "lambda_f": 0.0 if method in {"wo_cscl", "wo_dual_cl"} else config["lambda_f"],
        "lambda_g": 0.0 if method in {"wo_rcl", "wo_dual_cl"} else config["lambda_g"],
        "warmup_epochs": 0 if method == "wo_warmup" else config["warmup_epochs"],
        "pairwise_only": method == "pairwise_only",
        "uniform_readout": method == "uniform_readout",
    }


@torch.no_grad()
def evaluate(model, loader, device, use_msp: bool = False):
    model.eval()
    rows = []
    for batch in loader:
        labels = labels_from_batch(batch, device)
        outputs = model(batch)
        probabilities = torch.softmax(outputs["logits"], dim=-1)
        predictions = probabilities.argmax(dim=-1)
        scores = probabilities.max(dim=-1).values if use_msp else torch.sigmoid(outputs["reliability_logits"])
        for item, label, prediction, score, probability in zip(
            batch, labels, predictions, scores, probabilities
        ):
            rows.append(
                {
                    "dataset": item["dataset"],
                    "subject_id": item["subject_id"],
                    "group": item["group"],
                    "label": int(label.item()),
                    "prediction": int(prediction.item()),
                    "score": float(score.item()),
                    **{
                        f"probability_{index}": float(value.item())
                        for index, value in enumerate(probability)
                    },
                }
            )
    return rows


def _write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_run(train_rows, val_rows, test_rows, config: dict, method: str, seed: int, output_dir: Path):
    set_seed(seed)
    device = torch.device(config["device"])
    settings = method_settings(method, config)
    model = SHNDModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        uniform_rejection_readout=settings["uniform_readout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    def loader(rows, shuffle=False):
        return DataLoader(
            HypergraphDataset(rows, config["feature"], settings["pairwise_only"]),
            batch_size=config["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_graphs,
        )

    train_loader = loader(train_rows, shuffle=True)
    val_loader = loader(val_rows)
    test_loader = loader(test_rows)
    best_state = None
    best_score = float("inf")
    history = []

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            labels = labels_from_batch(batch, device)
            outputs = model(batch)
            classification = F.cross_entropy(outputs["logits"], labels)
            if method == "sr":
                loss = classification
            else:
                with torch.no_grad():
                    reliability_target = outputs["logits"].argmax(dim=-1).eq(labels).float()
                rejection = F.binary_cross_entropy_with_logits(
                    outputs["reliability_logits"], reliability_target
                )
                loss = classification + config["lambda_c"] * rejection
                if epoch >= settings["warmup_epochs"]:
                    cscl, rcl = dual_contrastive_losses(
                        outputs,
                        labels,
                        config["tau_f"],
                        config["tau_g"],
                        config["gamma"],
                        config["contrastive_k"],
                    )
                    loss = loss + settings["lambda_f"] * cscl + settings["lambda_g"] * rcl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

        should_evaluate = epoch == 0 or (epoch + 1) % config["eval_every"] == 0
        if epoch + 1 == config["epochs"]:
            should_evaluate = True
        row = {"epoch": epoch + 1, "train_loss": running_loss / max(1, batches)}
        if should_evaluate:
            val_predictions = evaluate(model, val_loader, device, use_msp=method == "sr")
            val_risks = risk_at_coverages(val_predictions)
            score = selection_score(val_risks, config["selection_metric"])
            row.update({f"val_risk_{coverage}": risk for coverage, risk in val_risks.items()})
            row["selection_score"] = score
            if score < best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
        history.append(row)
        print(
            f"epoch={epoch + 1:03d} loss={row['train_loss']:.6f} "
            f"best_{config['selection_metric']}={best_score:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = evaluate(model, test_loader, device, use_msp=method == "sr")
    risks = risk_at_coverages(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "training_history.csv", history)
    _write_csv(output_dir / "predictions.csv", predictions)
    _write_csv(
        output_dir / "coverage_risk.csv",
        [{"coverage": coverage, "risk": risk} for coverage, risk in risks.items()],
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config,
            "training_config": config,
            "method": method,
            "seed": seed,
        },
        output_dir / "best_model.pt",
    )
    return risks

