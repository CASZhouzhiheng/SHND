from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shnd.data import HypergraphDataset, collate_graphs, read_manifest
from shnd.model import SHNDModel 


def main():
    parser = argparse.ArgumentParser(description="Export subject-level SHND hyperedge attention.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shnd-paper-model-v1":
        raise ValueError("Legacy checkpoint: retrain the paper-aligned model")
    if checkpoint["method"] == "sr":
        raise ValueError("SR has no rejection head to export")
    model = SHNDModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device).eval()
    feature = checkpoint["training_config"]["feature"]
    rows = read_manifest(args.manifest, {args.dataset} if args.dataset else None)
    if args.split != "all":
        if any("split" not in row for row in rows):
            raise ValueError("Use the saved fold split.csv for out-of-fold export, or explicitly pass --split all")
        rows = [row for row in rows if row["split"] == args.split]
    loader = DataLoader(
        HypergraphDataset(rows, feature, checkpoint["method"] == "pairwise_only"), batch_size=1, collate_fn=collate_graphs
    )

    exported = []
    with torch.no_grad():
        for batch in loader:
            item = batch[0]
            outputs = model(batch)
            reliability = torch.sigmoid(outputs["reliability_logits"])[0].item()
            probabilities = torch.softmax(outputs["logits"], dim=-1)[0]
            attention = outputs["edge_attention"].cpu().tolist()
            for edge_id, (edge, beta) in enumerate(zip(item["edge_list"], attention)):
                exported.append(
                    {
                        "dataset": item["dataset"],
                        "subject_id": item["subject_id"],
                        "label": item["label"],
                        "edge_id": edge_id,
                        "roi_indices_0based": "-".join(map(str, edge)),
                        "roi_indices_1based": "-".join(str(node + 1) for node in edge),
                        "edge_size": len(edge),
                        "beta": beta,
                        "reliability": reliability,
                        "prediction": int(probabilities.argmax().item()),
                    }
                )
    if not exported:
        raise ValueError("No subject-hyperedge rows selected for export")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(f"saved {len(exported)} subject-hyperedge rows to {args.output}")


if __name__ == "__main__":
    main()

