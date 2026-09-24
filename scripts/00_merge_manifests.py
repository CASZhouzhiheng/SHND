from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge SHND manifests and rebase graph paths.")
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged = []
    for manifest in args.manifests:
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                graph_path = Path(row["pt_path"])
                if not graph_path.is_absolute():
                    graph_path = (manifest.parent / graph_path).resolve()
                row["pt_path"] = os.path.relpath(graph_path, args.output.parent.resolve())
                merged.append(row)
    if not merged:
        raise RuntimeError("no manifest rows were found")

    fieldnames = []
    for row in merged:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"merged {len(merged)} subjects into {args.output}")


if __name__ == "__main__":
    main()
