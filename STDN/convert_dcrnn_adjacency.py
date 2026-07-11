"""Convert a DCRNN ``adj_mx_*.pkl`` graph into STDN's edge-list CSV format."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def load_graph(path: Path):
    with path.open("rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


def convert(input_path: Path, output_path: Path) -> None:
    sensor_ids, _, adjacency = load_graph(input_path)
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"expected a square adjacency matrix, got {adjacency.shape}")
    if len(sensor_ids) != adjacency.shape[0]:
        raise ValueError("sensor IDs and adjacency matrix have inconsistent sizes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.argwhere((adjacency != 0) & ~np.eye(adjacency.shape[0], dtype=bool))
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["from", "to", "distance"])
        for source, target in rows:
            # STDN only tests whether the third field is present; use the
            # original DCRNN edge weight so no graph information is lost.
            writer.writerow([source, target, float(adjacency[source, target])])

    ids_path = output_path.with_suffix(".sensor_ids.txt")
    ids_path.write_text("\n".join(map(str, sensor_ids)) + "\n")
    print(
        f"saved {output_path}: {len(rows)} directed edges; "
        f"sensor IDs saved to {ids_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/PEMS-BAY/adj_mx_bay.pkl")
    parser.add_argument("--output", default="data/PEMS-BAY/PEMS-BAY.csv")
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
