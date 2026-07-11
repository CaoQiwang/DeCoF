"""Convert the standard PEMS-BAY HDF5 file to the STDN input format.

The STDN ``prepareData.py`` script expects ``data_file`` from the config to
contain an array named ``data`` with shape ``(T, N, 2)``.  Feature 0 is the
traffic value and feature 1 is a Unix timestamp repeated for every node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert(
    input_path: str | Path,
    output_path: str | Path,
    expected_nodes: int | None = None,
    key: str = "speed",
) -> None:
    """Convert an HDF5 traffic table into an STDN-compatible NPZ."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame = pd.read_hdf(input_path, key=key)
    if frame.ndim != 2 or (expected_nodes is not None and frame.shape[1] != expected_nodes):
        raise ValueError(
            f"expected {key} with {expected_nodes} sensors, got {frame.shape}"
        )
    expected_nodes = frame.shape[1]
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("PEMS-BAY contains duplicate timestamps")
    if frame.isna().any().any():
        raise ValueError("PEMS-BAY contains missing speed values; impute them first")

    values = frame.to_numpy(dtype=np.float32)
    timestamps = frame.index.astype("int64").to_numpy(dtype=np.float64) / 1e9
    # STDN reads data[..., :1] as the signal and data[:, 0, -1] as Unix time.
    data = np.empty((len(frame), expected_nodes, 2), dtype=np.float64)
    data[..., 0] = values
    data[..., 1] = timestamps[:, None]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, data=data)
    print(f"saved {output_path}: data={data.shape}, range={frame.index[0]}..{frame.index[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/PEMS-BAY/pems-bay.h5")
    parser.add_argument("--output", default="data/PEMS-BAY/PEMS-BAY.npz")
    parser.add_argument("--key", default="speed", help="HDF5 table key")
    parser.add_argument("--num-of-vertices", type=int, default=None)
    args = parser.parse_args()
    convert(args.input, args.output, args.num_of_vertices, args.key)


if __name__ == "__main__":
    main()
