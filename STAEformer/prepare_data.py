"""Build STAEformer ``data.npz`` and ``index.npz`` from HDF5 or STDN NPZ data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_indices(
    num_steps: int,
    in_steps: int,
    out_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``[start, input_end, target_end]`` windows for each split."""
    if in_steps <= 0 or out_steps <= 0:
        raise ValueError("in_steps and out_steps must be positive")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("invalid train/validation ratios")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    num_samples = num_steps - in_steps - out_steps + 1
    if num_samples <= 0:
        raise ValueError("time series is shorter than one input-target window")
    starts = np.arange(num_samples, dtype=np.int64)
    indices = np.column_stack(
        (starts, starts + in_steps, starts + in_steps + out_steps)
    )
    train_end = round(num_samples * train_ratio)
    val_end = train_end + round(num_samples * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def convert(
    input_path: Path,
    output_dir: Path,
    key: str,
    in_steps: int,
    out_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    if input_path.suffix.lower() == ".npz":
        source = np.load(input_path)
        source_data = source["data"]
        if source_data.ndim != 3 or source_data.shape[-1] < 2:
            raise ValueError(
                "STDN NPZ input must contain data with shape (time, nodes, "
                "at least 2 features): traffic value and Unix timestamp"
            )
        values = source_data[..., 0].astype(np.float32)
        timestamps = source_data[:, 0, 1]
        index = pd.to_datetime(timestamps, unit="s")
    else:
        frame = pd.read_hdf(input_path, key=key).sort_index()
        values = frame.to_numpy(dtype=np.float32)
        index = pd.DatetimeIndex(frame.index)

    if index.has_duplicates:
        raise ValueError("input data contains duplicate timestamps")
    if not np.isfinite(values).all():
        raise ValueError("input data contains missing traffic values")

    # STAEformer multiplies channel 1 by steps_per_day before looking up its
    # time-of-day embedding, so this channel must be in [0, 1), not 0..287.
    steps_per_day = 288
    time_of_day = (
        (index.hour * 60 + index.minute) // 5
    ).to_numpy(dtype=np.float32) / steps_per_day
    day_of_week = index.dayofweek.to_numpy(dtype=np.float32)
    data = np.empty((*values.shape, 3), dtype=np.float32)
    data[..., 0] = values
    data[..., 1] = time_of_day[:, None]
    data[..., 2] = day_of_week[:, None]

    train, val, test = build_indices(
        len(index), in_steps, out_steps, train_ratio, val_ratio
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "data.npz", data=data)
    np.savez_compressed(output_dir / "index.npz", train=train, val=val, test=test)
    print(
        f"saved {output_dir}: data={data.shape}, "
        f"train={train.shape}, val={val.shape}, test={test.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/METR-LA/METR-LA.npz")
    parser.add_argument("--key", default="df", help="HDF5 table key")
    parser.add_argument("--output-dir", default="STAEformer/data/METRLA")
    parser.add_argument("--in-steps", type=int, default=24)
    parser.add_argument("--out-steps", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    convert(
        Path(args.input),
        Path(args.output_dir),
        args.key,
        args.in_steps,
        args.out_steps,
        args.train_ratio,
        args.val_ratio,
    )


if __name__ == "__main__":
    main()
