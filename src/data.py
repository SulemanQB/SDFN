"""Data loaders for ETT, METR-LA / PEMS-BAY, and custom CSVs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ────────────────────────────────────────────────────────────────────────────
#  Standard ETT split sizes  (Informer / PatchTST / iTransformer convention)
# ────────────────────────────────────────────────────────────────────────────

ETT_SPLITS: dict[str, tuple[int, int, int]] = {
    "ETTh1": (8_640, 2_880, 2_880),      # 12 / 4 / 4 months  (hourly)
    "ETTh2": (8_640, 2_880, 2_880),
    "ETTm1": (34_560, 11_520, 11_520),    # same months, 15-min
    "ETTm2": (34_560, 11_520, 11_520),
}


# ────────────────────────────────────────────────────────────────────────────
#  Dataset class
# ────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset over a contiguous segment of normalised data.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(N, D)`` — already normalised by the caller.
    context_len : int
        Lookback window length.
    pred_len : int
        Forecast horizon.
    stride : int
        Step between consecutive windows.
    """

    def __init__(
        self,
        data: np.ndarray,
        context_len: int,
        pred_len: int,
        stride: int = 1,
    ):
        self.data = data.astype(np.float32)
        self.context_len = context_len
        self.pred_len = pred_len
        self.stride = stride
        self.n_samples = max(
            0, (len(data) - context_len - pred_len) // stride + 1,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        s = idx * self.stride
        x = self.data[s : s + self.context_len]
        y = self.data[s + self.context_len : s + self.context_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


# ────────────────────────────────────────────────────────────────────────────
#  ETT loader
# ────────────────────────────────────────────────────────────────────────────

def load_ett(
    dataset_name: str,
    data_dir: str | Path,
    context_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Load an ETT dataset with the standard chronological split.

    Returns
    -------
    train_loader, val_loader, test_loader, n_vars, None
        (the last element is ``static_adj=None``)
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["date"], errors="ignore")
    values = df.select_dtypes(include="number").values.astype(np.float32)

    n_train, n_val, n_test = ETT_SPLITS[dataset_name]
    total_needed = n_train + n_val + n_test
    if len(values) < total_needed:
        raise ValueError(
            f"{dataset_name} has {len(values)} rows but split needs {total_needed}"
        )

    # ── chronological split ─────────────────────────────────────────────
    # Validation/test can look back into the preceding segment by
    # `context_len` so the first prediction target is right at the boundary.
    raw_train = values[:n_train]
    raw_val   = values[n_train - context_len : n_train + n_val]
    raw_test  = values[n_train + n_val - context_len : n_train + n_val + n_test]

    # ── normalise with train statistics ─────────────────────────────────
    mean = raw_train.mean(axis=0)
    std  = raw_train.std(axis=0) + 1e-8
    train_data = (raw_train - mean) / std
    val_data   = (raw_val   - mean) / std
    test_data  = (raw_test  - mean) / std

    n_vars = values.shape[1]
    return (
        _make_loader(train_data, context_len, pred_len, batch_size, num_workers, shuffle=True),
        _make_loader(val_data,   context_len, pred_len, batch_size, num_workers, shuffle=False),
        _make_loader(test_data,  context_len, pred_len, batch_size, num_workers, shuffle=False),
        n_vars,
        None,  # no static adjacency
    )


# ────────────────────────────────────────────────────────────────────────────
#  Traffic loader  (METR-LA / PEMS-BAY)
# ────────────────────────────────────────────────────────────────────────────

def load_traffic(
    data_path: str | Path,
    adj_path: str | Path | None = None,
    context_len: int = 12,
    pred_len: int = 12,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    treat_zeros_as_missing: bool = False,
):
    """Load a traffic dataset (METR-LA, PEMS-BAY, or similar).

    Expects an HDF5 file readable by ``pandas.read_hdf`` *or* a CSV.
    Optionally loads a pre-computed adjacency from a pickle or numpy file.

    Returns
    -------
    train_loader, val_loader, test_loader, n_vars, static_adj
    """
    data_path = Path(data_path)

    # ── load time-series values ─────────────────────────────────────────
    if data_path.suffix in (".h5", ".hdf5"):
        values = _load_h5(data_path)
    elif data_path.suffix == ".npz":
        values = np.load(data_path)["data"].astype(np.float32)
        if values.ndim == 3:                     # (N, D, C) → take first channel
            values = values[..., 0]
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        df = df.drop(columns=["date", "Date"], errors="ignore")
        values = df.select_dtypes(include="number").values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")

    # Handle missing values with forward-fill -> back-fill.
    # By default, zeros are treated as valid observations.
    mask = np.isnan(values)
    if treat_zeros_as_missing:
        mask = mask | (values == 0)
    if mask.any():
        df_fill = pd.DataFrame(values)
        if treat_zeros_as_missing:
            df_fill = df_fill.replace(0, np.nan)
        df_fill = df_fill.ffill().bfill().fillna(0)
        values = df_fill.values.astype(np.float32)

    n_total = len(values)
    n_vars  = values.shape[1]

    # ── chronological 70/10/20 split ────────────────────────────────────
    train_end = int(n_total * train_ratio)
    val_end   = int(n_total * (train_ratio + val_ratio))

    raw_train = values[:train_end]
    raw_val   = values[train_end - context_len : val_end]
    raw_test  = values[val_end   - context_len :]

    # ── normalise with train statistics ─────────────────────────────────
    mean = raw_train.mean(axis=0)
    std  = raw_train.std(axis=0) + 1e-8
    train_data = (raw_train - mean) / std
    val_data   = (raw_val   - mean) / std
    test_data  = (raw_test  - mean) / std

    # ── adjacency (optional) ────────────────────────────────────────────
    static_adj = None
    if adj_path is not None:
        static_adj = load_adjacency(adj_path, n_vars)

    return (
        _make_loader(train_data, context_len, pred_len, batch_size, num_workers, shuffle=True),
        _make_loader(val_data,   context_len, pred_len, batch_size, num_workers, shuffle=False),
        _make_loader(test_data,  context_len, pred_len, batch_size, num_workers, shuffle=False),
        n_vars,
        static_adj,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Generic CSV loader
# ────────────────────────────────────────────────────────────────────────────

def load_custom_csv(
    csv_path: str | Path,
    context_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """Load an arbitrary CSV with numeric columns using a 70/10/20 split."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["date", "Date", "timestamp"], errors="ignore")
    values = df.select_dtypes(include="number").values.astype(np.float32)

    n_total = len(values)
    n_vars  = values.shape[1]
    train_end = int(n_total * train_ratio)
    val_end   = int(n_total * (train_ratio + val_ratio))

    raw_train = values[:train_end]
    raw_val   = values[train_end - context_len : val_end]
    raw_test  = values[val_end   - context_len :]

    mean = raw_train.mean(axis=0)
    std  = raw_train.std(axis=0) + 1e-8
    train_data = (raw_train - mean) / std
    val_data   = (raw_val   - mean) / std
    test_data  = (raw_test  - mean) / std

    return (
        _make_loader(train_data, context_len, pred_len, batch_size, num_workers, shuffle=True),
        _make_loader(val_data,   context_len, pred_len, batch_size, num_workers, shuffle=False),
        _make_loader(test_data,  context_len, pred_len, batch_size, num_workers, shuffle=False),
        n_vars,
        None,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Adjacency loader
# ────────────────────────────────────────────────────────────────────────────

def load_adjacency(
    path: str | Path,
    expected_n_vars: int | None = None,
) -> torch.Tensor:
    """Load a pre-computed adjacency matrix from pickle, numpy, or CSV.

    The DCRNN format stores ``[sensor_ids, id_to_ind, adj_mx]`` in a pickle.
    """
    path = Path(path)

    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if isinstance(data, (list, tuple)):
            adj = np.asarray(data[-1], dtype=np.float32)   # last element
        elif isinstance(data, np.ndarray):
            adj = data.astype(np.float32)
        elif isinstance(data, dict):
            for key in ("adj", "adj_mx", "A"):
                if key in data:
                    adj = np.asarray(data[key], dtype=np.float32)
                    break
            else:
                raise KeyError(f"Cannot find adjacency key in {list(data.keys())}")
        else:
            raise TypeError(f"Unexpected pickle type: {type(data)}")
    elif path.suffix in (".npy", ".npz"):
        loaded = np.load(path, allow_pickle=True)
        adj = loaded if isinstance(loaded, np.ndarray) else loaded["adj"]
        adj = adj.astype(np.float32)
    elif path.suffix == ".csv":
        adj = pd.read_csv(path, header=None).values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported adjacency format: {path.suffix}")

    if expected_n_vars is not None and adj.shape[0] != expected_n_vars:
        raise ValueError(
            f"Adjacency shape {adj.shape} doesn't match n_vars={expected_n_vars}"
        )
    return torch.from_numpy(adj)


# ────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_h5(path: Path) -> np.ndarray:
    """Load time-series from an HDF5 file (pandas or raw h5py)."""
    try:
        df = pd.read_hdf(path)
        return df.values.astype(np.float32)
    except Exception:
        import h5py
        with h5py.File(path, "r") as f:
            # Try common key names
            for key in ("df/block0_values", "data", "speed", "values"):
                if key in f:
                    return np.asarray(f[key], dtype=np.float32)
            # fallback: first dataset
            def _first_dataset(g):
                for v in g.values():
                    if isinstance(v, h5py.Dataset):
                        return np.asarray(v, dtype=np.float32)
                    ds = _first_dataset(v)
                    if ds is not None:
                        return ds
                return None
            ds = _first_dataset(f)
            if ds is not None:
                return ds
        raise ValueError(f"Could not read data from {path}")


def _make_loader(
    data: np.ndarray,
    context_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    ds = TimeSeriesDataset(data, context_len, pred_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,       # drop last incomplete batch only for training
    )


# ────────────────────────────────────────────────────────────────────────────
#  Dispatcher  (used by train.py)
# ────────────────────────────────────────────────────────────────────────────

def create_dataloaders(args):
    """Create train/val/test loaders from CLI arguments.

    ``args`` must have at least ``.dataset``, ``.data_dir`` (or
    ``.data_path``), ``.context_len``, ``.pred_len``, ``.batch_size``.
    """
    if args.dataset.startswith("ETT"):
        return load_ett(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            context_len=args.context_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=getattr(args, "num_workers", 4),
        )

    if args.dataset in ("metr-la", "pems-bay"):
        return load_traffic(
            data_path=args.data_path,
            adj_path=getattr(args, "adj_path", None),
            context_len=args.context_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=getattr(args, "num_workers", 4),
            treat_zeros_as_missing=getattr(args, "treat_zeros_as_missing", False),
        )

    # fallback: treat as generic CSV
    return load_custom_csv(
        csv_path=args.data_path,
        context_len=args.context_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 4),
    )
