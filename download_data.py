#!/usr/bin/env python3
"""Download METR-LA / PEMS-BAY traffic data. Example: python download_data.py --dataset metr-la"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
#  Known download URLs  (LibCity / DCRNN mirrors)
# ────────────────────────────────────────────────────────────────────────────

METR_LA_URLS = {
    "metr-la.h5": [
        # Google Drive (via gdown)
        "https://drive.google.com/uc?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC",
    ],
    "adj_mx.pkl": [
        "https://drive.google.com/uc?id=1wEq_dK0kf6PBZhABtrNRMiFArqMGjMEu",
    ],
}

PEMS_BAY_URLS = {
    "pems-bay.h5": [
        "https://drive.google.com/uc?id=1wD-BVo5Tn_sJNh1jOJqpf8cMrz7lEO5z",
    ],
    "adj_mx_bay.pkl": [
        "https://drive.google.com/uc?id=1HbRnRPaJnqJIkR-ePxCPOhQMFsfnJmKo",
    ],
}

DATASET_INFO = {
    "metr-la": {
        "files": METR_LA_URLS,
        "description": "METR-LA: 207 traffic sensors in Los Angeles, "
                       "4 months of 5-min aggregated speed data (34,272 timesteps).",
        "adj_file": "adj_mx.pkl",
    },
    "pems-bay": {
        "files": PEMS_BAY_URLS,
        "description": "PEMS-BAY: 325 traffic sensors in San Francisco Bay Area, "
                       "6 months of 5-min aggregated speed data (52,116 timesteps).",
        "adj_file": "adj_mx_bay.pkl",
    },
}


# ────────────────────────────────────────────────────────────────────────────
#  Download helpers
# ────────────────────────────────────────────────────────────────────────────

def download_gdrive(file_id_or_url: str, dest: Path) -> bool:
    """Download from Google Drive using gdown (if available) or direct URL."""
    try:
        import gdown
        # gdown handles large files & confirmation pages
        if file_id_or_url.startswith("https://"):
            gdown.download(file_id_or_url, str(dest), quiet=False)
        else:
            gdown.download(id=file_id_or_url, output=str(dest), quiet=False)
        return dest.exists() and dest.stat().st_size > 0
    except ImportError:
        print("  [!] gdown not installed. Install with: pip install gdown")
        print(f"      Or download manually from {file_id_or_url}")
        return False
    except Exception as e:
        print(f"  [!] gdown failed: {e}")
        return False


def download_file(urls: list[str], dest: Path) -> bool:
    """Try each URL in order until one succeeds."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [✓] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    for url in urls:
        if "drive.google.com" in url:
            if download_gdrive(url, dest):
                print(f"  [✓] Downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
                return True
        else:
            try:
                print(f"  Downloading {dest.name} …")
                urllib.request.urlretrieve(url, dest)
                if dest.exists() and dest.stat().st_size > 1000:
                    print(f"  [✓] Downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
                    return True
            except Exception as e:
                print(f"  [!] Failed: {e}")

    return False


def download_dataset(name: str, output_dir: Path) -> None:
    """Download all files for a given dataset."""
    info = DATASET_INFO[name]
    print(f"\n{'=' * 60}")
    print(f"  {info['description']}")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for filename, urls in info["files"].items():
        dest = output_dir / filename
        ok = download_file(urls, dest)
        if not ok:
            all_ok = False
            print(f"\n  [✗] Could not download {filename}.")
            print(f"      Manual download options:")
            print(f"      1. Install gdown:  pip install gdown")
            print(f"      2. Download from Google Drive and place at: {dest}")
            for url in urls:
                print(f"         URL: {url}")

    if all_ok:
        print(f"\n  [✓] All files for {name} are ready in {output_dir}/")
    else:
        print(f"\n  [!] Some files are missing. See instructions above.")


def verify_dataset(name: str, output_dir: Path) -> None:
    """Quick verification that downloaded files are loadable."""
    info = DATASET_INFO[name]
    print(f"\nVerifying {name} …")

    for filename in info["files"]:
        fpath = output_dir / filename
        if not fpath.exists():
            print(f"  [✗] Missing: {filename}")
            continue

        size_mb = fpath.stat().st_size / 1e6

        if filename.endswith(".h5"):
            try:
                import pandas as pd
                df = pd.read_hdf(fpath)
                print(f"  [✓] {filename}: {df.shape[0]} timesteps × "
                      f"{df.shape[1]} sensors ({size_mb:.1f} MB)")
            except Exception:
                try:
                    import h5py
                    with h5py.File(fpath, "r") as f:
                        keys = list(f.keys())
                        print(f"  [✓] {filename}: h5 keys={keys} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"  [?] {filename}: {size_mb:.1f} MB (could not verify: {e})")

        elif filename.endswith(".pkl"):
            try:
                import pickle
                with open(fpath, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                if isinstance(data, (list, tuple)):
                    adj = data[-1]
                    print(f"  [✓] {filename}: adjacency shape {adj.shape} ({size_mb:.1f} MB)")
                else:
                    print(f"  [✓] {filename}: type={type(data).__name__} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  [?] {filename}: {size_mb:.1f} MB (could not verify: {e})")


# ────────────────────────────────────────────────────────────────────────────
#  Build adjacency from sensor distances  (fallback)
# ────────────────────────────────────────────────────────────────────────────

def build_adjacency_from_distances(
    dist_csv: str | Path,
    sensor_ids: list | None = None,
    sigma2: float = 0.1,
    epsilon: float = 0.5,
) -> "np.ndarray":
    """Build a Gaussian-kernel adjacency from a pairwise-distance CSV.

    The CSV should have columns: ``from, to, cost`` (or ``distance``).
    Uses the standard DCRNN formula:

    .. math::

        A_{ij} = \\exp(-d_{ij}^2 / \\sigma^2) \\;\\text{if}\\;
                 \\exp(-d_{ij}^2 / \\sigma^2) \\ge \\epsilon

    Returns
    -------
    np.ndarray of shape (N, N)
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(dist_csv)
    col_dist = [c for c in df.columns if c.lower() in ("cost", "distance", "dist")][0]
    col_from = df.columns[0]
    col_to   = df.columns[1]

    # identify unique sensors
    if sensor_ids is None:
        sensor_ids = sorted(set(df[col_from].tolist() + df[col_to].tolist()))
    n = len(sensor_ids)
    id_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}

    dist_mx = np.zeros((n, n), dtype=np.float32)
    for _, row in df.iterrows():
        i = id_to_idx.get(row[col_from])
        j = id_to_idx.get(row[col_to])
        if i is not None and j is not None:
            d = float(row[col_dist])
            dist_mx[i, j] = d
            dist_mx[j, i] = d

    # Gaussian kernel
    var = dist_mx[dist_mx > 0].var() if sigma2 <= 0 else sigma2
    adj = np.exp(-dist_mx ** 2 / var)
    adj[adj < epsilon] = 0.0
    return adj


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download traffic datasets for SDFN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["metr-la", "pems-bay", "all"],
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify downloaded files after download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    datasets = (
        list(DATASET_INFO.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    for name in datasets:
        download_dataset(name, output_dir)
        if args.verify:
            verify_dataset(name, output_dir)

    print("\n" + "=" * 60)
    print("Done!  Example usage:")
    print()
    print("  # METR-LA (207 sensors, 5-min, 1h→1h forecast)")
    print("  python train.py --dataset metr-la \\")
    print(f"      --data_path {output_dir}/metr-la.h5 \\")
    print(f"      --adj_path  {output_dir}/adj_mx.pkl \\")
    print("      --context_len 12 --pred_len 12")
    print()
    print("  # PEMS-BAY (325 sensors)")
    print("  python train.py --dataset pems-bay \\")
    print(f"      --data_path {output_dir}/pems-bay.h5 \\")
    print(f"      --adj_path  {output_dir}/adj_mx_bay.pkl \\")
    print("      --context_len 12 --pred_len 12")
    print("=" * 60)


if __name__ == "__main__":
    main()
