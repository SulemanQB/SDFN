#!/usr/bin/env python3
"""Train SDFN on ETT or traffic datasets. See README for examples."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import create_dataloaders
from src.model import SDFN
from src.metrics import (
    compute_point_metrics,
    crps_empirical,
    coverage,
    interval_width,
)


# ────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────────────────
#  Training
# ────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: SDFN,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch.  Returns dict of averaged losses."""
    model.train()
    sums: dict[str, float] = {}
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        losses = model(x, y)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


# ────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: SDFN,
    loader,
    device: torch.device,
    n_samples: int = 1,
    sample_steps: int = 50,
    use_ddim: bool = True,
) -> dict[str, float]:
    """Evaluate the model on a data loader.

    When ``n_samples > 1``, also computes CRPS and coverage.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y_hat = model.predict(
            x,
            n_samples=n_samples,
            sample_steps=sample_steps,
            use_ddim=use_ddim,
        )
        # y_hat: (B,H,D) if n_samples==1 else (n_samples,B,H,D)
        all_preds.append(y_hat.cpu())
        all_targets.append(y)

    # concatenate along the batch dimension
    if n_samples == 1:
        preds   = torch.cat(all_preds, dim=0).numpy()   # (N,H,D)
        targets = torch.cat(all_targets, dim=0).numpy()
        return compute_point_metrics(targets, preds)
    else:
        preds   = torch.cat(all_preds, dim=1).numpy()   # (S,N,H,D)
        targets = torch.cat(all_targets, dim=0).numpy()  # (N,H,D)
        point = preds.mean(axis=0)                        # (N,H,D)
        metrics = compute_point_metrics(targets, point)
        metrics["crps"]     = crps_empirical(targets, preds)
        metrics["cov_90"]   = coverage(targets, preds, level=0.90)
        metrics["cov_50"]   = coverage(targets, preds, level=0.50)
        metrics["width_90"] = interval_width(preds, level=0.90)
        return metrics


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # ── data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, n_vars, static_adj = \
        create_dataloaders(args)

    print(f"Dataset: {args.dataset}  |  n_vars={n_vars}  |  "
          f"context={args.context_len}  pred={args.pred_len}")
    print(f"Train batches: {len(train_loader)}  |  "
          f"Val: {len(val_loader)}  |  Test: {len(test_loader)}")

    # ── model ───────────────────────────────────────────────────────────
    model = SDFN(
        n_vars=n_vars,
        horizon=args.pred_len,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_diff_steps=args.n_diff_steps,
        n_gnn_layers=args.n_gnn_layers,
        top_k=args.top_k,
        lambda_noise=args.lambda_noise,
        lambda_recon=args.lambda_recon,
        lambda_sparse=args.lambda_sparse,
        lambda_smooth=args.lambda_smooth,
        max_rank=args.max_rank,
        static_adj=static_adj,
        beta_schedule=args.beta_schedule,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  ({n_params / 1e6:.2f} M)")

    # ── optimizer & scheduler ───────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── training loop ───────────────────────────────────────────────────
    best_val_mae = float("inf")
    patience_ctr = 0
    save_path = Path(args.save_dir) / f"sdfn_{args.dataset}_H{args.pred_len}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []

    print("\n" + "=" * 80)
    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'Diff':>8}  {'Recon':>8}  "
          f"{'Sparse':>8}  {'Smooth':>8}  │  {'ValMAE':>8}  {'ValRMSE':>8}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = train_epoch(model, train_loader, optimizer, device,
                                   grad_clip=args.grad_clip)
        scheduler.step()

        # ── validation (fast: 1 sample, DDIM) ──────────────────────────
        val_metrics = evaluate(
            model, val_loader, device,
            n_samples=1, sample_steps=args.eval_steps, use_ddim=True,
        )

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time": dt,
        }
        history.append(row)

        print(
            f"{epoch:5d}  {train_losses['total']:8.4f}  "
            f"{train_losses['diffusion']:8.4f}  "
            f"{train_losses['reconstruction']:8.4f}  "
            f"{train_losses['sparse']:8.4f}  "
            f"{train_losses['smooth']:8.4f}  │  "
            f"{val_metrics['mae']:8.4f}  {val_metrics['rmse']:8.4f}  "
            f"({dt:.1f}s)"
        )

        # ── early stopping ──────────────────────────────────────────────
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val MAE = {best_val_mae:.4f})")
                break

    # ── test ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Loading best model for testing …")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Point metrics  (1 deterministic sample)
    test_point = evaluate(
        model, test_loader, device,
        n_samples=1, sample_steps=args.eval_steps, use_ddim=True,
    )
    print(f"\n[Test — point prediction]")
    for k, v in test_point.items():
        print(f"  {k:>8s}: {v:.4f}")

    # Probabilistic metrics  (multiple samples)
    if args.test_samples > 1:
        test_prob = evaluate(
            model, test_loader, device,
            n_samples=args.test_samples,
            sample_steps=args.eval_steps,
            use_ddim=True,
        )
        print(f"\n[Test — probabilistic ({args.test_samples} samples)]")
        for k, v in test_prob.items():
            print(f"  {k:>8s}: {v:.4f}")
    else:
        test_prob = {}

    # ── save results ────────────────────────────────────────────────────
    results = {
        "args": vars(args),
        "n_params": n_params,
        "best_val_mae": best_val_mae,
        "test_point": test_point,
        "test_prob": test_prob,
        "history": history,
    }
    results_path = save_path.with_suffix(".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Checkpoint saved to {save_path}")
    print("=" * 80)


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SDFN on time-series forecasting benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ────────────────────────────────────────────────────────────
    p.add_argument("--dataset", type=str, default="ETTh1",
                   help="Dataset name: ETTh1/h2/m1/m2, metr-la, pems-bay, custom")
    p.add_argument("--data_dir", type=str,
                   default=str(Path(__file__).resolve().parent / "data"),
                   help="Directory containing ETT CSVs")
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to .h5 / .csv for traffic / custom datasets")
    p.add_argument("--adj_path", type=str, default=None,
                   help="Path to adjacency pickle/numpy (traffic datasets)")
    p.add_argument("--treat_zeros_as_missing", action="store_true",
                   help="For traffic datasets, treat zeros as missing values"
                        " before forward/backward fill")
    p.add_argument("--context_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    # ── model ───────────────────────────────────────────────────────────
    p.add_argument("--d_model", type=int, default=128,
                   help="Hidden dim for encoders & GNN")
    p.add_argument("--d_latent", type=int, default=64,
                   help="Per-node latent dim for diffusion")
    p.add_argument("--n_diff_steps", type=int, default=200,
                   help="Number of forward-diffusion timesteps")
    p.add_argument("--n_gnn_layers", type=int, default=4,
                   help="GNN denoiser depth")
    p.add_argument("--top_k", type=int, default=20,
                   help="Top-k adjacency sparsification per node")
    p.add_argument("--beta_schedule", type=str, default="cosine",
                   choices=["linear", "cosine"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_rank", type=int, default=None,
                   help="Low-rank approximation rank for Laplacian noise "
                        "(None = full rank)")

    # ── loss weights ────────────────────────────────────────────────────
    p.add_argument("--lambda_noise", type=float, default=0.1,
                   help="Laplacian regularisation (added to eigenvalues)")
    p.add_argument("--lambda_recon", type=float, default=1.0,
                   help="Reconstruction loss weight")
    p.add_argument("--lambda_sparse", type=float, default=0.01,
                   help="Graph sparsity (L1) weight")
    p.add_argument("--lambda_smooth", type=float, default=0.1,
                   help="Graph smoothness weight")

    # ── training ────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU training")

    # ── evaluation ──────────────────────────────────────────────────────
    p.add_argument("--eval_steps", type=int, default=50,
                   help="DDIM reverse steps for inference")
    p.add_argument("--test_samples", type=int, default=20,
                   help="Number of samples for probabilistic test metrics")

    # ── output ──────────────────────────────────────────────────────────
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="Directory for saving checkpoints & results")

    args = p.parse_args()

    # point data_dir at the shared data folder when using ETT
    if args.dataset.startswith("ETT") and args.data_dir.endswith("/data"):
        parent = Path(__file__).resolve().parent.parent / "data"
        if parent.exists():
            args.data_dir = str(parent)

    return args


if __name__ == "__main__":
    main()
