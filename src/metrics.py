"""Point metrics (MSE/MAE/RMSE/MAPE) and probabilistic metrics (CRPS, coverage)."""

from __future__ import annotations

import math

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Point-prediction metrics
# ────────────────────────────────────────────────────────────────────────────

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """MAPE as a percentage (e.g. 2.5 means 2.5 %)."""
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_point_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all point-prediction metrics at once."""
    return {
        "mse":  mse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


# ────────────────────────────────────────────────────────────────────────────
#  Probabilistic metrics
# ────────────────────────────────────────────────────────────────────────────

def crps_empirical(
    y_true: np.ndarray,
    y_samples: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score from an ensemble of forecasts.

    Uses the energy form:

    .. math::

        \\text{CRPS} = \\frac{1}{n}\\sum_i |x_i - y|
                       - \\frac{1}{2n^2}\\sum_{i,j} |x_i - x_j|

    For efficiency the pairwise term is computed via the sorted-sample
    identity (O(n log n) per element instead of O(n²)).

    Parameters
    ----------
    y_true : array (...,)
        Ground truth values (flattened or multi-dim).
    y_samples : array (n_samples, ...)
        Forecast samples (first axis = sample index).

    Returns
    -------
    float
        Scalar CRPS averaged over all elements.
    """
    n = y_samples.shape[0]
    if n < 2:
        return float(np.mean(np.abs(y_samples[0] - y_true)))

    # Flatten spatial dims for vectorised computation
    y_flat = y_true.reshape(-1)                        # (M,)
    s_flat = y_samples.reshape(n, -1)                  # (n, M)
    s_sorted = np.sort(s_flat, axis=0)                 # (n, M) sorted along samples

    # E|X - y|
    term1 = np.mean(np.abs(s_flat - y_flat[np.newaxis, :]))

    # E|X - X'| via sorted-sample identity
    # Σ_{i,j} |x_i - x_j| = 2 Σ_k x_{(k)} (2k - n + 1)   (0-indexed)
    weights = (2 * np.arange(n) - n + 1).astype(np.float64)  # (n,)
    pairwise_sum = np.sum(
        s_sorted.astype(np.float64) * weights[:, np.newaxis], axis=0,
    )   # (M,)
    term2 = np.mean(pairwise_sum) / (n * n)            # divide by n² (double-sum)

    crps = term1 - term2
    return float(crps)


def coverage(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    level: float = 0.9,
) -> float:
    """Prediction-interval coverage at a given nominal level.

    Parameters
    ----------
    y_true : (...,)
    y_samples : (n_samples, ...)
    level : float in (0, 1)

    Returns
    -------
    float
        Fraction of true values falling within the prediction interval.
    """
    alpha = 1 - level
    lo = np.percentile(y_samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)
    within = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(within))


def interval_width(
    y_samples: np.ndarray,
    level: float = 0.9,
) -> float:
    """Average width of prediction intervals at a given level."""
    alpha = 1 - level
    lo = np.percentile(y_samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)
    return float(np.mean(hi - lo))
