"""Plotting helpers shared by the manuscript-facing notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OPTIMIZER_ORDER = ("tidyhebo", "logei", "hebo", "optuna", "random")
OPTIMIZER_LABELS = {
    "tidyhebo": "tidyHEBO",
    "logei": "LogEI",
    "hebo": "HEBO",
    "optuna": "Optuna",
    "random": "Random",
}
OPTIMIZER_COLORS = {
    "tidyhebo": "tab:red",
    "logei": "tab:orange",
    "hebo": "tab:purple",
    "optuna": "tab:green",
    "random": "tab:blue",
}


def load_values(root: str | Path, optimizer: str) -> np.ndarray:
    """Load a ``*_values.txt`` result matrix."""
    path = Path(root) / f"{optimizer.lower()}_values.txt"
    values = np.loadtxt(path)
    return np.atleast_2d(values)


def best_so_far(values: np.ndarray, *, minimize: bool = False) -> np.ndarray:
    """Convert raw observed values to per-run best-so-far trajectories."""
    values = np.asarray(values, dtype=float)
    if minimize:
        return np.minimum.accumulate(values, axis=1)
    return np.maximum.accumulate(values, axis=1)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean trajectory and a deterministic bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(n_resamples, values.shape[0]))
    boot_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(boot_means, [alpha, 1.0 - alpha], axis=0)
    return values.mean(axis=0), lower, upper


def plot_trajectory(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    minimize: bool = False,
    seed: int = 0,
    ci_outline: bool = False,
    ci_fill_alpha: float = 0.20,
) -> None:
    """Plot a mean best-so-far trajectory with a bootstrap 95% confidence band."""
    trajectories = best_so_far(values, minimize=minimize)
    mean, lower, upper = bootstrap_mean_ci(trajectories, seed=seed)
    x = np.arange(1, trajectories.shape[1] + 1)
    ax.plot(x, mean, label=label, color=color, linewidth=1.6)
    if ci_outline:
        ax.plot(x, lower, color=color, linewidth=1.5, alpha=0.5)
        ax.plot(x, upper, color=color, linewidth=1.5, alpha=0.5)
    ax.fill_between(x, lower, upper, color=color, alpha=ci_fill_alpha, linewidth=0)
