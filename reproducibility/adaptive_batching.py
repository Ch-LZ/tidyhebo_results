"""Analysis and plotting helpers for the Olympus adaptive-batching experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import calculate_metrics, load_ragged_batch_sizes, load_task_config


CAPS = (2, 4, 8, 16)
CAP_COLORS = {
    2: "tab:blue",
    4: "tab:orange",
    8: "tab:green",
    16: "tab:red",
}
METRIC_DISPLAY = {
    "median_nAUC": "Quality",
    "CVaR_nAUC": "CVaR",
    "IQR_nAUC": "IQR",
}
METRIC_ORDER = ("median_nAUC", "CVaR_nAUC", "IQR_nAUC")



def build_adaptive_summary(
    sequential_root: str | Path,
    adaptive_root: str | Path,
    configs_root: str | Path,
    *,
    caps: Iterable[int] = CAPS,
) -> pd.DataFrame:
    """Build task-level adaptive/sequential metric ratios and feedback reductions."""
    sequential_root = Path(sequential_root)
    adaptive_root = Path(adaptive_root)
    configs_root = Path(configs_root)

    rows: list[dict[str, float | int | str]] = []
    task_dirs = sorted(path for path in sequential_root.iterdir() if path.is_dir())
    for task_dir in task_dirs:
        task = task_dir.name
        config = load_task_config(task, configs_root)
        sequential_values = np.atleast_2d(
            np.loadtxt(task_dir / "tidyhebo_values.txt", dtype=float)
        )
        sequential_metrics = calculate_metrics(
            sequential_values,
            config["y_p_star"],
            config["y_p_w"],
        )

        for cap in caps:
            setting_dir = adaptive_root / f"adaptive_bs_{cap}" / task
            values = np.atleast_2d(
                np.loadtxt(setting_dir / "tidyhebo_values.txt", dtype=float)
            )
            batch_rows = load_ragged_batch_sizes(
                setting_dir / "tidyhebo_n_from_opt.txt"
            )
            if len(batch_rows) != values.shape[0]:
                raise ValueError(
                    f"{setting_dir}: found {len(batch_rows)} batch logs for "
                    f"{values.shape[0]} result runs"
                )
            adaptive_metrics = calculate_metrics(
                values,
                config["y_p_star"],
                config["y_p_w"],
                batch_sizes=batch_rows,
            )
            feedback_rounds = np.asarray([len(row) for row in batch_rows], dtype=float)
            feedback_reduction = 1.0 - float(feedback_rounds.mean()) / values.shape[1]

            record: dict[str, float | int | str] = {
                "task": task,
                "cap": int(cap),
                "feedback_reduction": feedback_reduction,
                "mean_feedback_rounds": float(feedback_rounds.mean()),
            }
            for metric in METRIC_ORDER:
                record[f"{metric}_ratio"] = (
                    adaptive_metrics[metric] / sequential_metrics[metric]
                )
            rows.append(record)

    return pd.DataFrame(rows)


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return a median and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    boot = np.median(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return float(np.median(values)), float(lo), float(hi)


def cross_task_summary(task_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate task-level adaptive-batching quantities by cap."""
    records = []
    for cap, group in task_level.groupby("cap", sort=True):
        record = {
            "cap": int(cap),
            "median_feedback_reduction": float(group["feedback_reduction"].median()),
        }
        for i, metric in enumerate(METRIC_ORDER):
            median, lo, hi = _bootstrap_median_ci(
                group[f"{metric}_ratio"].to_numpy(), seed=1000 * int(cap) + i
            )
            record[f"{metric}_ratio"] = median
            record[f"{metric}_ratio_ci_low"] = lo
            record[f"{metric}_ratio_ci_high"] = hi
        records.append(record)
    return pd.DataFrame(records)


def _plot_tradeoff_panel(
    ax: plt.Axes,
    task_level: pd.DataFrame,
    metric: str,
    *,
    panel_label: str,
    y_max: float = 3.0,
) -> None:
    """Draw one task-level quality-versus-feedback trade-off panel."""
    ratio_col = f"{metric}_ratio"
    for cap in sorted(task_level["cap"].unique()):
        group = task_level[task_level["cap"].eq(cap)]
        x = 100.0 * group["feedback_reduction"].to_numpy()
        y = group[ratio_col].to_numpy()
        color = CAP_COLORS[int(cap)]

        inside = y <= y_max
        ax.scatter(
            x[inside],
            y[inside],
            s=30,
            alpha=0.78,
            color=color,
            label=f"cap {int(cap)}",
            zorder=3,
        )
        if np.any(~inside):
            clipped_y = np.full((~inside).sum(), y_max - 0.03)
            ax.scatter(
                x[~inside],
                clipped_y,
                marker="v",
                s=42,
                color="black",
                zorder=5,
            )

        ax.axhline(
            float(np.median(y)),
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )
        ax.axvline(
            float(np.median(x)),
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1, zorder=2)
    ax.set_ylim(0.45, y_max)
    ax.set_xlabel("Reduction in feedback rounds (%)")
    ax.set_ylabel(f"Relative {METRIC_DISPLAY[metric]} (adaptive / sequential)")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        -0.14,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def plot_adaptive_batching_figure(
    task_level: pd.DataFrame,
    *,
    y_max: float = 3.0,
    figsize: tuple[float, float] = (11.2, 8.2),
) -> tuple[plt.Figure, np.ndarray]:
    """Reproduce the four-panel adaptive-batching manuscript figure."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    panel_metrics = ("median_nAUC", "CVaR_nAUC", "IQR_nAUC")
    for ax, metric, label in zip(axes.flat[:3], panel_metrics, "ABC"):
        _plot_tradeoff_panel(
            ax,
            task_level,
            metric,
            panel_label=label,
            y_max=y_max,
        )

    ax = axes.flat[3]
    summary = cross_task_summary(task_level)
    x = np.arange(len(METRIC_ORDER), dtype=float)
    caps = summary["cap"].astype(int).tolist()
    width = 0.18
    offsets = (np.arange(len(caps)) - (len(caps) - 1) / 2.0) * width

    for offset, cap in zip(offsets, caps):
        row = summary.loc[summary["cap"].eq(cap)].iloc[0]
        heights = np.asarray([row[f"{metric}_ratio"] for metric in METRIC_ORDER])
        lows = np.asarray([row[f"{metric}_ratio_ci_low"] for metric in METRIC_ORDER])
        highs = np.asarray([row[f"{metric}_ratio_ci_high"] for metric in METRIC_ORDER])
        yerr = np.vstack((heights - lows, highs - heights))
        ax.bar(
            x + offset,
            heights,
            width=width,
            color=CAP_COLORS[cap],
            alpha=0.82,
            label=f"cap {cap}",
            yerr=yerr,
            capsize=2.5,
            linewidth=0,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1)
    ax.set_xticks(x, [METRIC_DISPLAY[metric] for metric in METRIC_ORDER])
    ax.set_ylabel("Median relative metric across tasks")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        -0.14,
        1.04,
        "D",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.10, wspace=0.28, hspace=0.30)
    return fig, axes
