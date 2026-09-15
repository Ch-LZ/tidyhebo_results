"""Shared robustness metrics for the tidyHEBO reproducibility analyses.

The manuscript uses the historical name ``nAUC`` for the trapezoidal area under
normalized regret. All benchmark runs in this repository use the same budget
(100 evaluations), so the area is intentionally not divided by the number of
steps. Keeping the implementation here prevents notebooks from silently
reimplementing the metrics in slightly different ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml


OLYMPUS_CONFIG_SUBDIRS = ("olympus 2021", "olympus 2023")


def scale_utility(values, y_best: float, y_worst: float) -> np.ndarray:
    """Map objective values to utility where 1 is best and 0 is worst."""
    values = np.asarray(values, dtype=float)
    if y_best == y_worst:
        return np.ones_like(values, dtype=float)
    return (values - y_worst) / (y_best - y_worst)


def find_dataset_config(
    dataset: str,
    base_path: Path,
    subdirs: Sequence[str] = OLYMPUS_CONFIG_SUBDIRS,
) -> Path:
    """Return the YAML configuration file for an Olympus task."""
    base_path = Path(base_path)
    for subdir in subdirs:
        candidate = base_path / subdir / f"{dataset}.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {dataset}.yaml under {base_path} in {tuple(subdirs)}"
    )


def load_task_config(dataset: str, configs_root: Path) -> dict:
    """Load one Olympus task configuration."""
    path = find_dataset_config(dataset, Path(configs_root))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_ragged_batch_sizes(path: str | Path) -> list[list[int]]:
    """Load variable-length adaptive-batch logs written one run per line."""
    rows: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append([int(round(float(value))) for value in line.split()])
    return rows


def incumbent_utility(
    utility: np.ndarray,
    *,
    batch_sizes: Sequence[Sequence[int]] | None = None,
    sequential_initial_evaluations: int = 4,
) -> np.ndarray:
    """Return best-so-far utility, optionally respecting adaptive feedback rounds.

    Sequential runs use the ordinary cumulative maximum. For adaptive batching,
    the first ``sequential_initial_evaluations`` points retain the historical
    tidyHEBO initialization treatment used in the manuscript analysis. After
    that initialization, observations from the same proposed batch update the
    incumbent only when the batch is complete.
    """
    utility = np.asarray(utility, dtype=float)
    if utility.ndim == 1:
        utility = utility[None, :]

    if batch_sizes is None:
        return np.maximum.accumulate(utility, axis=1)

    if len(batch_sizes) != utility.shape[0]:
        raise ValueError(
            "The number of batch-size rows must equal the number of optimization runs."
        )

    result = np.empty_like(utility, dtype=float)
    n_eval = utility.shape[1]
    n_init = min(sequential_initial_evaluations, n_eval)

    for run_idx, (run_utility, run_batches) in enumerate(zip(utility, batch_sizes)):
        if sum(run_batches) != n_eval:
            raise ValueError(
                f"Run {run_idx}: batch sizes sum to {sum(run_batches)}, expected {n_eval}."
            )

        result[run_idx, :n_init] = np.maximum.accumulate(run_utility[:n_init])
        incumbent = result[run_idx, n_init - 1] if n_init else -np.inf

        start = 0
        for batch_size in run_batches:
            end = start + int(batch_size)
            if end <= n_init:
                start = end
                continue

            active_start = max(start, n_init)
            if end - active_start > 1:
                result[run_idx, active_start : end - 1] = incumbent
            incumbent = max(incumbent, float(run_utility[active_start:end].max()))
            result[run_idx, end - 1] = incumbent
            start = end

    return result


def regret_auc(
    values: np.ndarray,
    y_best: float,
    y_worst: float,
    *,
    batch_sizes: Sequence[Sequence[int]] | None = None,
    sequential_initial_evaluations: int = 4,
) -> np.ndarray:
    """Compute one normalized-regret AUC value per optimization run."""
    utility = scale_utility(values, y_best=y_best, y_worst=y_worst)
    incumbent = incumbent_utility(
        utility,
        batch_sizes=batch_sizes,
        sequential_initial_evaluations=sequential_initial_evaluations,
    )
    regret = 1.0 - incumbent
    return np.trapezoid(regret, x=np.arange(regret.shape[1]), axis=1)


def summarize_auc(auc: Iterable[float], cvar_alpha: float = 0.90) -> dict[str, float]:
    """Summarize run-level AUC by median, IQR, and upper-tail CVaR."""
    auc = np.asarray(list(auc), dtype=float)
    if auc.ndim != 1 or auc.size == 0:
        raise ValueError("auc must be a non-empty one-dimensional array")

    q25, q75 = np.quantile(auc, [0.25, 0.75])
    threshold = np.quantile(auc, cvar_alpha)
    tail = auc[auc >= threshold]
    return {
        "median_nAUC": float(np.median(auc)),
        "IQR_nAUC": float(q75 - q25),
        "CVaR_nAUC": float(tail.mean()),
    }


def calculate_metrics(
    values: np.ndarray,
    y_best: float,
    y_worst: float,
    *,
    batch_sizes: Sequence[Sequence[int]] | None = None,
    sequential_initial_evaluations: int = 4,
) -> dict[str, float]:
    """Compute the three manuscript robustness metrics from optimization traces."""
    auc = regret_auc(
        values,
        y_best,
        y_worst,
        batch_sizes=batch_sizes,
        sequential_initial_evaluations=sequential_initial_evaluations,
    )
    return summarize_auc(auc)
