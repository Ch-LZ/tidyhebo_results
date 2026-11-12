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


def final_regret(values, y_best: float, y_worst: float) -> float:
    """Final normalized regret; preserve the archived reference convention."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite trajectory")
    if y_best == y_worst:
        return 0.0
    best = values.min() if y_best < y_worst else values.max()
    return float((best - y_best) / (y_worst - y_best))


def average_ranks(values) -> np.ndarray:
    """Ascending one-based ranks, with average ranks for exact ties."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite vector")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2)[inverse]


def best_so_far(values, *, minimize=True, batch_sizes=None, n_init=4,
                hide_initial=False) -> np.ndarray:
    """Physical-scale incumbent with the archived feedback convention."""
    values = np.asarray(values, dtype=float)
    sign = -1.0 if minimize else 1.0
    if batch_sizes is None:
        return np.maximum.accumulate(sign * values, axis=-1) * sign
    curve = incumbent_utility(sign * values, batch_sizes=[batch_sizes],
                              sequential_initial_evaluations=n_init)[0] * sign
    if hide_initial and n_init == 0:
        curve[:batch_sizes[0] - 1] = np.nan
    return curve


def median_curve(histories):
    histories = np.asarray(histories,dtype=float)
    if histories.ndim != 2 or not histories.size or not np.isfinite(histories).all():
        raise ValueError('Expected finite run-by-evaluation histories')
    return np.median(histories,axis=0)


def curve_quartiles(histories):
    return np.quantile(np.asarray(histories,dtype=float),[.25,.5,.75],axis=0)


def needle_statistics(curves, threshold, *, higher=False, inclusive=True):
    curves = np.asarray(curves,dtype=float)
    if higher:
        passed = curves >= threshold if inclusive else curves > threshold
    else:
        passed = curves <= threshold if inclusive else curves < threshold
    success = passed.any(axis=1)
    first = passed[success].argmax(axis=1)+1
    return dict(n_hits=int(success.sum()),success_rate=float(success.mean()),
                median_first_hit_given_success=float(np.median(first)) if len(first) else np.nan)


def upper_tail_mean(values, alpha=.90) -> float:
    """Empirical CVaR: include every value at or above the quantile, including ties."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    return float(values[values >= np.quantile(values, alpha)].mean())


def trajectory_metrics(values, y_best, y_worst, *, batch_sizes=None, n_init=4):
    """Both historical AUC scales, and final regret, from one definition."""
    if len(values) < 2:
        raise ValueError("At least two evaluations are required for AUC")
    area = float(regret_auc(values, y_best, y_worst, batch_sizes=batch_sizes,
                           sequential_initial_evaluations=n_init)[0])
    return dict(nAUC=area, mean_regret_auc=area / (len(values) - 1),
                final_regret=final_regret(values, y_best, y_worst))


def olympus_run_table(records, tasks, methods, qs, seeds):
    """Strict paper coverage and paired initial designs; no partial selection."""
    import pandas as pd
    from .comparison import protocol_key, validate_record

    wanted = {(t, m, q, s) for t in tasks for m in methods for q in qs for s in seeds}
    selected = {k: r for k, r in records.items() if k[0] in tasks and k[1] in methods and k[2] in qs}
    if set(selected) != wanted:
        missing = sorted(wanted - set(selected))
        extra = sorted(set(selected) - wanted)
        raise ValueError(f"Paper coverage mismatch: {len(missing)} missing, {len(extra)} extra; "
                         f"first missing={missing[:3]}")
    rows = []
    initial_points = {}
    for task in tasks:
        for seed in seeds:
            for q in qs:
                group = [selected[task, m, q, seed] for m in methods]
                if len({protocol_key(r) for r in group}) != 1:
                    raise ValueError(f"Incompatible protocols: {task}, q={q}, seed={seed}")
                for method, record in zip(methods, group):
                    s = record['spec']
                    if (record['status'] != 'completed' or s['n_trials'] != 100
                            or s['n_init'] != 4 or s['seed'] != seed):
                        raise ValueError(f"Invalid paper trajectory: {task}, {method}, q={q}, seed={seed}")
                    points = record.get('initial_x')
                    if points is None and record.get('optimizer_x'):
                        points = np.concatenate(record['optimizer_x'], axis=0)[:s['n_init']]
                    if points is None or len(points) != s['n_init']:
                        raise ValueError("Missing paired initial X")
                    key = task, seed
                    if key in initial_points and not np.array_equal(initial_points[key], points):
                        raise ValueError(f"Unpaired initial X: {task}, {method}, q={q}, seed={seed}")
                    initial_points[key] = np.asarray(points)
                    ref = validate_record(record)['normalization']
                    values = trajectory_metrics(record['y'], ref['y_best'], ref['y_worst'],
                        batch_sizes=[record['batches']], n_init=s['n_init'])
                    rows.append(dict(task=task, optimizer=method, q=q, seed=seed, **values))
    return pd.DataFrame(rows)


def olympus_statistics(runs, methods, qs, *, baseline='vanilla_gp_qlogei',
                       alpha=.90, draws=10000, seed=20261105):
    """Task means/ranks/tails, then equal task weighting and paired bootstrap."""
    import pandas as pd

    metrics = ('final_regret', 'nAUC')
    task_rows = []
    for (task, method, q), group in runs.groupby(['task', 'optimizer', 'q']):
        for metric in metrics:
            values = group[metric].to_numpy()
            task_rows.append(dict(task=task, optimizer=method, q=q, metric=metric,
                mean=float(values.mean()), CVaR=upper_tail_mean(values, alpha), n_repeats=len(values)))
    table = pd.DataFrame(task_rows)
    table['rank'] = table.groupby(['task','q','metric'])['mean'].transform(
        lambda values: average_ranks(values.to_numpy()))
    tasks = sorted(table.task.unique())
    indices = np.random.default_rng(seed).integers(len(tasks), size=(draws, len(tasks)))
    rows = []
    for metric in metrics:
        for q in qs:
            part = table[(table.metric == metric) & (table.q == q)]
            for column, panel in [('rank','rank'),('mean','mean_delta'),('CVaR','cvar_delta')]:
                matrix = part.pivot(index='task', columns='optimizer', values=column).reindex(
                    index=tasks, columns=methods)
                if panel != 'rank':
                    matrix = matrix.subtract(matrix[baseline], axis=0)
                values = matrix.to_numpy()
                lows, highs = np.quantile(values[indices].mean(axis=1), [.025,.975], axis=0)
                for method, value, low, high in zip(methods,values.mean(axis=0),lows,highs):
                    rows.append(dict(metric=metric,optimizer=method,q=q,panel=panel,
                                     mean=value,low=low,high=high))
    return table, pd.DataFrame(rows)


def sequential_olympus_summary(task_metrics, summary):
    """Arrange shared Olympus statistics for the sequential plotting recipe."""
    result = task_metrics.groupby(['metric','optimizer']).agg(
        mean=('mean','mean'), CVaR=('CVaR','mean'), rank=('rank','mean'),
        rank_sem=('rank','sem')).fillna({'rank_sem':0})
    for row in summary.itertuples():
        for suffix, value in [('',row.mean),('_low',row.low),('_high',row.high)]:
            result.loc[(row.metric,row.optimizer),row.panel+suffix] = value
    return result


def olympus_task_robustness(runs):
    """Supplementary S2: absolute median and upper-tail AUC, without IQR."""
    import pandas as pd
    rows = []
    for (task,method), group in runs[runs.q == 1].groupby(['task','optimizer']):
        stats = summarize_auc(group.nAUC.to_numpy())
        rows.append(dict(task=task,optimizer=method,median_nAUC=stats['median_nAUC'],
                         CVaR_nAUC=stats['CVaR_nAUC']))
    return pd.DataFrame(rows)


def paired_mean_interval(values, *, draws=10000, seed=20261105):
    """Pointwise percentile interval; the first axis is the paired sampling unit."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or draws < 1:
        raise ValueError("Expected finite paired values and positive draws")
    indices = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975], axis=0)


def repeat_score_statistics(values):
    """Bayesmark paper bars: mean scores and paired repeat-bootstrap endpoints."""
    values = np.asarray(values,dtype=float)
    return values.mean(axis=0), paired_mean_interval(values,draws=5000,seed=42)


def bayesmark_scores(losses, tasks, methods):
    """Archived Bayesmark 0.0.8 score, with the pooled random baseline held fixed.

    Run in the pinned Bayesmark analysis environment. Input axes are
    (batch, point, task, method, repeat). No learner/oracle is executed here.
    """
    import xarray as xr
    from bayesmark import constants as cc
    from bayesmark.experiment_baseline import compute_baseline
    from bayesmark.experiment_analysis import compute_aggregates, get_perf_array
    from bayesmark.np_util import linear_rescale

    losses = np.asarray(losses, dtype=float)
    if (losses.ndim != 5 or losses.shape[2:4] != (len(tasks), len(methods))
            or not np.isfinite(losses).all() or "random" not in methods):
        raise ValueError("Invalid paired Bayesmark losses")
    names = [cc.RANDOM_SEARCH + cc.ARG_DELIM + 'local' if m == 'random' else m for m in methods]
    aggregate = xr.DataArray(losses,
        dims=(cc.ITER, cc.SUGGEST, cc.TEST_CASE, cc.METHOD, cc.TRIAL),
        coords={cc.ITER: np.arange(losses.shape[0]), cc.SUGGEST: np.arange(losses.shape[1]),
                cc.TEST_CASE: tasks, cc.METHOD: names, cc.TRIAL: np.arange(losses.shape[-1])},
        name=cc.VISIBLE_TO_OPT)
    baseline = compute_baseline(aggregate)
    by_task, summary = compute_aggregates(aggregate, baseline, visible_perf_da=aggregate)
    curves = 100 * (1 - summary[cc.PERF_MEAN].values)
    task_scores = 100 * (1 - by_task[cc.NORMED_MEAN].isel({cc.ITER: -1}).transpose(cc.TEST_CASE, cc.METHOD).values)
    trials = np.empty((len(tasks), len(methods), losses.shape[-1]))
    for ti, task in enumerate(tasks):
        for mi, _ in enumerate(methods):
            performance = get_perf_array(losses[:, :, ti, mi, :], losses[:, :, ti, mi, :])[-1]
            reference = baseline.sel({cc.TEST_CASE: task}, drop=True)
            normalized = linear_rescale(performance, reference[cc.PERF_BEST].item(),
                reference[cc.PERF_CLIP].item(), 0., 1., enforce_bounds=False)
            trials[ti, mi] = 100 * (1 - np.clip(normalized, -1., 1.))
    repeats = trials.mean(axis=0).T
    if not np.allclose(repeats.mean(axis=0), curves[-1], rtol=1e-10, atol=1e-10):
        raise ValueError("Repeat aggregation disagrees with Bayesmark")
    return curves, task_scores, repeats


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
            if line.strip() and not line.lstrip().startswith("#"):
                values = [float(value) for value in line.split()]
                if any(not np.isfinite(value) or value < 1 or not value.is_integer()
                       for value in values):
                    raise ValueError(f"{path}: batch sizes must be positive integers")
                rows.append([int(value) for value in values])
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
    incumbent only when the batch is complete. Use zero for
    ``sequential_initial_evaluations`` to respect *all* recorded batches,
    including initialization. Before the first feedback, utility is zero
    (the declared worst reference); subsequent values are not clipped.
    """
    utility = np.asarray(utility, dtype=float)
    if utility.ndim == 1:
        utility = utility[None, :]

    if utility.ndim != 2 or not utility.size or not np.isfinite(utility).all():
        raise ValueError("Utility must be a nonempty finite run-by-evaluation matrix")
    if (not isinstance(sequential_initial_evaluations, (int, np.integer))
            or sequential_initial_evaluations < 0):
        raise ValueError("sequential_initial_evaluations must be a nonnegative integer")
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
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
               or n < 1 for n in run_batches):
            raise ValueError(f"Run {run_idx}: batch sizes must be positive integers")
        if sum(run_batches) != n_eval:
            raise ValueError(
                f"Run {run_idx}: batch sizes sum to {sum(run_batches)}, expected {n_eval}."
            )

        result[run_idx, :n_init] = np.maximum.accumulate(run_utility[:n_init])
        incumbent = result[run_idx, n_init - 1] if n_init else None

        start = 0
        for batch_size in run_batches:
            end = start + int(batch_size)
            if end <= n_init:
                start = end
                continue

            active_start = max(start, n_init)
            if end - active_start > 1:
                result[run_idx, active_start : end - 1] = 0.0 if incumbent is None else incumbent
            observed = float(run_utility[active_start:end].max())
            incumbent = observed if incumbent is None else max(incumbent, observed)
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
    return {
        "median_nAUC": float(np.median(auc)),
        "IQR_nAUC": float(q75 - q25),
        "CVaR_nAUC": upper_tail_mean(auc, cvar_alpha),
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


def runtime_metric_rows(records):
    """Live status-table metrics; serialization remains in the application."""
    from collections import defaultdict
    rows=[];groups=defaultdict(list);directions={};expected=defaultdict(set)
    for record in records:
        label=record['spec']
        expected[(label['task'],label['q'],label['repeat'])].add(label['variant'])
        if record['status']!='completed':continue
        directions.setdefault(label['task'],record['minimize'])
        best=best_so_far(record['y'], minimize=record['minimize'])
        row={**{k:label[k] for k in ('task','variant','q','repeat','seed')},
             'best':best[-1],'mean_best':sum(best)/len(best),'seconds':record['elapsed'],'rank':None}
        rows.append(row);groups[(label['task'],label['q'],label['repeat'])].append(row)
    for key,group in groups.items():
        if {row['variant'] for row in group}!=expected[key]:continue
        minimize=directions[key[0]]
        ranks=average_ranks([row['best'] if minimize else -row['best'] for row in group])
        for row,rank in zip(group,ranks):
            row['rank']=float(rank)
    return rows
