"""Supplementary S2, using the original grouped-bar layout and central metrics."""
import numpy as np
import matplotlib.pyplot as plt
from bo_benchmark.analysis.metrics import olympus_run_table, olympus_task_robustness
from bo_benchmark.seeds import SeedManager
from bo_benchmark.task_catalog import load_catalog
from .paper import METHODS, history_records

OPTIMIZER_ORDER = METHODS
OPTIMIZER_LABELS = {'random':'Random','optuna':'Optuna','hebo':'HEBO',
    'vanilla_gp_qlogei':'qLogEI','yj_qlogei':'tidyHEBO w/qLogEI','qmace_retained':'tidyHEBO'}
OPTIMIZER_COLORS = dict(zip(METHODS,['tab:blue','tab:green','tab:purple','tab:orange','tab:brown','tab:red']))

def plot_metric_bars(
    ax,
    df,
    metric,
    task_order=None,
    value_fmt=".3g",
    clip_value=None,
    clip_prefix="> ",
    label_fontsize=8,
):
    """Grouped horizontal bars: one group per task, one bar per optimizer."""
    if task_order is None:
        task_order = list(df["task"].drop_duplicates())

    n_opt = len(OPTIMIZER_ORDER)
    y = np.arange(len(task_order))
    group_height = 0.78
    bar_height = group_height / n_opt

    for i, optimizer in enumerate(OPTIMIZER_ORDER):
        tmp = (
            df[df["optimizer"] == optimizer]
            .set_index("task")
            .reindex(task_order)
        )
        values = tmp[metric].to_numpy(dtype=float)
        offset = (i - (n_opt - 1) / 2) * bar_height
        ypos = y + offset

        display_values = []
        clipped_flags = []
        for v in values:
            if np.isnan(v):
                display_values.append(np.nan)
                clipped_flags.append(False)
            elif clip_value is not None and v > clip_value:
                display_values.append(clip_value)
                clipped_flags.append(True)
            else:
                display_values.append(v)
                clipped_flags.append(False)

        bars = ax.barh(
            ypos,
            display_values,
            height=bar_height * 0.88,
            color=OPTIMIZER_COLORS[optimizer],
            label=OPTIMIZER_LABELS[optimizer],
        )

        for bar, actual, is_clipped in zip(bars, values, clipped_flags):
            if np.isnan(actual):
                continue
            text = (
                f"{clip_prefix}{actual:{value_fmt}}"
                if is_clipped
                else f"  {actual:{value_fmt}}"
            )
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                text,
                va="center",
                ha="left",
                fontsize=label_fontsize,
            )

    ax.set_yticks(y, task_order)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    xmin, xmax = ax.get_xlim()
    if clip_value is not None:
        ax.set_xlim(xmin, max(xmax, clip_value * 1.25))
    else:
        ax.set_xlim(xmin, xmax * 1.12)


def plot_multiple_metric_bars(
    df,
    metrics,
    task_order=None,
    figsize=(14.0, 10.0),
    clip_values=None,
    value_fmt=".3g",
):
    """One panel per metric; each panel is a grouped horizontal bar chart."""
    n_metrics = len(metrics)
    if task_order is None:
        task_order = list(df["task"].drop_duplicates())
    if clip_values is None:
        clip_values = [None] * n_metrics

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric, clip_value in zip(axes, metrics, clip_values):
        plot_metric_bars(
            ax,
            df,
            metric,
            task_order=task_order,
            value_fmt=value_fmt,
            clip_value=clip_value,
        )
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelrotation=30, labelsize=10)

    for ax in axes[1:]:
        ax.set_yticklabels([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(OPTIMIZER_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.14, bottom=0.09, wspace=0.14)
    return fig, axes



def render(root):
    from pathlib import Path
    records,_ = history_records('olympus',root)
    tasks = sorted(load_catalog())
    runs = olympus_run_table(records,tasks,METHODS,(1,),SeedManager(42,30).seeds)
    metrics = olympus_task_robustness(runs)
    fig,_ = plot_multiple_metric_bars(metrics,['median_nAUC','CVaR_nAUC'],
        task_order=tasks,figsize=(10.,10.),clip_values=[40.,40.])
    out = Path(root)/'figures/supplement'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('Figure_S2_Olympus_tasks.'+ext),dpi=300,bbox_inches='tight')
    metrics.to_csv(out/'Figure_S2_values.csv',index=False)
    plt.close(fig)
    return out
