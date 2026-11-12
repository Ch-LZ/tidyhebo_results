"""Plot the archived NIAH and ZoMBI histories with the manuscript style."""

def render(root):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from bo_benchmark.analysis.metrics import best_so_far, median_curve
    from .paper import NIAH_DISPLAY, history_records
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    RESULTS_ROOT = Path(root) / 'data/historical/niah_reference'
    BUDGET = 100
    records, _ = history_records("niah", root)
    seeds = sorted({key[3] for key in records})
    current_methods = {"tidyhebo": "qmace_retained", "logei": "vanilla_gp_qlogei"}
    # src/tasks.py from the same repository commit, unchanged:
    from dataclasses import dataclass


    @dataclass(frozen=True)
    class Task:
        name: str
        direction: str = "minimize"
        display_name: str = ""
        display_scale: float = 1.0
        display_sign: float = 1.0
        needle_levels: tuple[float, ...] = ()
        default_oracle_relpath: str | None = None

        def display(self, y):
            return self.display_sign * y / self.display_scale


    TASKS = {
        "poisson_ratio": Task(
            name="poisson_ratio",
            display_name="Poisson ratio",
            needle_levels=(-1.2, -1.7),
            default_oracle_relpath="data/poisson/poisson_RF_trained.pkl",
        ),
        "thermoelectric": Task(
            name="thermoelectric",
            display_name="Thermoelectric figure of merit (ZT)",
            display_scale=NIAH_DISPLAY["thermoelectric"][1],
            display_sign=NIAH_DISPLAY["thermoelectric"][0],
            needle_levels=(1.4,),
            default_oracle_relpath="data/thermoelectric/zt_RF_trained.pkl",
        ),
    }

    METHODS = [
        ('tidyhebo', 'tidyHEBO', 'tab:red', '-'),
        ('logei', 'LogEI', 'tab:orange', '-'),
        ('yj_qlogei', 'tidyHEBO w/qLogEI', 'tab:brown', '-'),
        ('hebo', 'HEBO', 'tab:purple', '-'),
        ('optuna', 'optuna', 'tab:green', '-'),
        ('random', 'random', 'tab:blue', '-'),
        ('zombi_ada', 'ZoMBI (LCB Adaptive)', 'tab:blue', '--'),
        ('zombi_abrupt', 'ZoMBI (EI Abrupt)', 'tab:orange', '--'),
        ('zombi_EI', 'ZoMBI (EI)', 'tab:red', '--'),
        ('zombi_LCB', 'ZoMBI (LCB)', 'tab:green', '--'),
    ]

    NEEDLE_TEXT = {
        'poisson_ratio': [
            ('Needle 1', -1.2, 2.0, 0.04),
            ('Needle 2', -1.7, 2.0, 0.04),
        ],
        'thermoelectric': [
            ('Needle 1', 1.4, 40.0, -0.03),
        ],
    }

    PANEL_SPECS = [
        ('poisson_ratio', 'A'),
        ('thermoelectric', 'B'),
    ]


    def cumulative_min(values):
        return best_so_far(np.asarray(values, dtype=float), minimize=True)


    def load_trace(task_name, method, budget=BUDGET):
        if method.startswith('zombi_'):
            path = RESULTS_ROOT / task_name / f'{method}_values.txt'
            values = np.loadtxt(path, ndmin=2)
        else:
            variant = current_methods.get(method, method)
            values = np.asarray([records[task_name, variant, 1, seed]['y'] for seed in seeds])
        if values.shape[0] != 12 or values.shape[1] < budget or not np.isfinite(values).all():
            raise ValueError(f"Invalid paper histories: {task_name}/{method}")
        values = cumulative_min(values)
        return values[:, -budget:]


    def available_run_counts(task_name):
        counts = {}
        for method, label, _, _ in METHODS:
            try:
                counts[label] = load_trace(task_name, method).shape[0]
            except (OSError, ValueError):
                continue
        return counts

    def _plot_panel(ax, task_name, panel_letter, budget=BUDGET):
        task = TASKS[task_name]
        x = np.arange(1, budget + 1)
        handles = []
        labels = []

        plotted = 0
        for method, label, color, linestyle in METHODS:
            try:
                raw = load_trace(task_name, method, budget)
            except (OSError, ValueError):
                continue

            displayed = task.display(raw)
            median = median_curve(displayed)
            line, = ax.plot(
                x,
                median,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            handles.append(line)
            labels.append(label)
            plotted += 1

        if plotted == 0:
            raise FileNotFoundError(f'No result traces found for task: {task_name}')

        for needle_label, level, x_text, y_offset in NEEDLE_TEXT[task_name]:
            ax.axhline(level, color='black', linestyle='--', linewidth=1.1, alpha=0.8)
            ax.text(
                x_text,
                level + y_offset,
                needle_label,
                fontsize=9,
                ha='left',
                va='bottom' if y_offset >= 0 else 'top',
            )

        ylabel = 'best value'
        ax.set_xlabel('number of function evaluations')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, budget)
        ax.grid(alpha=0.35)

        ax.text(
            -0.12,
            1.00,
            panel_letter,
            transform=ax.transAxes,
            fontsize=19,
            fontweight='bold',
            ha='left',
            va='top',
            clip_on=False,
        )

        return handles, labels


    def plot_manuscript_figure(budget=BUDGET):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

        handles = labels = None
        for ax, (task_name, panel_letter) in zip(axes, PANEL_SPECS):
            panel_handles, panel_labels = _plot_panel(ax, task_name, panel_letter, budget)
            if handles is None:
                handles, labels = panel_handles, panel_labels

        legend = fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        fig.tight_layout(rect=(0.02, 0.22, 1.0, 1.0))
        return fig, axes, legend

    for task_name, _ in PANEL_SPECS:
        for method, _, _, _ in METHODS:
            load_trace(task_name,method)
    manuscript_fig, _, legend = plot_manuscript_figure()
    out = Path(root) / 'figures/niah'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        manuscript_fig.savefig(out / ('Figure_5_NIAH.'+ext), dpi=240, facecolor='white',
                               bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.close(manuscript_fig)
    return out
