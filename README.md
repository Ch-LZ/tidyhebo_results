# tidyHEBO paper results

Reproduce the Olympus, NIAH and Bayesmark experiments reported in the tidyHEBO
paper. The optimizer library is available at
[Ch-LZ/tidyhebo](https://github.com/Ch-LZ/tidyhebo).

## Rerun the experiments

Run these commands from the repository root in a terminal with Git, Python
3.10 or newer, and Mamba (for example, from Miniforge).

### 1. Set up the environments

```sh
python -m pip install PyYAML
python run_experiments.py setup all
```

Setup creates isolated CPU environments using the versions in
[`environments/`](environments/). It fetches tidyHEBO from the GitHub release tag
`v0.1.0` and checks it against the commit in
[`environments/sources.yaml`](environments/sources.yaml). It also fetches the
pinned Olympus emulators, prepares the NIAH oracles from the pinned ZoMBI
assets, and sets up the legacy Bayesmark 0.0.8 oracle.

### 2. Run the smoke tests

```sh
mamba run -p environments/optimizers/scientific python run_experiments.py smoke all
```

The smoke tests use the real oracles and all six methods with one repeat and
one optimization batch after initialization. They cover two tasks per
benchmark and all five Olympus batch sizes. Use `olympus`, `niah` or
`bayesmark` in place of `all` to test one experiment. The smoke tests write
to `results/smoke/` and check completion, finite objectives and matching
initial points across methods.

### 3. Run the full paper campaigns

```sh
python run_experiments.py run olympus
python run_experiments.py run niah
python run_experiments.py run bayesmark
```

Run the commands one at a time. Each experiment uses six methods and derives
its repeat seeds from master seed `42`. The budgets below include initialization.

| Experiment | Configuration | Tasks | Repeats | Batch sizes | Initial points | Evaluations per run |
|---|---|---:|---:|---|---:|---:|
| Olympus | [OLY-002.json](experiments/OLY-002.json) | 19 | 30 | 1, 2, 4, 8, 16 | 4 | 100 |
| NIAH | [NIAH-005.json](experiments/NIAH-005.json) | 2 | 12 | 1 | 4 | 100 |
| Bayesmark | [BAYESMARK-Q8-FULL108.json](experiments/BAYESMARK-Q8-FULL108.json) | 108 | 15 | 8 | 8 | 128 |

The NIAH campaign reproduces the paper's sequential runs on Poisson ratio and
thermoelectric. The ZoMBI comparison curves are archived with the paper data.

Each config sets its campaign directory under `results/`. Inside that
directory, `results/trajectories/` holds the per-run JSON trajectories and
`results/summary.json` records run status. Use
`python run_experiments.py status olympus` to inspect progress and
`python run_experiments.py resume olympus` to continue an interrupted run.
Replace `olympus` with `niah` or `bayesmark` as needed. Keep the same source
and environment versions when resuming from a native checkpoint.

## Rebuild the paper figures and metrics

The commands below use the 26,964 archived Parquet trajectories and seeds in
[`data/historical/`](data/historical/). You can run them in a separate Python
environment without setting up the benchmark oracles.

```sh
python -m pip install -e ".[analysis]"
python run_experiments.py check all
python run_experiments.py figures all
```

The exact analysis environment is defined in
[`environments/analysis.yaml`](environments/analysis.yaml). `check all`
validates the archive hashes, run coverage, seeds, budgets and shared initial
designs. `figures all` writes PNG, PDF, SVG and derived tables under `figures/`.
Use `figures olympus`, `figures niah` or `figures bayesmark` for one group.

| Paper figure | Recipe or notebook with saved output |
|---|---|
| Fig. 3 · Olympus sequential | [03_olympus_sequential.ipynb](notebooks/03_olympus_sequential.ipynb) |
| Fig. 4 · Olympus batch | [04_olympus_batch.ipynb](notebooks/04_olympus_batch.ipynb) |
| Fig. 5 · NIAH | [05_niah.ipynb](notebooks/05_niah.ipynb) |
| Fig. 6 · Bayesmark | [06_bayesmark.ipynb](notebooks/06_bayesmark.ipynb) |
| Fig. S2 · Olympus per-task comparisons | `python run_experiments.py figures olympus` |

The workflow diagram and illustrative artwork in the manuscript have separate
sources. To restore the compact history into an older source-only checkout, run
`python run_experiments.py import-data path/to/tidyhebo_paper_history.zip`.

Use `python run_experiments.py metrics olympus` or
`python run_experiments.py metrics niah` to write tables from the archived
trajectories under `analysis/`. The Bayesmark figure uses the archived
repeat-score table. To recompute those scores from the archived losses after
setup, run:

```sh
mamba run -p environments/optimizers/scientific python run_experiments.py metrics bayesmark
```

The shared metric definitions are in
[`src/bo_benchmark/analysis/metrics.py`](src/bo_benchmark/analysis/metrics.py).

## License

See [LICENSE.md](LICENSE.md). Third-party sources retain their own licenses.
