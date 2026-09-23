# Bayesmark benchmark reproducibility

This directory contains the scripts and optimizer wrappers used to reproduce the Bayesmark experiments reported for `tidyHEBO`.

The benchmark compares five optimizers:

- `tidyHEBO`
- `HEBO`
- `LogEI`
- `optuna` (TPE)
- Bayesmark's built-in `RandomSearch`

Each optimizer/problem combination is evaluated for **100 optimization steps**.

## 1. Installation

The benchmark was run with Python 3.10. The supplied environment uses the project-specific Bayesmark fork at <https://github.com/Ch-LZ/bayesmark>.

```bash
mamba env create -f environment.yml
mamba activate tidyhebo-bayesmark
```

The original full Conda environment export is provided as `bayesmark_environment_full.yml`.

## 2. Check the commands without running them

```bash
python run_benchmark.py --dry-run
```

`python run_benchmark.py run --dry-run` is equivalent. The commands are stored in `jobs.txt`; the runner only adds the output directory and benchmark repeat.

## 3. Smoke test

The smoke test uses exactly the same runner as the complete benchmark, but only the first two commands in `jobs.txt` and two repeats. The first two commands are Optuna jobs and still use the full 100 optimization steps.

```bash
python run_benchmark.py smoke-test
python run_benchmark.py aggregate --results-dir smoke_results
```

The output is written to `smoke_results/part1` and `smoke_results/part2`, with logs in `smoke_logs/`.

A fresh run requires an empty output/state directory. To repeat the smoke test from scratch:

```bash
rm -rf smoke_results smoke_logs .smoke_state
python run_benchmark.py smoke-test
```

The supplied `baseline.json` is copied into each `derived/` directory before `bayesmark-anal`. It is the baseline distributed for the original NeurIPS 2020 Black Box Optimization Challenge; see the [challenge starter kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit).

## 4. Run the complete benchmark

```bash
python run_benchmark.py run --workers 8
```

By default, all five optimizers are run for 20 repeats. Results are written to `results/part1`, ..., `results/part20`, and per-job stdout/stderr is written to `logs/`.

## 5. Stop or resume a run

`Ctrl+C` stops the runner and its child processes. The run can then be continued with:

```bash
python run_benchmark.py resume --workers 8
```

Only unfinished jobs are submitted again. If only later repeats remain, use for example:

```bash
python run_benchmark.py resume --start-repeat 12 --workers 8
```

If the original terminal is unavailable, stop the runner from another shell with:

```bash
python run_benchmark.py stop
```

## 6. Aggregate and analyse results

After the jobs have finished:

```bash
python run_benchmark.py aggregate
```

This runs `bayesmark-agg` and `bayesmark-anal` for every `part*` directory found under the selected results directory and copies the historical `baseline.json` into each `derived/` directory before analysis.

Large benchmark outputs are not stored in Git. Archived results from Zenodo should be extracted into this directory so that the layout is `results/part1`, ..., `results/part20`.

## 7. Reproduce the barplot

`plot_results.ipynb` reproduces the barplot from the aggregated data in `results/part1`, ..., `results/part20`.

```bash
jupyter lab plot_results.ipynb
```
