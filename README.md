# tidyHEBO benchmark results

This repository contains the archived optimization traces and the small amount of analysis code needed to reproduce the figures and tables reported for tidyHEBO. The archived objective-value matrices are kept unchanged; the manuscript-facing notebooks rebuild the analyses directly from those files.

## Environments

Three Conda environments are provided in `venvs/`:

```bash
conda env create -f venvs/tidyhebo.yaml
conda env create -f venvs/hebo.yaml
conda env create -f venvs/o11.yaml
```

`tidyhebo` is used for tidyHEBO, LogEI, Optuna, Random, and the analysis notebooks. `hebo` is used for the original HEBO implementation. `o11` is the legacy Python 3.7/TensorFlow environment used by the Olympus emulators; the historical name is retained because assembling the legacy emulator stack otherwise requires several old dependencies.

The `o11` export contains the runtime dependencies but not the editable Olympus source checkout. Install Olympus from source into that environment after creating it:

```bash
git clone https://github.com/the-matter-lab/olympus.git ../olympus
conda run -n o11 python -m pip install --no-deps -e ../olympus
conda run -n o11 python -c "from olympus import Emulator; print('Olympus import OK')"
```

The YAML files are exported environments and are intended for Linux. `mamba env create -f ...` can be used instead of `conda env create -f ...` if Mamba is available; the benchmark launcher itself invokes `conda run`, so `conda` must remain available on `PATH`.

## Reproducing the reported results

The notebooks contain the exact commands for regenerating raw optimization traces and rebuild the manuscript-facing summaries from the archived results:

1. **Ackley and Hartmann:** [`01_synthetic_reproducibility.ipynb`](01_synthetic_reproducibility.ipynb)
2. **Olympus, sequential optimization:** [`02_olympus_sequential_reproducibility.ipynb`](02_olympus_sequential_reproducibility.ipynb)
3. **Olympus, adaptive batching:** [`03_olympus_adaptive_batching_reproducibility.ipynb`](03_olympus_adaptive_batching_reproducibility.ipynb)

An additional Needle-in-a-Haystack analysis is documented in [`04_needle_in_a_haystack/ZoMBI_reproducibility.ipynb`](04_needle_in_a_haystack/ZoMBI_reproducibility.ipynb).

For a quick check without rerunning the optimizers, open the notebooks from the repository root in the `tidyhebo` environment and execute all cells. The archived matrices contain 30 runs × 100 evaluations for every manuscript benchmark setting.

## Repository structure

- `01_synthetic_functions/` — archived Ackley and Hartmann traces.
- `02_olympus/` — archived sequential Olympus traces.
- `03_olympus_adaptive_batching/` — archived adaptive-batching traces and per-run batch logs.
- `04_needle_in_a_haystack/` — ZoMBI/Needle-in-a-Haystack analysis.
- `functions_run/` — synthetic benchmark runners.
- `olympus_run/` — Olympus emulator/optimizer orchestration, task configs, adaptors, and tests.
- `reproducibility/metrics.py` — **single source of truth** for median nAUC, IQR nAUC, and CVaR nAUC.
- `reproducibility/plotting.py` and `reproducibility/adaptive_batching.py` — shared plotting/analysis helpers used by the notebooks.
- `venvs/` — Conda environment specifications.

`reproducibility/update_olympus_extrema.py` is a maintenance utility for recomputing task extrema; it is not required for reproducing the archived manuscript results.

## Tests

The Olympus code has environment-specific tests. The runner detects either Mamba or Conda and executes the appropriate marker set in each environment:

```bash
python olympus_run/run_all_tests.py -q
```

The lightweight analysis helpers can also be checked directly with:

```bash
python -m pytest -q reproducibility/tests
```
