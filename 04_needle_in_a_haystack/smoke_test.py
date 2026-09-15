#!/usr/bin/env python3
"""Fast end-to-end smoke test for the ZoMBI reproducibility notebook.

Run from the tidyHEBO environment:
    python smoke_test.py

The test does not require external ZoMBI emulator files and does not launch
expensive benchmark runs. It validates the optimizer interface, then executes
all code cells from ZoMBI_reproducibility.ipynb with figure output redirected
to a temporary directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizers import RandomSearch


def check_optimizer_contract() -> None:
    opt = RandomSearch(dim=5, seed=7)
    values = []
    for _ in range(8):
        x = opt.suggest()
        assert x.shape == (1, 5)
        assert np.all((0.0 <= x) & (x <= 1.0))
        y = float(np.square(x - 0.2).sum())
        opt.observe(x, y)
        values.append(y)

    assert len(values) == 8
    assert np.isfinite(values).all()


def execute_notebook() -> dict:
    notebook_path = ROOT / "ZoMBI_reproducibility.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    old_cwd = Path.cwd()
    old_backend = os.environ.get("MPLBACKEND")

    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(ROOT)

    namespace = {"__name__": "__main__"}
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(old_cwd)
        if old_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = old_backend
    return namespace


def main() -> None:
    check_optimizer_contract()
    namespace = execute_notebook()

    fig = namespace.get("manuscript_fig")
    axes = namespace.get("manuscript_axes")
    legend = namespace.get("manuscript_legend")

    assert fig is not None, "Notebook did not create manuscript_fig"
    assert axes is not None and len(axes) == 2, "Notebook did not create two panel axes"
    assert legend is not None, "Notebook did not create the shared legend"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
