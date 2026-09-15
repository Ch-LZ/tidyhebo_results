from pathlib import Path

import numpy as np

from reproducibility.metrics import load_ragged_batch_sizes


ROOT = Path(__file__).resolve().parents[2]


def test_adaptive_batch_logs_match_archived_runs():
    sequential_tasks = sorted(
        path.name for path in (ROOT / "02_olympus").iterdir() if path.is_dir()
    )

    for cap in (2, 4, 8, 16):
        for task in sequential_tasks:
            setting = ROOT / "03_olympus_adaptive_batching" / f"adaptive_bs_{cap}" / task
            values = np.atleast_2d(np.loadtxt(setting / "tidyhebo_values.txt", dtype=float))
            batch_rows = load_ragged_batch_sizes(setting / "tidyhebo_n_from_opt.txt")

            assert values.shape == (30, 100), f"Unexpected result shape in {setting}"
            assert len(batch_rows) == values.shape[0], f"Run/log mismatch in {setting}"
            assert all(sum(row) == values.shape[1] for row in batch_rows), \
                f"Batch sizes do not sum to the evaluation budget in {setting}"
