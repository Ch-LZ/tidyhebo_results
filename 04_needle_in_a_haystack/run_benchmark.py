#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.oracle import PickleOracle
from src.optimizers import make_optimizer
from src.tasks import TASKS


def resolve_oracle(task_name, oracle, zombi_root):
    if oracle:
        return Path(oracle)
    task = TASKS[task_name]
    if zombi_root and task.default_oracle_relpath:
        return Path(zombi_root) / task.default_oracle_relpath
    raise ValueError(
        "Provide --oracle FILE, or --zombi-root DIR for a task with a known "
        "ZoMBI data path."
    )


def run_once(oracle, optimizer_name, evaluations, seed, n_init):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    opt = make_optimizer(optimizer_name, oracle.dim, seed, n_init=n_init)
    values = []
    for _ in range(evaluations):
        x = opt.suggest()
        y = float(oracle(x)[0])
        values.append(y)
        opt.observe(x, y)
    return np.asarray(values, dtype=float)


def main():
    p = argparse.ArgumentParser(description="Run a ZoMBI emulator benchmark.")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--optimizer", choices=["tidyhebo", "hebo", "optuna", "random"], required=True)
    p.add_argument("--oracle", help="Path to a fitted sklearn-compatible oracle (.pkl).")
    p.add_argument("--zombi-root", help="Root of the original ZoMBI repository/data tree.")
    p.add_argument("--evaluations", type=int, default=100)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--n-init", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    oracle_path = resolve_oracle(args.task, args.oracle, args.zombi_root)
    oracle = PickleOracle(oracle_path)
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31 - 1, size=args.runs, dtype=np.int64)

    rows = [run_once(oracle, args.optimizer, args.evaluations, int(s), args.n_init)
            for s in seeds]
    values = np.vstack(rows)

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.optimizer}_values.txt"
    np.savetxt(out_file, values)
    meta = {
        "task": args.task,
        "optimizer": args.optimizer,
        "oracle": str(oracle_path.resolve()),
        "oracle_dim": oracle.dim,
        "evaluations": args.evaluations,
        "runs": args.runs,
        "master_seed": args.seed,
        "run_seeds": [int(x) for x in seeds],
    }
    (out_dir / f"{args.optimizer}_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {out_file}  shape={values.shape}")


if __name__ == "__main__":
    main()
