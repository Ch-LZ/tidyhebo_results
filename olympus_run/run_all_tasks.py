#!/usr/bin/env python3
"""
Run optimization for every task in OLYMPUS_TASKS.
Uses the inference.sh helper.

Usage:
    python run_all_tasks.py --optimizers optuna,random --batch-size 4
    python run_all_tasks.py --optimizers tidyhebo --adaptive-ceil 3
"""

import argparse
import subprocess
import sys
from pathlib import Path


project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tasks import OLYMPUS_TASKS


def main():
    parser = argparse.ArgumentParser(description="Run optimization for all Olympus tasks.")
    parser.add_argument('--optimizers', type=str, default="tidyhebo",
                        help="Comma-separated optimizers (for example: optuna,random)")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Optimization batch size")
    parser.add_argument('--init-size', type=int, default=1,
                        help="Initial batch size (first iteration)")
    parser.add_argument('--adaptive-ceil', type=int, default=None,
                        help="Adaptive batch cap (overrides batch-size when set)")

    args = parser.parse_args()

    optimizers = args.optimizers
    batch_size = args.batch_size
    init_size = args.init_size
    adaptive_ceil = args.adaptive_ceil


    script_path = Path(__file__).parent / "inference.sh"
    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        sys.exit(1)

    print(f"Tasks: {OLYMPUS_TASKS}")
    print(f"Optimizers: {optimizers}")
    print(f"Batch size: {batch_size}")
    print(f"Initial size: {init_size}")
    print(f"Adaptive cap: {adaptive_ceil if adaptive_ceil is not None else 'not set'}")
    print("----------------------------------------")

    total = len(OLYMPUS_TASKS)
    for idx, task in enumerate(OLYMPUS_TASKS, 1):
        print(f"\n[{idx}/{total}] Running task: {task}")

        cmd = [str(script_path), optimizers, task, str(batch_size), str(init_size)]

        if adaptive_ceil is not None:
            cmd.append(str(adaptive_ceil))


        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Task {task} failed (exit code {result.returncode})")
            sys.exit(1)
        print(f"Task {task} completed.")

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
