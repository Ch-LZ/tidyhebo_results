#!/usr/bin/env python3
"""
Run the Olympus benchmark and optimizer in separate Conda environments.
Supports repeated runs and loguru-based logging.
"""

import os
import subprocess
import tempfile
import sys
import argparse
import threading
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from src.tasks import OLYMPUS_TASKS
from src.optimizer_adaptors.model_lib import Model


_OPT_TO_ENV = {
    'optuna': 'tidyhebo',
    'logei': 'tidyhebo',
    'tidyhebo': 'tidyhebo',
    'hebo': 'hebo',
    'random': 'tidyhebo'
}


def setup_logging(log_level: str = "DEBUG", log_file: Optional[Path] = None):
    """Configure loguru output."""
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=log_level,
            rotation="10 MB",
            retention="10 days"
        )
        logger.debug(f"Logs are also written to {log_file}")


def main(
    optimizer: str,
    dataset: str,
    num_iterations: int = 100,
    num_runs: int = 1,
    init_size: int = 1,
    batch_size: int = 1,
    adaptive_ceil: Optional[int] = None,
    output_dir: str = "02_olympus",
    seed: Optional[int] = 42,
    log_level: str = "DEBUG",
    log_file: Optional[str] = None,
) -> None:
    setup_logging(log_level, Path(log_file) if log_file else None)

    model_env = _OPT_TO_ENV.get(optimizer, 'tidyhebo')
    benchmark_env = 'o11'

    logger.info(f"Starting: dataset={dataset}, optimizer={optimizer}, "
                f"iterations={num_iterations}, runs={num_runs}, env_model={model_env}")


    if seed is not None:
        import numpy as np
        rng = np.random.RandomState(seed)
        seeds = [rng.randint(0, 2**31 - 1) for _ in range(num_runs)]
    else:
        seeds = [None] * num_runs

    script_dir = Path(__file__).parent
    run_benchmark_file = script_dir / "src" / "run_benchmark.py"
    run_optimization_file = script_dir / "src" / "run_optimization.py"

    if not run_benchmark_file.exists():
        logger.error(f"run_benchmark.py not found at {run_benchmark_file}")
        sys.exit(1)
    if not run_optimization_file.exists():
        logger.error(f"run_optimization.py not found at {run_optimization_file}")
        sys.exit(1)

    for run_idx in range(num_runs):
        current_seed = seeds[run_idx] if seeds[run_idx] is not None else None
        logger.debug(f"Run #{run_idx+1} (seed={current_seed})")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe_send = os.path.join(tmpdir, "pipe_send")
            pipe_recv = os.path.join(tmpdir, "pipe_recv")
            os.mkfifo(pipe_send)
            os.mkfifo(pipe_recv)

            inp_file = os.path.join(tmpdir, "input.txt")
            out_file = os.path.join(tmpdir, "output.txt")

            cmd_benchmark = [
                "conda", "run", "-n", benchmark_env,
                "python", str(run_benchmark_file),
                "--from-M-to-B", pipe_send,
                "--from-B-to-M", pipe_recv,
                "--dataset", dataset,
                "--num-iterations", str(num_iterations),
                "--inp-file", inp_file,
                "--init-size", str(init_size),
                "--output-file", out_file,
            ]

            cmd_model = [
                "conda", "run", "-n", model_env,
                "python", str(run_optimization_file),
                "--from-M-to-B", pipe_send,
                "--from-B-to-M", pipe_recv,
                "--dataset", dataset,
                "--num-iterations", str(num_iterations),
                "--inp-file", inp_file,
                "--init-size", str(init_size),
                "--batch-size", str(batch_size),
                "--adaptive-ceil", str(adaptive_ceil),
                "--output-file", out_file,
                "--model-type", optimizer,
                "--run-idx", str(run_idx),
                "--output-dir", output_dir,
            ]
            if current_seed is not None:
                cmd_benchmark.append("--seed")
                cmd_benchmark.append(str(current_seed))
                cmd_model.append("--seed")
                cmd_model.append(str(current_seed))

            logger.debug(f"Benchmark command: {' '.join(cmd_benchmark)}")
            logger.debug(f"Model command: {' '.join(cmd_model)}")


            proc_benchmark = subprocess.Popen(
                cmd_benchmark,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            proc_model = subprocess.Popen(
                cmd_model,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            def log_output(proc, name):
                for line in proc.stdout:
                    logger.debug(f"[{name}] {line.strip()}")

            t1 = threading.Thread(target=log_output, args=(proc_benchmark, "benchmark"))
            t2 = threading.Thread(target=log_output, args=(proc_model, "model"))
            t1.start()
            t2.start()

            proc_benchmark.wait()
            proc_model.wait()
            t1.join()
            t2.join()

            if proc_benchmark.returncode != 0:
                logger.error(f"Benchmark failed (exit code {proc_benchmark.returncode})")

                sys.exit(1)
            if proc_model.returncode != 0:
                logger.error(f"Model failed (exit code {proc_model.returncode})")
                sys.exit(1)

        logger.info(f"Run #{run_idx+1} completed.")

    logger.success("All runs completed.")


if __name__ == '__main__':

    def int_or_none(value):
            if str(value).lower() in ('none', 'null', ''):
                return None
            return int(value)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=OLYMPUS_TASKS)
    parser.add_argument('--optimizer', required=True, choices=Model.available_models())
    parser.add_argument('--num-iterations', '-ni', type=int, default=100)
    parser.add_argument('--num-runs', '-nr', type=int, default=1)
    parser.add_argument('--init-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--adaptive-ceil', type=int_or_none, default=None)
    parser.add_argument('--output-dir', type=str, default="02_olympus")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-level', type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--log-file', type=str, default=None, help="Optional log file path")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.batch_size > 1:
        output_dir = Path(args.output_dir) / f"bs_{args.batch_size}"
    if args.adaptive_ceil is not None:
        output_dir = Path(args.output_dir) / f"adaptive_bs_{args.adaptive_ceil}"

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main(
        optimizer=args.optimizer,
        dataset=args.dataset,
        num_iterations=args.num_iterations,
        num_runs=args.num_runs,
        init_size=args.init_size,
        batch_size=args.batch_size,
        adaptive_ceil=args.adaptive_ceil,
        output_dir=str(output_dir),
        seed=args.seed,
        log_level=args.log_level,
        log_file=args.log_file,
    )
