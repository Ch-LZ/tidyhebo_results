"""Maintenance utility for recomputing Olympus task objective extrema.

This script is not part of the normal manuscript reproduction workflow. It is
kept under ``reproducibility/`` because the task configurations store the
reference best/worst values used to normalize regret metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import yaml
from olympus import Emulator
from scipy.optimize import differential_evolution
from tqdm import tqdm

from olympus_run.src.optimizer_adaptors.simplex_transformer import SimplexTransformer

warnings.filterwarnings("ignore", category=UserWarning)


def find_global_extrema(dataset_name, bounds, on_simplex, max_evals=2000, seed=42):
    """Estimate global minimum and maximum values for one Olympus emulator."""
    emulator = Emulator(dataset_name, "BayesNeuralNet")
    dimension = len(bounds)
    transformer = SimplexTransformer(K=dimension + 1) if on_simplex else None

    def objective(x):
        emulator_x = transformer.transform(x) if transformer is not None else x
        result = emulator.run([emulator_x])
        return float(result[0])

    result_min = differential_evolution(
        objective,
        bounds,
        maxiter=max_evals,
        popsize=15,
        seed=seed,
        disp=False,
        updating="deferred",
        workers=1,
    )
    result_max = differential_evolution(
        lambda x: -objective(x),
        bounds,
        maxiter=max_evals,
        popsize=15,
        seed=seed,
        disp=False,
        updating="deferred",
        workers=1,
    )
    return result_min.fun, result_min.x, -result_max.fun, result_max.x


def process_config(file_path: Path, max_evals=2000, seed=42):
    """Recompute and write ``y_p_star`` and ``y_p_w`` for one task config."""
    with file_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dataset_name = file_path.stem
    print(f"Processing {dataset_name}...")
    params = config.get("params")
    if params is None:
        raise ValueError(f"{file_path} does not contain a 'params' section")

    on_simplex = bool(config.get("on_simplex", False))
    less_is_better = bool(config.get("less_is_better", False))
    optimization_params = params[:-1] if on_simplex else params
    if on_simplex and len(params) < 2:
        raise ValueError("A simplex task must contain at least two parameters")
    if any(param.get("type") != "continuous" for param in optimization_params):
        raise ValueError("Only continuous parameters are supported by this utility")

    bounds = [(param["low"], param["high"]) for param in optimization_params]
    y_min, _, y_max, _ = find_global_extrema(
        dataset_name,
        bounds,
        on_simplex,
        max_evals=max_evals,
        seed=seed,
    )
    y_best, y_worst = (y_min, y_max) if less_is_better else (y_max, y_min)
    config["y_p_star"] = float(y_best)
    config["y_p_w"] = float(y_worst)

    with file_path.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  wrote y_p_star={y_best:.6f}, y_p_w={y_worst:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=Path("olympus_run/configs/tasks"),
    )
    parser.add_argument("--max-evals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.configs_root.exists():
        raise FileNotFoundError(f"Configuration directory not found: {args.configs_root}")
    yaml_files = sorted(args.configs_root.rglob("*.yaml")) + sorted(
        args.configs_root.rglob("*.yml")
    )
    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found below {args.configs_root}")

    for file_path in tqdm(yaml_files, desc="Updating task extrema"):
        process_config(file_path, max_evals=args.max_evals, seed=args.seed)


if __name__ == "__main__":
    main()
