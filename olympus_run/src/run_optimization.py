
import sys
from pathlib import Path

# Add the project root to sys.path.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import argparse
import numpy as np
import os
from typing import Optional

from optimizer_adaptors.model_lib import Model
from src.tasks import OLYMPUS_TASKS


def batch_sizes(total, n_init, n_batch):
    remaining = total
    if remaining > 0:
        first = min(n_init, remaining)
        yield first
        remaining -= first
    while remaining > 0:
        chunk = min(n_batch, remaining)
        yield chunk
        remaining -= chunk


def find_dataset_config(
    dataset: str,
    base_path: Optional[Path] = Path(__file__).parent.parent / "configs" / "tasks",
    subdirs: list = ["olympus 2021", "olympus 2023"]
) -> Path:
    """Find the configuration file for a task."""
    for sub in subdirs:
        candidate = base_path / sub / f"{dataset}.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{dataset}.yaml was not found in {subdirs} below {base_path}"
    )


def append_history_to_file(optimizer: Model, output_dir: Path, task_name: str, run_idx: int) -> None:
    """Append one optimization history row to the result file."""
    history = optimizer.get_history()  # shape (num_iterations,)
    opt_name = optimizer._model_name
    out_fpath = output_dir / task_name / f"{opt_name}_values.txt"
    out_fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(out_fpath, "a") as f: 
        np.savetxt(f, history)  # np.savetxt appends the line break.

    # *Note: monkeypatching to support only tidyhebo for recording batch logging
    if getattr(optimizer, "n_from_pareto_front", []):
        n_from_pareto_front = getattr(optimizer, "n_from_pareto_front")
        n_from_pareto_front = np.asarray(n_from_pareto_front).reshape(1, -1)
        out_fpath = output_dir / task_name / f"{opt_name}_n_from_opt.txt"
        with open(out_fpath, "a") as f:
            np.savetxt(f, n_from_pareto_front)



def main(
    from_M_to_B: str,
    from_B_to_M: str,
    dataset: str,
    inp_file: str,
    output_file: str,
    num_iterations: int,
    model_type: str,
    run_idx: int = 0,
    init_size: int = 1,
    batch_size: int = 1,
    adaptive_ceil: Optional[int] = None,
    output_dir: str = "02_olympus",
    seed: Optional[int] = None,
) -> None:
    """
    Run the optimizer while displaying progress with tqdm.
    """
    print("Python in Model env: ", sys.version)
    venv_name = Path(os.environ.get('CONDA_PREFIX')).name
    print(f"optimization is running in: {venv_name} with {sys.executable}")

    if seed is not None:
        np.random.seed(seed)

    config_path = find_dataset_config(dataset).resolve()
    optimizer: Model = Model.from_config(model_type, config_path)

    iters_remained = num_iterations
    while iters_remained > 0:
        current_ceil = min(adaptive_ceil, iters_remained) if adaptive_ceil is not None else None
        candidates = optimizer.suggest(batch_size=batch_size, adaptive_ceil=current_ceil)  # The four-point initialization is compatible with the current budget logic.

        np.savetxt(inp_file, candidates)

        with open(from_M_to_B, "w") as f:
            f.write("1")
            f.flush()

        with open(from_B_to_M, "r") as f:
            f.read()  # Wait until the benchmark writes the result.

        rv = np.loadtxt(output_file)
        optimizer.observe(candidates, rv.reshape(-1, 1))

        iters_remained -= candidates.shape[0]

    # Save optimization history.
    output_path = (Path(__file__).parent.parent.parent / output_dir).resolve()
    append_history_to_file(optimizer, output_path, dataset, run_idx)

    print(f"Run {run_idx} completed.")


if __name__ == "__main__":

    def int_or_none(value):
        if str(value).lower() in ('none', 'null', ''):
            return None
        return int(value)

    parser = argparse.ArgumentParser()
    parser.add_argument('--from-M-to-B', required=True)
    parser.add_argument('--from-B-to-M', required=True)
    parser.add_argument('--dataset', required=True, choices=OLYMPUS_TASKS)
    parser.add_argument('--num-iterations', type=int, required=True)
    parser.add_argument('--inp-file', required=True)
    parser.add_argument('--init-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--adaptive-ceil', type=int_or_none, default=None)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--model-type', required=True, choices=Model.available_models())
    parser.add_argument('--run-idx', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default="02_olympus")
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    main(
        from_M_to_B=args.from_M_to_B,
        from_B_to_M=args.from_B_to_M,
        dataset=args.dataset,
        inp_file=args.inp_file,
        init_size=args.init_size,
        batch_size=args.batch_size,
        adaptive_ceil=args.adaptive_ceil,
        output_file=args.output_file,
        num_iterations=args.num_iterations,
        model_type=args.model_type,
        run_idx=args.run_idx,
        output_dir=args.output_dir,
        seed=args.seed,
    )