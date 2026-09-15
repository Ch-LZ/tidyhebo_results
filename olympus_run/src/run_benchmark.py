
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

from olympus import Emulator
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


def main(
    from_M_to_B: str,
    from_B_to_M: str,
    dataset: str,
    inp_file: str,
    init_size: int,
    output_file: str,
    num_iterations: int,
    seed: Optional[int] = None,
) -> None:
    """
    Run the Olympus emulator benchmark.
    """
    print("Python in Olympus benchmark env: ", sys.version)
    venv_name = Path(os.environ.get('CONDA_PREFIX')).name
    print(f"benchmark is running in: {venv_name} with {sys.executable}")

    if seed is not None:
        np.random.seed(seed)

    em = Emulator(dataset, "BayesNeuralNet")

    iter_remained = num_iterations
    while iter_remained > 0:
        with open(from_M_to_B, "r") as f:
            f.read()

        candidates = np.loadtxt(inp_file, ndmin=2)
        # based on `olympus.Emulator.run`` code
        y_pred_objects, y_pred_std_ep, y_pred_std_al = em.run(candidates)
        rv = y_pred_objects

        rv = np.asarray(rv)

        np.savetxt(output_file, rv)

        # Signal the model process that benchmark data are ready.
        with open(from_B_to_M, "w") as f:
            f.write("1")
            f.flush()

        iter_remained -= candidates.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-M-to-B', required=True)
    parser.add_argument('--from-B-to-M', required=True)
    parser.add_argument('--dataset', required=True, choices=OLYMPUS_TASKS)
    parser.add_argument('--num-iterations', type=int, required=True)
    parser.add_argument('--inp-file', required=True)
    parser.add_argument('--init-size', type=int, default=1)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    main(
        from_M_to_B=args.from_M_to_B,
        from_B_to_M=args.from_B_to_M,
        dataset=args.dataset,
        inp_file=args.inp_file,
        init_size=args.init_size,
        output_file=args.output_file,
        num_iterations=args.num_iterations,
        seed=args.seed,
    )