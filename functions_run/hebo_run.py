# =============================================================================
# *Note: optimizer HEBO requires different venv
# =============================================================================

import argparse
import numpy as np
import torch
from tqdm import tqdm
from botorch.test_functions.synthetic import Ackley, Hartmann

from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace


def test_hebo(n_total, n_init, problem):
    """Run HEBO optimization and return the best-so-far trajectory."""
    # HEBO minimizes, so observe receives -y.
    # The benchmark already uses negate=True, i.e. a maximization objective.
    lbs = problem.bounds[0, :].detach().numpy()
    ubs = problem.bounds[1, :].detach().numpy()
    dim = len(lbs)
    params = [
        {'name': f'x{i}', 'type': 'num', 'lb': lb, 'ub': ub}
        for i, lb, ub in zip(range(dim), lbs, ubs)
    ]
    hebo_design_space = DesignSpace().parse(params)

    optimizer = HEBO(hebo_design_space, rand_sample=n_init)

    best_results = []
    for _ in tqdm(range(n_total), total=n_total):
        cand = optimizer.suggest(n_suggestions=1)
        x = torch.from_numpy(cand.iloc[-1].to_numpy())
        y = problem(x)
        # observe expects a minimized objective, so invert the sign.
        optimizer.observe(cand, -1.0 * y.detach().numpy().reshape(-1))

        if best_results:
            best_results.append(max(best_results[-1], y.item()))
        else:
            best_results.append(y.item())

    return np.asarray(best_results)


problems = {
    'ackley': Ackley,
    'hartmann': Hartmann,
    # others - unsupported
}


if __name__ == '__main__':
    from datetime import datetime
    from pathlib import Path

    def str_or_none(value):
        if str(value).lower() in ('none', 'null', ''):
            return None
        return str(value)

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='ackley')
    parser.add_argument('--n-repeats', type=int, default=30)
    parser.add_argument('--output-file', type=str_or_none, default=None)
    args = parser.parse_args()

    problem = problems[args.problem](negate=True, dim=6)  # must be a maximization problem
    output_file = args.output_file or Path("01_synthetic_functions") / args.problem / "hebo_values.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_repeats = args.n_repeats
    n_trials = 100
    n_init = 4  # Matches the original notebook: rand_sample=init_n.

    for i in range(n_repeats):
        print()
        print(f"iter: {i} time: {datetime.now()}")
        current_res = test_hebo(n_total=n_trials, n_init=n_init, problem=problem)
        with open(output_file, "a") as f:
            np.savetxt(f, current_res.reshape(1, -1))