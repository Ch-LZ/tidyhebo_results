import argparse
import numpy as np
import torch
from tqdm import tqdm
from botorch.test_functions.synthetic import Ackley, Hartmann
from tidyhebo import RealP
from tidyhebo.design_space import MixedDesignSpace

# from olympus_run.optimizers_reference import RealParameter, RealDesignSpace


def test_random(n_total, problem):
    """Run independent random search in domain space and return best-so-far."""
    params = [
        RealP(i, lb, ub)
        for (i, (lb, ub)) in enumerate(problem.bounds.T.detach().numpy())
    ]
    ds = MixedDesignSpace(params)

    y_best_so_far = []
    best = -np.inf
    for _ in tqdm(range(n_total), total=n_total):
        # Draw one random point.
        x_new = torch.rand(1, len(params)).double()
        x_new = ds.from_GP_to_DS(x_new, return_df=False)
        y_new = problem(x_new).item()
        if y_new > best:
            best = y_new
        y_best_so_far.append(best)

    return np.asarray(y_best_so_far)


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
    output_file = args.output_file or Path("01_synthetic_functions") / args.problem / "random_values.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_repeats = args.n_repeats
    n_trials = 100

    for i in range(n_repeats):
        print()
        print(f"iter: {i} time: {datetime.now()}")
        current_res = test_random(n_total=n_trials, problem=problem)
        with open(output_file, "a") as f:
            np.savetxt(f, current_res.reshape(1, -1))
