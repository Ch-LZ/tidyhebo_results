import argparse
import numpy as np
from tidyhebo import RealP
from tqdm import tqdm
from botorch.test_functions.synthetic import Ackley, Hartmann
import torch

import rootutils
rootutils.setup_root(".", pythonpath=True)
from olympus_run.src.reference_optimizer import LogEIMixedOptimizer


def test_optimizer(n_total, n_init, problem):
    # dfeine a parameter space
    params = [
        RealP(i, lb, ub) for (i, (lb, ub)) in enumerate(problem.bounds.T.detach().numpy())
    ]

    # initialize optimizer
    optimizer = LogEIMixedOptimizer(parameters=params, n_init=n_init)

    for _ in tqdm(range(n_total), total=n_total):
        x_new = torch.from_numpy(optimizer.suggest().values.copy())
        y_new = problem(x_new)
        optimizer.observe(x_new, y_new)
    return optimizer.train_y.detach().numpy()

problems = {
    'ackley': Ackley,
    'hartmann': Hartmann
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

    problem = problems[args.problem](negate=True, dim=6)  #* must be a maximization problem
    output_file = args.output_file or Path("01_synthetic_functions") / args.problem / "logei_values.txt"

    n_repeats = args.n_repeats
    n_trials = 100

    for i in range(n_repeats):
        print()
        print(f"iter: {i} time: {datetime.now()}")
        current_res = test_optimizer(n_total=n_trials, n_init=4, problem=problem).T
        with open(output_file, "a") as f:
            np.savetxt(f, current_res)
