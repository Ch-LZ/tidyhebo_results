import argparse
import numpy as np
import torch
from tqdm import tqdm
from botorch.test_functions.synthetic import Ackley, Hartmann
from tidyhebo import RealP
from tidyhebo.design_space import MixedDesignSpace

import optuna

#* Optuna
class OptunaOptimizer:
    def __init__(self, parameters, train_x=None, train_y=None):
        self.parameters = parameters
        self.study = optuna.create_study(direction="maximize")
        self._prev_trial = None
        self.train_y = train_y

    def suggest(self) -> torch.Tensor:
        trial = self.study.ask()
        self._prev_trial = trial
        cand = []
        for i, p in enumerate(self.parameters):
            x = trial.suggest_float(f"{i}", p.ds_lb, p.ds_ub)
            cand.append(x)
        return torch.tensor(cand, dtype=torch.float).reshape(1, -1)


    def observe(self, x, y) -> None:
        self.train_y = torch.cat([self.train_y, y], axis=0)
        y = y.reshape(-1).item()
        self.study.tell(self._prev_trial, y)
        self._prev_trial = None


def test_optuna(n_total, n_init, problem):
    """Run Optuna after ``n_init`` random points and return best-so-far."""
    params = [
        RealP(i, lb, ub)
        for (i, (lb, ub)) in enumerate(problem.bounds.T.detach().numpy())
    ]
    ds = MixedDesignSpace(params)

    # initial design
    x0 = torch.rand(n_init, len(params)).double()
    x0 = ds.from_GP_to_DS(x0, return_df=False)
    y0 = problem(x0)

    optimizer = OptunaOptimizer(
        parameters=params,
        train_x=x0,
        train_y=y0,
    )

    for _ in tqdm(range(n_total - n_init), total=n_total - n_init):
        x_new = optimizer.suggest()
        y_new = problem(x_new)
        optimizer.observe(x_new, y_new)

    y_all = optimizer.train_y
    if hasattr(y_all, "detach"):
        y_all = y_all.detach().cpu().numpy()
    y_all = np.asarray(y_all).ravel()
    return np.maximum.accumulate(y_all)


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
    output_file = args.output_file or Path("01_synthetic_functions") / args.problem / "optuna_values.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_repeats = args.n_repeats
    n_trials = 100
    n_init = 8

    for i in range(n_repeats):
        print()
        print(f"iter: {i} time: {datetime.now()}")
        current_res = test_optuna(n_total=n_trials, n_init=n_init, problem=problem)
        with open(output_file, "a") as f:
            np.savetxt(f, current_res.reshape(1, -1))
