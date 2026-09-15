import numpy as np


def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=float)


class RandomSearch:
    def __init__(self, dim, seed):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def suggest(self):
        return self.rng.random((1, self.dim))

    def observe(self, x, y):
        pass


class TidyHEBO:
    """Minimal adapter for the current public tidyHEBO API."""
    def __init__(self, dim, seed, n_init=4):
        import torch
        from tidyhebo import RealP, tidyHEBO

        torch.manual_seed(seed)
        params = [RealP(i, 0.0, 1.0, name=f"x{i}") for i in range(dim)]
        self.opt = tidyHEBO(parameters=params, n_init=n_init)

    def suggest(self):
        return _as_numpy(self.opt.suggest()).reshape(1, -1)

    def observe(self, x, y):
        # tidyHEBO maximizes internally; ZoMBI benchmark objectives are stored
        # in the minimization convention, hence the sign flip.
        import pandas as pd
        cols = [p.name for p in self.opt.design_space.params]
        self.opt.observe(pd.DataFrame(x, columns=cols), np.asarray([-y], dtype=float))


class OptunaTPE:
    def __init__(self, dim, seed):
        import optuna
        self.dim = dim
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self.trial = None

    def suggest(self):
        self.trial = self.study.ask()
        return np.array([[self.trial.suggest_float(f"x{i}", 0.0, 1.0)
                          for i in range(self.dim)]])

    def observe(self, x, y):
        self.study.tell(self.trial, float(y))


class HEBO:
    def __init__(self, dim, seed):
        from hebo.design_space.design_space import DesignSpace
        from hebo.optimizers.hebo import HEBO as _HEBO
        np.random.seed(seed)
        space = DesignSpace().parse([
            {"name": f"x{i}", "type": "num", "lb": 0.0, "ub": 1.0}
            for i in range(dim)
        ])
        self.opt = _HEBO(space)
        self.cand = None

    def suggest(self):
        self.cand = self.opt.suggest(n_suggestions=1)
        return self.cand.values.astype(float)

    def observe(self, x, y):
        self.opt.observe(self.cand, np.asarray([y], dtype=float))


def make_optimizer(name, dim, seed, n_init=4):
    if name == "random":
        return RandomSearch(dim, seed)
    if name == "tidyhebo":
        return TidyHEBO(dim, seed, n_init=n_init)
    if name == "optuna":
        return OptunaTPE(dim, seed)
    if name == "hebo":
        return HEBO(dim, seed)
    raise ValueError(f"Unknown optimizer: {name}")
