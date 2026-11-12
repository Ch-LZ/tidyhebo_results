"""Batch/native-state adaptation of the legacy HEBO and Optuna wrappers.

No BO loop here: the existing controller owns feedback boundaries and budgets.
All three external baselines share the audited legacy Sobol initial design.
"""
import copy
import numpy as np
import pandas as pd
import torch
from bo_benchmark.workers.modern_adapters import RandomAdapter, rng_state, restore_rng, seed_all


class BaselineAdapter:
    def __init__(self, task, seed, options, kind):
        self.kind = kind
        self.columns = [p['name'] for p in task['parameters']]
        self.bounds = np.array([[p['lower'], p['upper']] for p in task['parameters']])
        self.n_init = options.get('n_init', 4)
        self.issued = 0
        self.pending = []
        seed_all(seed)
        unit = torch.quasirandom.SobolEngine(len(self.columns), scramble=True, seed=seed).draw(
            self.n_init, dtype=torch.double).numpy()
        self.initial = self.bounds[:, 0] + unit * np.diff(self.bounds, axis=1).ravel()
        self.metadata = {'baseline': kind, 'startup': 'legacy_qmace', 'n_init': self.n_init}
        if kind == 'hebo':
            from hebo.design_space.design_space import DesignSpace
            from hebo.optimizers.hebo import HEBO
            space = DesignSpace().parse([{'name': p['name'], 'type': 'num',
                'lb': p['lower'], 'ub': p['upper']} for p in task['parameters']])
            self.core = HEBO(space, rand_sample=self.n_init, scramble_seed=seed)
        elif kind == 'optuna':
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.core = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=self.n_init, multivariate=False, constant_liar=True))
            self.metadata.update(sampler='TPE', constant_liar=True, multivariate=False)
        elif kind == 'random':
            self.core = RandomAdapter(task, seed, options)
        else:
            raise ValueError('Unknown external baseline: ' + kind)

    def suggest(self, n=1):
        if self.pending:
            raise ValueError('Observe the complete pending batch before suggesting again')
        remaining = self.n_init - self.issued
        if remaining > 0:
            if n > remaining:
                raise ValueError('Request crosses initial-design boundary')
            frame = pd.DataFrame(self.initial[self.issued:self.issued+n], columns=self.columns)
            self.issued += n
            self.pending = [None] * n
            return frame
        if self.kind == 'hebo':
            frame = self.core.suggest(n_suggestions=n)[self.columns]
            frame = self._bounded_hebo_frame(frame)
            self.pending = [None] * n
        elif self.kind == 'optuna':
            # Sample each complete trial before asking the next one: constant-liar
            # must see the preceding RUNNING trials' parameter values.
            self.pending = []
            rows = []
            for _ in range(n):
                trial = self.core.ask()
                rows.append([trial.suggest_float(name, float(lo), float(hi))
                             for name, (lo, hi) in zip(self.columns, self.bounds)])
                self.pending.append(trial.number)
            frame = pd.DataFrame(rows, columns=self.columns)
        else:
            frame = self.core.suggest(n)
            self.pending = [None] * n
        return frame.reset_index(drop=True)

    def _bounded_hebo_frame(self, frame):
        """Repair only float32-scale boundary roundoff from HEBO's inverse transform."""
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("HEBO returned nonfinite candidates")
        lower, upper = self.bounds.T
        tolerance = 8 * np.finfo(np.float32).eps * np.maximum(
            np.maximum(np.abs(lower), np.abs(upper)), upper - lower)
        if ((values < lower - tolerance) | (values > upper + tolerance)).any():
            raise ValueError("HEBO returned candidates substantially outside task bounds")
        clipped = np.clip(values, lower, upper)
        if not np.array_equal(values, clipped):
            import warnings
            warnings.warn("Clipped HEBO boundary roundoff to task bounds", RuntimeWarning,
                          stacklevel=2)
        return pd.DataFrame(clipped, columns=self.columns, index=frame.index)

    def observe(self, X, y):
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(values) != len(self.pending) or len(X) != len(values):
            raise ValueError('Observation must match the complete pending batch')
        # The audited controller supplies maximization-oriented y to every adapter.
        if self.kind == 'hebo':
            self.core.observe(X[self.columns], -values.reshape(-1, 1))
        elif self.kind == 'optuna':
            import optuna
            distributions = {name: optuna.distributions.FloatDistribution(float(lo), float(hi))
                             for name, (lo, hi) in zip(self.columns, self.bounds)}
            for (_, row), value, number in zip(X.iterrows(), values, self.pending):
                if number is None:
                    self.core.add_trial(optuna.trial.create_trial(
                        params={name: float(row[name]) for name in self.columns},
                        distributions=distributions, value=float(value)))
                else:
                    self.core.tell(number, float(value))
        else:
            self.core.observe(X, y)
        self.pending = []

    def save_state(self):
        return copy.deepcopy({'core': self.core, 'issued': self.issued,
                              'pending': self.pending, 'rng': rng_state()})

    def load_state(self, state):
        self.core = state['core']
        self.issued, self.pending = state['issued'], state['pending']
        restore_rng(state['rng'])
