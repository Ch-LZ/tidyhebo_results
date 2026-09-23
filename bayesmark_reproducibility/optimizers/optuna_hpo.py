from collections import defaultdict, deque
from math import exp, log, log1p

import numpy as np
import optuna

from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


class OptunaOptimizer(AbstractOptimizer):
    """Optuna TPE wrapper for Bayesmark."""

    primary_import = "optuna"

    def __init__(self, api_config, n_startup_trials=8, random=np_util.random):
        """
        Parameters
        ----------
        api_config : dict-like of dict-like
            Bayesmark search-space configuration.
        n_startup_trials : int, default=8
            Number of random trials before TPE starts, matching the n_init=8
            used in the tidyHEBO wrapper.
        random : numpy.random.RandomState
            Bayesmark-controlled random stream used to seed Optuna.
        """
        super().__init__(api_config)

        self.random = random
        self.param_names = list(api_config.keys())

        sampler = optuna.samplers.TPESampler(
            seed=int(np_util.random_seed(random)),
            n_startup_trials=n_startup_trials,
        )
        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        # A point can be suggested more than once, therefore each key stores a
        # queue rather than a single trial.
        self._pending = defaultdict(deque)

    @staticmethod
    def _logit(x):
        return log(x / (1.0 - x))

    @staticmethod
    def _expit(x):
        if x >= 0:
            z = exp(-x)
            return 1.0 / (1.0 + z)
        z = exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _bilog(x):
        return np.sign(x) * log1p(abs(x))

    @staticmethod
    def _biexp(x):
        return np.sign(x) * np.expm1(abs(x))

    @staticmethod
    def _to_python_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _suggest_parameter(self, trial, param_name, conf):
        param_type = conf["type"]
        param_space = conf.get("space", None)
        param_range = conf.get("range", None)
        param_values = conf.get("values", None)

        # Explicit value lists are safest to expose as discrete Optuna choices.
        if param_values is not None:
            values = [self._to_python_scalar(v) for v in param_values]
            value = trial.suggest_categorical(param_name, values)
            if param_type == "int":
                return int(value)
            if param_type == "real":
                return float(value)
            return value

        if param_type == "bool":
            return bool(trial.suggest_categorical(param_name, [False, True]))

        if param_type in ("cat", "ordinal"):
            raise ValueError(f"'{param_name}' of type '{param_type}' requires 'values'.")

        if param_range is None:
            raise ValueError(f"'{param_name}' of type '{param_type}' requires 'range'.")

        low, high = param_range

        if param_type == "int":
            low, high = int(low), int(high)

            if param_space in (None, "linear"):
                return int(trial.suggest_int(param_name, low, high))

            if param_space == "log":
                if low <= 0:
                    raise ValueError(f"Log-scaled integer '{param_name}' must have low > 0.")
                return int(trial.suggest_int(param_name, low, high, log=True))

            if param_space == "bilog":
                # Bayesmark optimizes in warped space and rounds back to int.
                z_low = float(self._bilog(low))
                z_high = float(self._bilog(high))
                z = trial.suggest_float(f"__bilog__{param_name}", z_low, z_high)
                value = int(np.rint(self._biexp(z)))
                return int(np.clip(value, low, high))

            if param_space == "logit":
                raise ValueError("Bayesmark does not define logit warping for integer parameters.")

            raise ValueError(f"Unsupported space '{param_space}' for integer '{param_name}'.")

        if param_type == "real":
            low, high = float(low), float(high)

            if param_space in (None, "linear"):
                return float(trial.suggest_float(param_name, low, high))

            if param_space == "log":
                if low <= 0:
                    raise ValueError(f"Log-scaled real '{param_name}' must have low > 0.")
                return float(trial.suggest_float(param_name, low, high, log=True))

            if param_space == "logit":
                if not (0.0 < low < high < 1.0):
                    raise ValueError(
                        f"Logit-scaled real '{param_name}' must satisfy 0 < low < high < 1."
                    )
                z_low = self._logit(low)
                z_high = self._logit(high)
                z = trial.suggest_float(f"__logit__{param_name}", z_low, z_high)
                return float(self._expit(z))

            if param_space == "bilog":
                z_low = float(self._bilog(low))
                z_high = float(self._bilog(high))
                z = trial.suggest_float(f"__bilog__{param_name}", z_low, z_high)
                return float(self._biexp(z))

            raise ValueError(f"Unsupported space '{param_space}' for real '{param_name}'.")

        raise ValueError(f"Unsupported Bayesmark parameter type '{param_type}'.")

    def _point_key(self, point):
        """Create a stable key for matching observations to pending trials."""
        return tuple(
            (name, self._to_python_scalar(point[name]))
            for name in self.param_names
        )

    def suggest(self, n_suggestions=1):
        """
        Return ``n_suggestions`` parameter dictionaries for Bayesmark.
        """
        if n_suggestions < 1:
            raise ValueError("n_suggestions must be >= 1")

        suggestions = []

        for _ in range(n_suggestions):
            trial = self.study.ask()
            point = {
                name: self._suggest_parameter(trial, name, self.api_config[name])
                for name in self.param_names
            }

            self._pending[self._point_key(point)].append(trial)
            suggestions.append(point)

        return suggestions

    def observe(self, X, y):
        """
        Feed Bayesmark objective values back to Optuna.

        Bayesmark objectives are losses, so lower is better. The Optuna study
        is therefore created with ``direction='minimize'`` and values are
        passed without changing their sign.
        """
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of observations.")

        for point, value in zip(X, y):
            key = self._point_key(point)
            if key not in self._pending or not self._pending[key]:
                raise ValueError(
                    "Received an observation for a point that was not produced by suggest()."
                )

            trial = self._pending[key].popleft()
            if not self._pending[key]:
                del self._pending[key]

            self.study.tell(trial, float(value))


if __name__ == "__main__":
    experiment_main(OptunaOptimizer)
