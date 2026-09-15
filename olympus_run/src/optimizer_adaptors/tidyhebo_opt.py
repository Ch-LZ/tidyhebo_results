# src/optimizer_adaptors/tidyhebo_opt.py

import numpy as np
import pandas as pd
from typing import List, Dict

from .model_lib import Model, XPoint, YPoint, make_x_point

try:
    from tidyhebo import tidyHEBO
    from tidyhebo.mixed_params import RealP
except ImportError:
    raise ImportError("tidyhebo is not installed. Please install it first.")


class TidyheboOptimizer(Model):
    _model_name = 'tidyhebo'

    @classmethod
    def _from_config_impl(cls, params: list, task_name: str, on_simplex: bool, task_minimization: bool) -> 'TidyheboOptimizer':
        if params is None:
            raise ValueError("Missing 'params' in config")
        for p in params:
            if p.get('type') != 'continuous':
                raise ValueError(f"Only 'continuous' parameters are supported, got {p.get('type')}")
        return cls(
            params=params,
            task_minimization=task_minimization,
            task_name=task_name,
            on_simplex=on_simplex,
            n_init=4,
            return_df=True
        )

    def __init__(
        self,
        params: List[Dict],
        *,
        task_name: str,
        task_minimization: bool,
        on_simplex: bool = False,
        n_init: int = 4,
        return_df: bool = True
    ):
        super().__init__(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

        self.params = params
        self.n_init = n_init
        self._return_df = return_df

        # Convert parameters to tidyHEBO RealP objects.
        real_params = []
        for i, p in enumerate(params):
            real_params.append(
                RealP(i, p['low'], p['high'], name=p['name'])
            )

        self.optimizer: tidyHEBO = tidyHEBO(
            parameters=real_params,
            n_init=n_init
        )
        self.n_from_pareto_front = []
        # Store bounds for vectorized random-point generation.
        self._lows = np.array([p['low'] for p in self.params])
        self._highs = np.array([p['high'] for p in self.params])


    def _suggest_impl(self) -> XPoint:
        result = self.optimizer.suggest()
        # The underlying optimizer always returns a pandas DataFrame.
        point = result.iloc[0].values.astype(np.float64)
        return make_x_point(point.reshape(1, -1))


    def _suggest_batch_impl(self, batch_size: int) -> XPoint:
        result = self.optimizer.suggest(return_all=True)
        # The underlying optimizer always returns a pandas DataFrame.
        pareto_points = result.values.astype(np.float64)

        n_pareto = len(pareto_points)
        self.n_from_pareto_front.append(n_pareto)

        if n_pareto >= batch_size:
            points = pareto_points[:batch_size]
        else:
            n_random = batch_size - n_pareto
            random_points = np.random.uniform(self._lows, self._highs, size=(n_random, len(self.params)))
            points = np.vstack([pareto_points, random_points])

        return points.reshape(batch_size, -1)


    def _suggest_adaptive_batch_impl(self, adaptive_ceil: int) -> XPoint:
        result: pd.DataFrame = self.optimizer.suggest(return_all=True)
        q_i = min(adaptive_ceil, len(result))

        self.n_from_pareto_front.append(q_i)  # remember how many points were sampled

        if q_i == len(result):
            return result.values.astype(np.float64)
        else:
            return result.sample(n=q_i, replace=False).values.astype(np.float64)


    def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
        if x0.ndim == 1:
            x0 = x0.reshape(1, -1)
        df_x = pd.DataFrame(x0, columns=[p['name'] for p in self.params])
        self.optimizer.observe(df_x, y0)

    def _observe_batch_impl(self, x_batch: XPoint, y_batch: YPoint) -> None:
        df_x = pd.DataFrame(x_batch, columns=[p['name'] for p in self.params])
        self.optimizer.observe(df_x, y_batch)