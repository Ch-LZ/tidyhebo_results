# src/optimizer_adaptors/hebo_opt.py

from typing import List
import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from .model_lib import Model, XPoint, YPoint, make_x_point, make_y_point


class HeboOptimizer(Model):
    _model_name = 'hebo'

    @classmethod
    def _from_config_impl(cls, params: list, task_name: str, on_simplex: bool, task_minimization: bool) -> 'HeboOptimizer':
        if params is None:
            raise ValueError("Missing 'params' in config")
        return cls(params=params, task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

    def __init__(self, params, task_name: str, on_simplex: bool = False, task_minimization: bool = False):
        # Default missing parameter types to 'continuous'.
        processed_params = []
        for p in params:
            if 'type' not in p:
                p = dict(p)
                p['type'] = 'continuous'
            processed_params.append(p)
        params = processed_params

        super().__init__(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

        # Convert parameters to HEBO format:
        # - map 'continuous' to 'num'
        # - map 'low'/'high' to 'lb'/'ub'
        hebo_params = []
        for p in params:
            p_copy = p.copy()
            if p_copy.get('type') == 'continuous':
                p_copy['type'] = 'num'
            if 'low' in p_copy:
                p_copy['lb'] = p_copy.pop('low')
            if 'high' in p_copy:
                p_copy['ub'] = p_copy.pop('high')
            hebo_params.append(p_copy)

        self.design_space = DesignSpace().parse(hebo_params)
        self.optimizer = HEBO(self.design_space)
        self._cols = None
        self.params = params
        self.dim = len(params)

    def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
        if self._cols is None:
            self._cols = [f'col_{i}' for i in range(x0.shape[1])]
        x0_df = pd.DataFrame(data=x0, columns=self._cols)
        # HEBO minimizes, while y0 follows the maximization-oriented adaptor interface.
        # Therefore HEBO receives -y0.
        self.optimizer.observe(x0_df, -y0.ravel())

    def _suggest_impl(self) -> XPoint:
        cand = self.optimizer.suggest(n_suggestions=1)
        self._cols = cand.columns.tolist()
        next_point = cand.values.reshape(1, -1)
        return make_x_point(next_point)

