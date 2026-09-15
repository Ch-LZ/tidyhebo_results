# src/optimizer_adaptors/random_opt.py

import numpy as np
from .model_lib import Model, XPoint, YPoint, make_x_point, make_y_point


class RandomOptimizer(Model):
    _model_name = 'random'

    @classmethod
    def _from_config_impl(cls, params: list, task_name: str, on_simplex: bool, task_minimization: bool) -> 'RandomOptimizer':
        if params is None:
            raise ValueError("Missing 'params' in config")
        return cls(params=params, task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

    def __init__(self, params, task_name: str, on_simplex: bool = False, task_minimization: bool = False):
        # Normalize params by defaulting missing types to 'continuous'.
        processed_params = []
        for p in params:
            if 'type' not in p:
                p = dict(p)
                p['type'] = 'continuous'
            processed_params.append(p)
        params = processed_params

        super().__init__(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

        for p in params:
            if p.get('type') != 'continuous':
                raise ValueError(
                    f"Parameter '{p.get('name')}' has type '{p.get('type')}', "
                    f"but only 'continuous' is supported by RandomOptimizer."
                )
        self.params = params
        self.bounds = [(p['low'], p['high']) for p in params]
        self.dim = len(params)

    def _suggest_impl(self) -> XPoint:
        x = [np.random.uniform(low, high) for (low, high) in self.bounds]
        point = make_x_point(np.array(x).reshape(1, -1))
        return point

    def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
        pass  # nothing to observe for random optimizer
