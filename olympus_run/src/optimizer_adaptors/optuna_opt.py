import numpy as np
import optuna
from .model_lib import Model, XPoint, YPoint, make_x_point, make_y_point


class OptunaOptimizer(Model):
    _model_name = 'optuna'

    @classmethod
    def _from_config_impl(cls, params: list, task_name: str, on_simplex: bool, task_minimization: bool) -> 'OptunaOptimizer':
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

        for p in params:
            if p.get('type') != 'continuous':
                raise ValueError(
                    f"Parameter '{p.get('name')}' has type '{p.get('type')}', "
                    f"but only 'continuous' is supported by OptunaOptimizer."
                )

        self.study = optuna.create_study(direction="maximize")
        self._last_trial_number = None

        self.params = params
        self.bounds = [(p['low'], p['high']) for p in params]
        self.names = [p['name'] for p in params]
        self.dim = len(params)

    def _suggest_impl(self) -> XPoint:
        trial = self.study.ask()
        x = [trial.suggest_float(self.names[i], *self.bounds[i]) for i in range(self.dim)]
        next_point = make_x_point(np.array(x).reshape(1, -1))
        self._last_trial_number = trial.number
        return next_point

    def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
        if self._last_trial_number is None:
            raise ValueError("suggest() must be called before observe()")
        value = float(y0.ravel()[0])
        self.study.tell(self._last_trial_number, value)
