# src/optimizer_adaptors/logei_opt.py

import numpy as np
import torch
import pandas as pd
from typing import Optional, Union

from .model_lib import Model, XPoint, YPoint, make_x_point, make_y_point

try:
    from src.reference_optimizer import LogEIMixedOptimizer
    from tidyhebo.mixed_params import RealP
except ImportError:
    import warnings
    warnings.warn("Import error while importing logei_opt. This might indicate, that the module was imported in mismatched environment")
    LogEIMixedOptimizer = None


class LogEIOptimizer(Model):
    _model_name = 'logei'

    @classmethod
    def _from_config_impl(cls, params: list, task_name: str, on_simplex: bool, task_minimization: bool) -> 'LogEIOptimizer':
        if params is None:
            raise ValueError("Missing 'params' in config")

        # Convert parameters to RealP objects; only continuous variables are supported.
        hebo_params = []
        for i, p in enumerate(params):
            if p.get('type') != 'continuous':
                raise ValueError(f"Only 'continuous' parameters are supported, got {p.get('type')}")
            hebo_params.append(
                RealP(i, p['low'], p['high'], name=p.get('name', f'param_{i}'))
            )

        # Default model options; these can be extended through config if needed.
        return cls(
            params=hebo_params,
            task_name=task_name,
            task_minimization=task_minimization,
            on_simplex=on_simplex,
            n_init=4,
            kernel='Matern52',
            power_outcome=False,
            warp_input=False,
            use_gate=False
        )

    def __init__(
        self,
        params,
        *,
        task_name: str,
        task_minimization: bool,
        on_simplex: bool = False,
        n_init: int = 4,
        kernel: str = 'Matern52',
        power_outcome: bool = False,
        warp_input: bool = False,
        use_gate: bool = False,
        train_x: Optional[Union[pd.DataFrame, torch.Tensor, np.ndarray]] = None,
        train_y: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        if LogEIMixedOptimizer is None:
            raise ImportError("tidyhebo with LogEIMixedOptimizer is not installed. Please install it first.")

        super().__init__(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

        self.params = params
        self.n_init = n_init
        self.kernel = kernel
        self.power_outcome = power_outcome
        self.warp_input = warp_input
        self.use_gate = use_gate

        # Create the underlying optimizer.
        self.optimizer = LogEIMixedOptimizer(
            parameters=params,
            train_x=train_x,
            train_y=train_y,
            n_init=n_init if (train_x is None and train_y is None) else None
        )
        # Configure model parameters.
        self.optimizer.kernel = kernel
        self.optimizer.power_outcome = power_outcome
        self.optimizer.warp_input = warp_input
        self.optimizer._use_gate = use_gate

    def _suggest_impl(self) -> XPoint:
        result = self.optimizer.suggest()  # Returns a pandas DataFrame.
        point = result.iloc[0].values.astype(np.float64).reshape(1, -1)
        return make_x_point(point)

    def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
        if x0.ndim == 1:
            x0 = x0.reshape(1, -1)

        # Parameter names.
        param_names = [p.name for p in self.params]
        df_x = pd.DataFrame(x0, columns=param_names)

        y_val = y0.ravel()[0]
        y_tensor = torch.tensor([[y_val]], dtype=torch.double)
        self.optimizer.observe(df_x, y_tensor)