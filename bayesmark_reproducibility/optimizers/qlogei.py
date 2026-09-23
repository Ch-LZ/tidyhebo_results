from time import sleep

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


from tidyhebo.optimizers_reference import qLogEIMixedOptimizer


class LogEIBoTorchOptimizer(AbstractOptimizer):
    def __init__(self, api_config):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        _params = self.parce_space(api_config)
        n_init = 8
        self._optimizer = qLogEIMixedOptimizer(_params, n_init=n_init)


    def suggest(self, n_suggestions=1):
        """Get suggestion.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        cand = self._optimizer.suggest()
        rv = cand.to_dict(orient="records")
        # print(n_suggestions)
        return rv


    def observe(self, X, y):
        # print(X, y)
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Random search so don't do anything
        # x -> from list(dict) to dataframe
        import pandas as pd
        import numpy as np
        import torch

        x = pd.DataFrame.from_records(X)
        y = torch.from_numpy(np.array(y)).double()
        self._optimizer.observe(x, -1 * y)


    def parce_space(self, api_config):
        from tidyhebo.mixed_params import RealP, IntegerP, CategoricalP, BoolP, RealPowP, AbstractParameter
        from typing import List

        params: List[AbstractParameter] = []
        for i, param_name in enumerate(api_config):
            param_conf   = api_config[param_name]
            param_type   = param_conf['type']
            param_space  = param_conf.get('space', None)
            param_range  = param_conf.get("range", None)
            param_values = param_conf.get("values", None)

            if param_type == "int":
                p = IntegerP(index=i, lb=param_range[0], ub=param_range[1], name=param_name)
                params.append(p)

            elif param_type == "bool":
                p = BoolP(i, name=param_name)
                params.append(p)

            elif param_type in ("cat", "ordinal"):
                p = CategoricalP(i, list(set(param_values)), name=param_name)
                params.append(p)

            elif param_type == "real":
                if param_space in ('log', 'logit'):   # todo: что с log-space int? На практике в бенчмарке они вообще встречаются?
                    p = RealPowP(i, lb=param_range[0], ub=param_range[1], base=10.0, name=param_name)
                    params.append(p)

                else:
                    p = RealP(i, lb=param_range[0], ub=param_range[1], name=param_name)
                    params.append(p)

            else:
                assert False, "type %s not handled in API" % param_type
        return params


if __name__ == "__main__":
    experiment_main(LogEIBoTorchOptimizer)
