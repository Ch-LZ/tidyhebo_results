from time import sleep

import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main


from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace


class HEBOOptimizer(AbstractOptimizer):
    def __init__(self, api_config):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        param_space = self.parce_space(api_config)


        n_init = 8
        self._optimizer = HEBO(param_space, rand_sample=n_init)


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
        cand = self._optimizer.suggest(n_suggestions=1)
        rv = cand.to_dict(orient="records")
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
        y = np.array(y)
        self._optimizer.observe(x, y)


    def parce_space(self, api_config):
        space  = DesignSpace()
        params = []
        for param_name in api_config:
            param_conf   = api_config[param_name]
            param_type   = param_conf['type']
            param_space  = param_conf.get('space', None)
            param_range  = param_conf.get("range", None)
            param_values = param_conf.get("values", None)

            bo_param_conf = {'name' : param_name}
            if param_type == 'int': # ignore 'log' space # TODO: support log-scale int
                #*int
                bo_param_conf['type'] = 'int'
                bo_param_conf['lb']   = param_range[0]
                bo_param_conf['ub']   = param_range[1]
            elif param_type == 'bool':
                #* bool
                bo_param_conf['type'] = 'bool'
            elif param_type in ('cat', 'ordinal'):
                #* categorical был
                bo_param_conf['type']       = 'cat'
                bo_param_conf['categories'] = list(set(param_values))
                #* real
            elif param_type == 'real':
                #* real logarithmic
                if param_space in ('log', 'logit'):
                    bo_param_conf['type'] = 'pow'
                    bo_param_conf['base'] = 10
                    bo_param_conf['lb']   = param_range[0]
                    bo_param_conf['ub']   = param_range[1]
                else:
                    #* real numeric
                    bo_param_conf['type'] = 'num'
                    bo_param_conf['lb']   = param_range[0]
                    bo_param_conf['ub']   = param_range[1]
            else:
                assert False, "type %s not handled in API" % param_type
            params.append(bo_param_conf)
        # print(params)
        space.parse(params)
        return space


if __name__ == "__main__":
    experiment_main(HEBOOptimizer)
