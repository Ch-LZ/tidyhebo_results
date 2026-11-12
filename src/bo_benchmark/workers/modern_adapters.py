"""Mathematical adapters in the model Python. No controller/IPC dependencies."""
from dataclasses import replace, asdict
import random
import copy
import numpy as np
import pandas as pd
import torch


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seeds import SeedManager


def seed_all(seed):
    SeedManager.apply(seed)


def rng_state():
    return SeedManager.capture_rng()


def restore_rng(state):
    SeedManager.restore_rng(state)


def core_config(options, q, seed):
    from tidyhebo.config import (TidyHEBOConfig, ModelConfig, AcquisitionConfig,
                                GASearchConfig, StopConfig, BoTorchSearchConfig)
    search = dict(options.get('search', {}))
    target = search.get('_target_', 'layered' if 'search' in options else 'pymoo')
    if target == 'layered':
        from tidyhebo.layered import LayeredSearchConfig
        defaults = LayeredSearchConfig()
        for key in ('point_init', 'batch_init', 'points', 'batches', 'scalarization'):
            if isinstance(search.get(key), dict):
                search[key] = replace(getattr(defaults, key), **search[key])
        search = LayeredSearchConfig(**search)
    elif target == 'pymoo':
        if isinstance(search.get('stop'), dict): search['stop'] = StopConfig(**search['stop'])
        search = GASearchConfig(**search)
    elif target == 'botorch':
        search = BoTorchSearchConfig(**search)
    else:
        raise ValueError('Unsupported search target: ' + target)
    acquisition = dict(options.get('acquisition', {}))
    family = acquisition.get('_target_', 'MACE')
    if family not in ('MACE', 'qMACE', 'qLogEI'):
        raise ValueError('Supported acquisition families: MACE/qMACE/qLogEI')
    point_batching = getattr(search, 'adaptive_directions', 'off') == 'success'
    if point_batching:
        if family == 'qLogEI':
            raise ValueError('Point batching requires MACE, not qLogEI')
        search = replace(search, point_batch_size=q)
    acquisition.update(q=1 if point_batching else q, qmc_seed=seed,
                       _target_='MACE' if point_batching else (
                           'qLogEI' if family == 'qLogEI' else ('MACE' if q == 1 else 'qMACE')))
    config = TidyHEBOConfig(model=ModelConfig(**options.get('model', {})),
                           acquisition=AcquisitionConfig(**acquisition), search=search)
    config.check_supported()
    return config


class TidyHEBOAdapter:
    """Exact sync DataFrame suggest/observe, optional native math state.

    Default startup uses the library's OWN Sobol generator. legacy_qmace mode
    explicitly reproduces qmace_lab's single seeded initial-design batch.
    """
    def __init__(self, task, seed, options):
        from tidyhebo import tidyHEBO, RealP
        self.task, self.seed, self.options = task, seed, options
        self.columns = [p['name'] for p in task['parameters']]
        self.parameters = [RealP(i, p['lower'], p['upper'], name=p['name'])
                           for i, p in enumerate(task['parameters'])]
        self.n_init = options.get('n_init', 4)
        self.issued = 0
        self.legacy = options.get('startup', 'internal') == 'legacy_qmace'
        seed_all(seed)
        self.config = core_config(options, 1, seed)
        self.metadata = {"config":asdict(self.config),"startup":options.get("startup", "internal")}
        self.core = tidyHEBO(self.parameters, n_init=self.n_init, config=self.config)
        self.initial = None
        if self.legacy:
            # Match qmace_lab exactly: explicit seeded batched Sobol before core construction.
            design = torch.quasirandom.SobolEngine(len(self.parameters), scramble=True,
                                                   seed=seed).draw(self.n_init, dtype=torch.double)
            self.initial = self.core.design_space.from_GP_to_DS(design, return_df=True)
        self.simplex = None
        if task.get('metadata', {}).get('simplex'):
            from simplex import SimplexTransformer
            self.simplex = SimplexTransformer(len(self.columns) + 1)

    def suggest(self, n=1):
        remaining = self.n_init - self.issued
        if remaining > 0 and n > remaining:
            raise ValueError('Request crosses Sobol startup boundary; split the batch in protocol')
        if self.legacy and remaining > 0:
            frame = self.initial.iloc[self.issued:self.issued+n].copy()
        else:
            self.core.config = core_config(self.options, n, self.seed)
            frame = self.core.suggest()
        self.issued += len(frame) if remaining > 0 else 0
        if len(frame) != n:
            raise ValueError('Library candidate count differs from requested n')
        return frame.reset_index(drop=True)

    def observe(self, X, y):
        # Standalone mathematical API keeps its two-method contract.
        physical = self.simplex.transform(X.to_numpy(dtype=float)) if self.simplex else None
        self._observe_simplex(X, y, physical)

    def observe_evaluated(self, X, y, physical_X):
        if self.simplex is None:
            # Respect subclasses such as the finite Wildfire adapter.
            return self.observe(X, y)
        physical = np.asarray(physical_X, dtype=float)
        expected = self.simplex.transform(X.to_numpy(dtype=float))
        if (physical.shape != expected.shape or not np.isfinite(physical).all()
                or not self.simplex.validate(physical)
                or not np.allclose(physical, expected, rtol=0, atol=1e-12)):
            raise ValueError('Evaluated simplex composition does not match issued points')
        # The benchmark's actual physical values are authoritative across Python environments.
        self._observe_simplex(X, y, physical)

    def _observe_simplex(self, X, y, physical):
        if self.legacy and self.core.train_y.numel() == 0:
            # qmace_lab's external initial design trained on the ORIGINAL auxiliary
            # frame; only subsequent adapter.observe calls applied the guarded inverse.
            from tidyhebo import tidyHEBO
            self.core = tidyHEBO(self.parameters, train_x=X, train_y=y,
                                 config=core_config(self.options, 1, self.seed))
            return
        if self.simplex is not None:
            if getattr(self.core.config.search, 'adaptive_directions', 'off') == 'success':
                from tidyhebo.layered.adaptive import prepare_simplex_feedback
                prepare_simplex_feedback(self.core, self.simplex, physical)
            X = pd.DataFrame(self.simplex.inverse_transform(physical), columns=self.columns)
        self.core.observe(X, y)

    def save_state(self):
        remaining = None
        if self.core._init_design_generator is not None:
            # Sobol draws use the already seeded engine, not global RNG. Preserve
            # all remaining frames; generators themselves cannot be pickled.
            remaining = list(self.core._init_design_generator)
            self.core._init_design_generator = iter(remaining)
        return copy.deepcopy({'train_x': self.core.train_x, 'train_y': self.core.train_y,
                'parameters': self.core.state_dict,
                'direction_state': self.core.direction_state,
                'remaining': remaining, 'initial': self.initial, 'issued': self.issued,
                'rng': rng_state()})

    def diagnostics(self):
        return {'af_seconds': None,
                'warp_active': bool(self.core.model.use_power_outcome_transform) if self.core.model else None,
                'native_repair': copy.deepcopy(getattr(self.core, 'search_diagnostics', {})),
                'ga_budget': None,
                'adaptive_history': copy.deepcopy(self.core.direction_state.get('adaptive_success', {}).get('history', []))}

    def load_state(self, state):
        from tidyhebo import tidyHEBO
        frame = self.core.design_space.from_GA_to_DS(state['train_x'], return_df=True)
        self.core = tidyHEBO(self.parameters, train_x=frame,
                            train_y=state['train_y'].numpy(), config=self.config,
                            _state_dict=state['parameters'])
        self.core.train_x, self.core.train_y = state['train_x'], state['train_y']
        self.core._init_design_generator = (None if state['remaining'] is None
                                             else iter(state['remaining']))
        self.core.direction_state = copy.deepcopy(state.get('direction_state', {}))
        self.initial, self.issued = state['initial'], state['issued']
        restore_rng(state['rng'])


class RandomAdapter:
    def __init__(self, task, seed, options):
        self.parameters = task['parameters']
        seed_all(seed)

    def suggest(self, n=1):
        return pd.DataFrame(np.random.uniform([p['lower'] for p in self.parameters],
            [p['upper'] for p in self.parameters], (n, len(self.parameters))),
            columns=[p['name'] for p in self.parameters])

    def observe(self, X, y):
        pass

    def save_state(self):
        return rng_state()

    def load_state(self, state):
        restore_rng(state)


class SyntheticBenchmark:
    def __init__(self, name, seed, options):
        from botorch.test_functions.synthetic import Ackley, Hartmann, Branin
        seed_all(seed)
        self.function = {'ackley': lambda: Ackley(dim=6), 'hartmann': lambda: Hartmann(dim=6),
                         'branin': Branin}[name]().double()
        self.description = {'parameters': [{'name': 'x'+str(i), 'lower': float(a), 'upper': float(b)}
                            for i, (a,b) in enumerate(self.function.bounds.T)],
                            'direction': 'minimize', 'metadata': {'benchmark': 'synthetic', 'task': name}}

    def evaluate(self, X):
        x = X.to_numpy(dtype=float) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=float)
        return self.function(torch.tensor(x, dtype=torch.double)).detach().numpy().reshape(-1, 1), x

    def save_state(self):
        return rng_state()

    def load_state(self, state):
        restore_rng(state)
