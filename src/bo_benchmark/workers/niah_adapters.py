"""NIAH RF oracles and finite Wildfire search, preserving the legacy pool policy."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
import torch
from tidyhebo.config import DiscreteSearchConfig
from tidyhebo.candidate_search import DiscreteSearch
from tidyhebo.archive import ParetoAccumulator
from modern_adapters import TidyHEBOAdapter, core_config, seed_all
from niah_assets import load_asset, load_candidates


@dataclass(frozen=True)
class NIAHPoolConfig(DiscreteSearchConfig):
    _target_: str = 'niah_pool'


class NIAHPoolSearch(DiscreteSearch, _target_='niah_pool'):
    def search(self, acqf, train_x):
        archive = ParetoAccumulator()
        with torch.no_grad():
            for chunk in self.candidates.to(train_x).split(64):
                chunk = chunk[self.design_space.are_values_new(chunk, train_x)]
                if not len(chunk): continue
                scores = acqf(self.design_space.from_GA_to_GP(chunk).unsqueeze(1))
                if not torch.isfinite(scores).all():
                    raise RuntimeError('Nonfinite acquisition scores in finite search')
                archive.add(chunk.numpy(), -scores.numpy())
        if not archive.has_data: raise RuntimeError('No unobserved finite candidates remain')
        return archive.take_front()[0]


class FiniteTidyHEBOAdapter(TidyHEBOAdapter):
    def __init__(self, task, seed, options):
        super().__init__(task, seed, options)
        meta = task['metadata']
        self.a, self.candidate_metadata = load_candidates(meta['data_dir'], meta['asset_id'])
        self.features = self.a['x']
        if self.n_init > len(self.features): raise ValueError('Not enough unique Wildfire rows for initialization')
        self.first_ids = np.random.default_rng(seed).choice(len(self.features), self.n_init, replace=False)
        self.used = set(); self.pending = None; self.last = {}
        self.candidate_limit = meta.get('candidate_limit', 0)
        self.core._init_design_generator = None  # Finite table uses row initialization, never off-table Sobol.
        self.metadata.update(finite_policy='legacy_niah_pool', candidate_asset_id=self.candidate_metadata['asset_id'])

    def suggest(self, n=1):
        if n != 1: raise ValueError('Legacy finite Wildfire protocol supports q=1 only')
        if self.pending is not None: raise ValueError('Observe the pending row first')
        step = len(self.used); pool_ids = None
        if step < self.n_init:
            row = int(self.first_ids[step])
            frame = pd.DataFrame(self.features[row:row+1], columns=self.columns)
        else:
            remaining = np.setdiff1d(np.arange(len(self.features)), list(self.used), assume_unique=True)
            if not len(remaining): raise RuntimeError('Finite Wildfire table is exhausted')
            if self.candidate_limit and len(remaining) > self.candidate_limit:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed,step,913]))
                pool_ids = rng.choice(remaining, self.candidate_limit, replace=False)
            else: pool_ids = remaining
            pool_x = self.features[pool_ids]
            self.core.config = replace(core_config(self.options,1,self.seed), search=NIAHPoolConfig(
                candidates=pd.DataFrame(pool_x,columns=self.columns)))
            frame = self.core.suggest().reset_index(drop=True)
            match = np.flatnonzero((pool_x == frame.to_numpy()[0]).all(axis=1))
            if len(match) != 1: raise RuntimeError('Point outside finite candidate pool')
            row = int(pool_ids[match[0]])
        self.pending = row
        self.last = {'candidate_id': row, 'source_row': int(self.a['source_row'][row]),
                     'source_samples': int(self.a['counts'][row]),
                     'candidate_pool_size': len(pool_ids) if pool_ids is not None else None}
        return frame

    def observe(self, X, y):
        if self.pending is None or not np.array_equal(X.to_numpy()[0],self.features[self.pending]):
            raise ValueError('Observation does not match the pending finite row')
        self.core.observe(X,y)
        self.used.add(self.pending); self.pending = None
        self.issued = min(self.n_init,len(self.used))

    def diagnostics(self):
        return {**super().diagnostics(), **self.last}

    def save_state(self):
        if self.pending is not None: raise ValueError('Cannot checkpoint an unobserved finite row')
        state = super().save_state()
        state.update(used=sorted(self.used), first_ids=self.first_ids.copy(), last=dict(self.last))
        return state

    def load_state(self, state):
        super().load_state(state)
        self.used = set(state['used']); self.first_ids = state['first_ids']; self.last = state['last']
        self.pending = None


class FiniteRandomAdapter:
    def __init__(self, task, seed, options):
        self.task, self.seed = task, seed
        meta = task['metadata']
        self.a, self.meta = load_candidates(meta['data_dir'],meta['asset_id'])
        self.features = self.a['x']; self.columns = [p['name'] for p in task['parameters']]
        self.n_init = options.get('n_init',4)
        seed_all(seed)
        self.rng = np.random.default_rng(seed)
        self.first_ids = self.rng.choice(len(self.features), self.n_init, replace=False)
        self.used = set(); self.pending = None; self.last = {}

    def suggest(self,n=1):
        if n != 1 or self.pending is not None: raise ValueError('Finite protocol requires one pending row')
        remaining = np.setdiff1d(np.arange(len(self.features)),list(self.used),assume_unique=True)
        if not len(remaining): raise RuntimeError('Finite Wildfire table is exhausted')
        row = int(self.first_ids[len(self.used)]) if len(self.used)<self.n_init else int(self.rng.choice(remaining))
        self.pending = row
        self.last = {'candidate_id':row,'source_row':int(self.a['source_row'][row]),
                     'source_samples':int(self.a['counts'][row]),'candidate_pool_size':len(remaining)}
        return pd.DataFrame(self.features[row:row+1],columns=self.columns)

    def observe(self,X,y):
        if self.pending is None or not np.array_equal(X.to_numpy()[0],self.features[self.pending]):
            raise ValueError('Observation does not match pending row')
        self.used.add(self.pending);self.pending=None

    def diagnostics(self): return dict(self.last)

    def save_state(self):
        return {'used':sorted(self.used),'first_ids':self.first_ids,'rng':self.rng.bit_generator.state,'last':self.last}

    def load_state(self,state):
        self.used=set(state['used']);self.first_ids=state['first_ids'];self.rng.bit_generator.state=state['rng'];self.last=state['last']
