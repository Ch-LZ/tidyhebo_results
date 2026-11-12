"""One seed schedule and framework RNG contract, matching legacy SeedManager.

No Torch import until explicitly requested. This file also loads directly in
standalone worker Python without importing the host bo_benchmark package.
"""
import json
from pathlib import Path
import random


class SeedManager:
    def __init__(self, master_seed=42, repeats=30):
        import numpy as np
        if type(master_seed) is not int or not 0 <= master_seed < 2**32:
            raise ValueError('master_seed must be a uint32 integer')
        if type(repeats) is not int or repeats < 1:
            raise ValueError('repeats must be positive')
        generator = np.random.RandomState(master_seed)
        self.master_seed = master_seed
        self.seeds = [int(generator.randint(0, 2**31 - 1)) for _ in range(repeats)]

    def seed_for(self, repeat):
        if type(repeat) is not int or not 0 <= repeat < len(self.seeds):
            raise IndexError('Repeat outside saved seed schedule')
        return self.seeds[repeat]

    def state_dict(self):
        return {'version': 1, 'master_seed': self.master_seed, 'seeds': self.seeds.copy()}

    @classmethod
    def from_state(cls, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported seed schedule version')
        values = state.get('seeds')
        if not isinstance(values, list) or not values or any(type(x) is not int or not 0 <= x < 2**31 for x in values):
            raise ValueError('Invalid saved seeds')
        result = cls(state['master_seed'], len(values))
        # Do not silently accept a modified schedule under the same master seed.
        if result.seeds != values:
            raise ValueError('Saved schedule differs from declared master_seed')
        return result

    def save(self, path):
        import os
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix='.seeds-')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(self.state_dict(), stream, indent=2)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, str(path))
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    @classmethod
    def load(cls, path):
        return cls.from_state(json.loads(Path(path).read_text(encoding='utf-8')))

    @staticmethod
    def apply(seed, include_torch=True):
        import numpy as np
        random.seed(seed); np.random.seed(seed)
        if include_torch:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    @staticmethod
    def capture_rng(include_torch=True):
        import numpy as np
        state = {'python': random.getstate(), 'numpy': np.random.get_state()}
        if include_torch:
            import torch
            state.update(torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
        return state

    @staticmethod
    def restore_rng(state):
        import numpy as np
        if 'torch' in state:
            import torch
            if state.get('cuda') and len(state['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA topology changed')
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
            if state.get('cuda'): torch.cuda.set_rng_state_all(state['cuda'])
