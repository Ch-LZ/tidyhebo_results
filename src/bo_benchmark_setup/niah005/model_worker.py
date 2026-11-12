"""Experiment-specific paired Sobol startup; reuse the audited native worker."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/bo_benchmark/workers')]
import scientific_worker
import modern_adapters
import numpy as np
import pandas as pd
import torch


class PairedTidyHEBOAdapter(modern_adapters.TidyHEBOAdapter):
    """Own only initial-design delivery; the library owns all subsequent BO."""
    def __init__(self, task, seed, options):
        super().__init__(task, seed, options)
        if task.get('metadata', {}).get('benchmark') != 'niah':
            raise ValueError('This adapter is restricted to NIAH')
        if self.legacy or self.simplex is not None:
            raise ValueError('Expected internal startup in a box domain')
        bounds = np.array([[p['lower'], p['upper']] for p in task['parameters']])
        unit = torch.quasirandom.SobolEngine(len(self.columns), scramble=True, seed=seed).draw(
            self.n_init, dtype=torch.double).numpy()
        self.initial = pd.DataFrame(bounds[:, 0] + unit*np.diff(bounds,axis=1).ravel(), columns=self.columns)
        self.core._init_design_generator = None
        self.metadata['initial_design'] = 'paired-explicit-sobol-double-v1'

    def suggest(self, n=1):
        remaining = self.n_init - self.issued
        if remaining > 0:
            if not 1 <= n <= remaining:
                raise ValueError('Request crosses paired initial-design boundary')
            frame = self.initial.iloc[self.issued:self.issued+n].copy()
            self.issued += n
            return frame.reset_index(drop=True)
        return super().suggest(n)


original_fingerprint = scientific_worker.fingerprint

def fingerprint(configuration, adapter):
    identity = original_fingerprint(configuration, adapter)
    digest = hashlib.sha256(identity['code'].encode())
    digest.update(Path(__file__).read_bytes().replace(b'\r\n',b'\n'))
    identity['code'] = digest.hexdigest()
    return identity


if __name__ == '__main__':
    modern_adapters.TidyHEBOAdapter = PairedTidyHEBOAdapter
    scientific_worker.fingerprint = fingerprint
    raise SystemExit(scientific_worker.main())
