"""NumPy-only NIAH oracle service, separate from optimizer implementation."""
import numpy as np
from niah_assets import load_asset


class NIAHBenchmark:
    def __init__(self, name, seed, options):
        try: self.oracle = load_asset(options['data_dir'], name)
        except FileNotFoundError as exc:
            raise FileNotFoundError('Prepare pinned NIAH assets with python run_experiments.py setup niah: '+str(exc)) from exc
        if self.oracle.metadata['kind']=='table' and options.get('n_trials',0)>len(self.oracle.features):
            raise ValueError('Evaluation budget exceeds unique Wildfire rows')
        self.name = name
        self.metadata = dict(self.oracle.metadata)
        metadata = {**self.metadata, 'benchmark': 'niah', 'task': name,
                    'data_dir': options['data_dir'], 'candidate_limit': options.get('candidate_limit', 0)}
        self.description = {'parameters': [{'name':'x'+str(i), 'lower':0., 'upper':1.}
                                          for i in range(self.oracle.dim)],
                            'direction':'minimize', 'metadata':metadata}
        self.row_index = None
        if self.metadata['kind'] == 'table':
            # Exact byte lookup of the normalized vectors, preserving row identity.
            dtype = np.dtype((np.void, self.oracle.features.dtype.itemsize*self.oracle.dim))
            keys = np.ascontiguousarray(self.oracle.features).view(dtype).ravel()
            self.row_index = {key.tobytes(): i for i,key in enumerate(keys)}

    def evaluate(self, X):
        x = np.asarray(X, dtype=float)
        if self.row_index is None:
            return np.asarray(self.oracle(x)).reshape(-1,1), x
        values, physical = [], []
        for point in x:
            row = self.row_index.get(np.ascontiguousarray(point).tobytes())
            if row is None: raise ValueError('Wildfire can evaluate only an actual candidate row')
            values.append([self.oracle.query(row)]); physical.append(self.oracle.physical(row))
        return np.asarray(values), np.asarray(physical)

    def save_state(self):
        return {'asset_id': self.metadata['asset_id']}

    def load_state(self, state):
        if state['asset_id'] != self.metadata['asset_id']: raise ValueError('NIAH asset changed')


