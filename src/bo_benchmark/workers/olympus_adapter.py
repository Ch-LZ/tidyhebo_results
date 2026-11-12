"""Python 3.7-compatible Olympus adapter; TF never enters controller/model Python."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unittest.mock import patch


def _task_assets_digest(root, name):
    """Hash stable task assets; never descend into ephemeral Olympus scratch dirs.

    ``.scratch`` is shared by concurrent emulator instances and its temporary
    directories can disappear at any moment. Traversal must be pruned before
    descending, not filtered after recursive discovery. All other files keep
    the previous lexicographic hash order and task-name filter.
    """
    def raise_walk_error(error):
        raise error

    paths = []
    for parent, folders, filenames in os.walk(str(root), topdown=True, onerror=raise_walk_error):
        folders[:] = [folder for folder in folders if folder not in ('.scratch', '__pycache__')]
        paths.extend(Path(parent) / filename for filename in filenames)

    digest = hashlib.sha256()
    assets = 0
    for path in sorted(paths):
        relative = str(path.relative_to(root))
        if path.is_file() and name in relative and '__pycache__' not in relative:
            digest.update(relative.encode('utf-8'))
            digest.update(path.read_bytes())
            assets += 1
    return digest.hexdigest(), assets


class OlympusBenchmark:
    def __init__(self, name, seed, options):
        import numpy as np
        import tensorflow as tf
        # Olympus uses a check-then-mkdir during import. Create its scratch
        # directory atomically before parallel workers import the package.
        import importlib.util
        module = importlib.util.find_spec('olympus') if 'olympus' not in sys.modules else None
        if module is not None and module.origin:
            scratch = Path(os.environ.get('OLYMPUS_SCRATCH', str(Path(module.origin).parent/'.scratch')))
            scratch.mkdir(parents=True, exist_ok=True)
        import olympus
        from olympus import Emulator
        self.np, self.seed, self.history = np, seed, []
        from task_catalog import task_reference
        reference = task_reference(name)
        config = reference['task_config']
        if options.get('task_config_sha256', reference['task_config_sha256']) != reference['task_config_sha256']:
            raise ValueError('Olympus task configuration changed between plan and worker')
        from seeds import SeedManager
        SeedManager.apply(seed, include_torch=False)
        if hasattr(tf, 'set_random_seed'):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.set_random_seed(seed)
        original = tf.Graph.__init__
        def seeded_init(graph, *args, **kwargs):
            original(graph, *args, **kwargs)
            graph.seed = seed
        with patch.object(tf.Graph, '__init__', seeded_init):
            self.emulator = Emulator(name, 'BayesNeuralNet')
        self.simplex = None
        params = config['params']
        if config.get('on_simplex'):
            from simplex import SimplexTransformer
            self.simplex = SimplexTransformer(len(params))
            params = params[:-1]
        self.description = {'parameters': [{'name': p['name'], 'lower': p['low'], 'upper': p['high']}
                                           for p in params],
                'direction': 'minimize' if config['less_is_better'] else 'maximize',
                'metadata': {'benchmark': 'olympus', 'task': name, 'simplex': self.simplex is not None,
                             'physical_columns': [p['name'] for p in config['params']]}}
        root = Path(olympus.__file__).resolve().parent
        asset_sha256, assets = _task_assets_digest(root, name)
        self.metadata = {'tensorflow': tf.__version__, 'numpy': np.__version__,
                         'olympus': str(getattr(olympus, '__version__', 'unknown')),
                         'task_assets_sha256': asset_sha256, 'task_asset_count': assets,
                         **reference}
        self.description['metadata'].update(reference)

    def evaluate(self, X):
        x = self.np.asarray(X, dtype=float)
        params = self.description['parameters']
        if (x.ndim != 2 or x.shape[1] != len(params) or not len(x)
                or not self.np.isfinite(x).all()
                or (x < self.np.array([p['lower'] for p in params])).any()
                or (x > self.np.array([p['upper'] for p in params])).any()):
            raise ValueError('Olympus input must be a finite batch inside the declared domain')
        physical = self.simplex.transform(x) if self.simplex else x
        values, _, _ = self.emulator.run(physical)
        y = self.np.asarray(values, dtype=float).reshape(len(x), -1)
        if y.shape != (len(x), 1) or not self.np.isfinite(y).all():
            raise ValueError('Expected finite scalar Olympus responses')
        self.history.append((x.copy(), y.copy()))
        return y, physical

    def save_state(self):
        from seeds import SeedManager
        return {'history': [(x.copy(), y.copy()) for x,y in self.history],
                'rng': SeedManager.capture_rng(include_torch=False)}

    def load_state(self, state):
        if self.history:
            raise ValueError('Restore requires a fresh emulator')
        # Same strategy as the working legacy runner: seeded TF graph plus
        # ordered replay, validating every raw response before continuing.
        for x, old in state['history']:
            y, _ = self.evaluate(x)
            self.np.testing.assert_array_equal(y, old)
        from seeds import SeedManager
        SeedManager.restore_rng(state['rng'])
