"""Reuse the audited worker protocol/checkpoints for external optimizers only."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bo_benchmark/workers'))
from scientific_worker import Service, fingerprint, main
from bo_benchmark_app.baseline_adapters import BaselineAdapter


def initialize(self, payload):
    if self.adapter is not None or self.role != 'optimizer':
        raise ValueError('Baseline worker requires an uninitialized optimizer role')
    self.adapter = BaselineAdapter(payload['task'], payload['seed'], payload['options'],
                                   os.environ['BO_BASELINE'])
    self.columns = [p['name'] for p in payload['task']['parameters']]
    self.root = Path(payload['checkpoint_root']).resolve()
    self.identity = fingerprint({k: v for k, v in payload.items() if k != 'checkpoint_root'}, self.adapter)
    digest = hashlib.sha256(self.identity['code'].encode())
    for name in ('baseline_adapters.py', 'baseline_worker.py'):
        digest.update(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n'))
    self.identity['code'] = digest.hexdigest()
    for name in ('HEBO', 'optuna', 'scipy', 'scikit-learn', 'GPy', 'catboost',
                 'pymoo', 'linear-operator', 'paramz'):
        try:
            self.identity['dependencies'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'identity': self.identity}


if __name__ == '__main__':
    Service.initialize = initialize
    raise SystemExit(main())
