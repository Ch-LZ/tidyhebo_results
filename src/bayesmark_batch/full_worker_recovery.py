"""Full Bayesmark worker with a scoped MLP-SGD regression failure policy."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bayesmark_batch import full_worker as legacy, mlp_failure
from bayesmark_batch.full_suite import TASKS


class Oracle(legacy.Oracle):
    def __init__(self, name, seed):
        super().__init__(name, seed)
        self.name = name
        self.penalty = mlp_failure.failure_penalty(self.test.data_y, TASKS[name][2])
        self.metadata['numerical_failure_policy'] = {
            'id': 'mlp_sgd_regression_v1',
            'cv_penalty': self.penalty,
            'holdout_failure': 'missing; leave CV loss unchanged',
        }

    def evaluate(self, X):
        import numpy as np
        x = np.asarray(X,dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(self.columns) or not len(x)
                or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any()):
            raise ValueError('Invalid Bayesmark unit coordinates')
        lo, hi = self.bounds.T
        params = self.space.unwarp(lo + x * (hi - lo))
        start = sum(map(len,self.held_out))
        scores, held_out = [], []
        previous = np.random.get_state()
        try:
            for i, config in enumerate(params,start):
                np.random.seed(np.random.SeedSequence([self.task_key,self.seed,i]).generate_state(1)[0])
                score, test, failure = mlp_failure.evaluate(self.test, config, self.penalty)
                if failure:
                    print(f'BAYESMARK_NUMERICAL_FAILURE task={self.name} seed={self.seed} '
                          f'eval={i} kind={failure} cv_loss={score!r} config={config!r}',
                          file=sys.stderr, flush=True)
                scores.append(score)
                held_out.append(test)
        finally:
            np.random.set_state(previous)
        if not np.isfinite(scores).all() or not all(v is None or np.isfinite(v) for v in held_out):
            raise ValueError('Non-finite Bayesmark loss')
        self.held_out.append(held_out)
        return np.asarray(scores)[:,None], np.asarray([[float(p[k]) for k in self.columns] for p in params])


class Worker(legacy.Worker):
    def initialize(self, payload):
        name = payload['name']
        task = TASKS.get(name)
        if task is None or task[0] != 'MLP-sgd' or task[2] not in ('mae', 'mse'):
            return super().initialize(payload)
        if self.role != 'benchmark' or payload['options'].get('benchmark') != 'bayesmark_full':
            raise ValueError('Expected full Bayesmark benchmark worker')
        if self.adapter is not None:
            raise ValueError('Already initialized')
        self.adapter = Oracle(name, payload['seed'])
        self.root = Path(payload['checkpoint_root']).resolve()
        self.identity = legacy.service.fingerprint(
            {key: val for key, val in payload.items() if key != 'checkpoint_root'}, self.adapter)
        digest = hashlib.sha256(self.identity['code'].encode())
        for path in (Path(__file__), Path(legacy.__file__), HERE / 'full_suite.py',
                     HERE / 'mlp_failure.py'):
            digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
        self.identity['code'] = digest.hexdigest()
        return {'identity': self.identity}


if __name__ == '__main__':
    legacy.service.Service = Worker
    raise SystemExit(legacy.service.main())
