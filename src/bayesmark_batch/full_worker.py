"""Native worker for the official public Bayesmark objective implementation."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'bo_benchmark/workers'), str(HERE.parent)]
import scientific_worker as service
from bayesmark_batch.full_suite import SHA256, TASKS

class Oracle:
    def __init__(self, name, seed):
        import numpy as np
        import sklearn
        from bayesmark.sklearn_funcs import SklearnModel
        from bayesmark.space import JointSpace

        self.seed = seed
        self.task_key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        self.test = SklearnModel(*TASKS[name], shuffle_seed=0)
        config = self.test.get_api_config()
        self.columns = sorted(config)
        self.space = JointSpace(config)
        self.bounds = self.space.get_bounds()
        if self.bounds.shape != (len(self.columns), 2):
            raise ValueError(f'Expected scalar Bayesmark parameters: {name}')
        digest = hashlib.sha256()
        for x in (self.test.data_X, self.test.data_Xt, self.test.data_y, self.test.data_yt):
            digest.update(str(x.shape).encode())
            digest.update(np.asarray(x, dtype='<f8', order='C').tobytes())
        self.metadata = dict(benchmark='bayesmark_full', task=name, bayesmark='0.0.8',
                             bayesmark_sha256=SHA256, sklearn=sklearn.__version__,
                             task_assets_sha256=digest.hexdigest(), split_seed=0,
                             metric=TASKS[name][2], physical_columns=self.columns,
                             learner_seed_policy='seedsequence(task, run seed, zero-based evaluation)')
        self.description = {'parameters':[dict(name=p,lower=0.,upper=1.) for p in self.columns],
                            'direction':'minimize','metadata':self.metadata}
        self.held_out = []

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
                score, test = self.test.evaluate(config)
                scores.append(score)
                held_out.append(test)
        finally:
            np.random.set_state(previous)
        if not np.isfinite([scores,held_out]).all():
            raise ValueError('Non-finite Bayesmark loss')
        self.held_out.append(held_out)
        return np.asarray(scores)[:,None], np.asarray([[float(p[k]) for k in self.columns] for p in params])

    def save_state(self):
        return {'held_out':self.held_out}

    def load_state(self,state):
        self.held_out = state['held_out']

class Worker(service.Service):
    def initialize(self,payload):
        if self.role != 'benchmark' or payload['options'].get('benchmark') != 'bayesmark_full':
            raise ValueError('Expected full Bayesmark benchmark worker')
        if self.adapter is not None:
            raise ValueError('Already initialized')
        self.adapter = Oracle(payload['name'],payload['seed'])
        self.root = Path(payload['checkpoint_root']).resolve()
        self.identity = service.fingerprint(
            {key:val for key,val in payload.items() if key != 'checkpoint_root'},self.adapter)
        digest = hashlib.sha256(self.identity['code'].encode())
        for path in (Path(__file__), HERE / 'full_suite.py'):
            digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
        self.identity['code'] = digest.hexdigest()
        return {'identity':self.identity}

def check():
    import warnings
    import numpy as np
    import sklearn
    from bayesmark.constants import DATA_LOADER_NAMES, MODEL_NAMES, SCORERS_CLF, SCORERS_REG
    if (np.__version__,sklearn.__version__) != ('1.23.5','1.1.3'):
        raise RuntimeError('Unexpected legacy NumPy or scikit-learn version')
    if (set(DATA_LOADER_NAMES) != {v[1] for v in TASKS.values()}
            or set(MODEL_NAMES) != {v[0] for v in TASKS.values()}
            or set(SCORERS_CLF) != {'acc','nll'} or set(SCORERS_REG) != {'mae','mse'}
            or len(TASKS) != 108):
        raise RuntimeError('Bayesmark 0.0.8 task inventory differs')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=UserWarning)
        for task in TASKS:
            oracle = Oracle(task,42)
            oracle.evaluate([[.5] * len(oracle.columns)])
    print('Official Bayesmark oracle checked at 108/108 public tasks.',flush=True)

if __name__ == '__main__':
    if sys.argv[1] == '--check':
        check()
    else:
        service.Service = Worker
        raise SystemExit(service.main())
