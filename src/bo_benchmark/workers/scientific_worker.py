"""Standalone JSON-lines service. Stdlib outer shell supports Python 3.7+."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import tempfile
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = sys.stdout.buffer
MAX_LINE = 8 * 1024 * 1024


def send(value):
    OUT.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    OUT.flush()


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def fingerprint(configuration, adapter):
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    for name in ('seeds.py', 'task_catalog.py', 'assets/olympus_tasks.json'):
        digest.update((Path(__file__).parent.parent/name).read_bytes().replace(b'\r\n', b'\n'))
    deps = {}
    for name in ('numpy', 'pandas', 'torch', 'botorch', 'gpytorch', 'pymoo', 'tensorflow'):
        module = sys.modules.get(name)
        if module is not None: deps[name] = str(getattr(module, '__version__', 'unknown'))
    if 'tidyhebo' in sys.modules:
        root = Path(sys.modules['tidyhebo'].__file__).parent
        for path in sorted(root.rglob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode('utf-8'))
            digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return {'python': platform.python_version(), 'code': digest.hexdigest(), 'dependencies': deps,
            'configuration': hashlib.sha256(json.dumps(configuration, sort_keys=True,
                allow_nan=False).encode('utf-8')).hexdigest(),
            'adapter': getattr(adapter, 'metadata', {})}


class Service:
    def __init__(self, role):
        self.role, self.adapter, self.batch = role, None, 0
        self.root = self.identity = self.columns = None

    def initialize(self, payload):
        if self.adapter is not None: raise ValueError('Already initialized')
        options, name, seed = payload['options'], payload['name'], payload['seed']
        if self.role == 'optimizer':
            from modern_adapters import TidyHEBOAdapter, RandomAdapter
            adapter = {'tidyhebo': TidyHEBOAdapter, 'random': RandomAdapter}[name]
            if payload['task'].get('metadata',{}).get('kind') == 'table':
                from niah_adapters import FiniteTidyHEBOAdapter, FiniteRandomAdapter
                adapter = {'tidyhebo':FiniteTidyHEBOAdapter,'random':FiniteRandomAdapter}[name]
            self.adapter = adapter(payload['task'], seed, options)
            self.columns = [p['name'] for p in payload['task']['parameters']]
        elif options.get('benchmark') == 'niah':
            from niah_benchmark import NIAHBenchmark
            self.adapter = NIAHBenchmark(name,seed,options)
        elif options.get('benchmark', 'synthetic') == 'olympus':
            from olympus_adapter import OlympusBenchmark
            self.adapter = OlympusBenchmark(name, seed, options)
        else:
            from modern_adapters import SyntheticBenchmark
            self.adapter = SyntheticBenchmark(name, seed, options)
        self.identity = fingerprint({k: v for k, v in payload.items() if k != 'checkpoint_root'}, self.adapter)
        self.root = Path(payload['checkpoint_root']).resolve()
        return {'identity': self.identity}

    def save(self):
        directory = self.root / self.role
        directory.mkdir(parents=True, exist_ok=True)
        state = self.adapter.save_state()
        data = pickle.dumps({'batch': self.batch, 'state': state}, protocol=4)
        name = uuid.uuid4().hex + '.pkl'
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=str(directory))
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, str(directory / name))
            if os.name == 'posix':
                fd = os.open(str(directory), os.O_RDONLY)
                try: os.fsync(fd)
                finally: os.close(fd)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        return {'kind': 'worker-native', 'schema_version': 1, 'format': 'native-pickle-v1',
                'path': self.role + '/' + name, 'sha256': hashlib.sha256(data).hexdigest(),
                'compatibility': self.identity, 'batch': self.batch,
                'diagnostics': self.adapter.diagnostics() if hasattr(self.adapter, 'diagnostics') else {}}

    def load(self, descriptor):
        if (descriptor.get('compatibility') != self.identity or descriptor.get('format') != 'native-pickle-v1'
                or descriptor.get('schema_version') != 1 or descriptor.get('kind') != 'worker-native'):
            raise ValueError('Native worker code/dependencies/configuration changed')
        relative = Path(descriptor['path'])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or path.parent != (self.root / self.role).resolve():
            raise ValueError('Native path outside worker state directory')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
            raise ValueError('Native checksum mismatch')
        record = pickle.loads(data)  # Only this worker's trusted, checked local native artifacts.
        if record['batch'] != descriptor['batch']: raise ValueError('Native batch mismatch')
        self.adapter.load_state(record['state'])
        self.batch = record['batch']

    def handle(self, command, payload):
        if command == 'initialize': return self.initialize(payload)
        if self.adapter is None: raise ValueError('Not initialized')
        if command == 'save_state': return self.save()
        if command == 'load_state': return self.load(payload)
        if command == 'describe' and self.role == 'benchmark': return self.adapter.description
        if command == 'suggest' and self.role == 'optimizer':
            frame = self.adapter.suggest(payload['n'])
            if list(frame.columns) != self.columns or len(frame) != payload['n']:
                raise ValueError('Optimizer violated DataFrame contract')
            return {'columns': list(frame.columns), 'data': frame.values.tolist()}
        if command == 'observe' and self.role == 'optimizer':
            import pandas as pd
            import numpy as np
            X, y = payload['X'], payload['y']
            if X['columns'] != self.columns or y['shape'] != [len(X['data']), 1]:
                raise ValueError('Observation columns/shape mismatch')
            frame = pd.DataFrame(X['data'], columns=X['columns'])
            values = np.asarray(y['data'])
            if 'physical_X' in payload and hasattr(self.adapter,'observe_evaluated'):
                result = self.adapter.observe_evaluated(frame, values, payload['physical_X'])
            else:
                result = self.adapter.observe(frame, values)
            if result is not None: raise ValueError('observe must return None')
            self.batch += 1
            return None
        if command == 'evaluate' and self.role == 'benchmark':
            if payload['X']['columns'] != [p['name'] for p in self.adapter.description['parameters']]:
                raise ValueError('Benchmark columns mismatch')
            y, physical = self.adapter.evaluate(payload['X']['data'])
            self.batch += 1
            return {'shape': list(y.shape), 'data': y.tolist(), 'physical_X': physical.tolist()}
        raise ValueError('Unknown command: ' + command)


def main():
    service = Service(sys.argv[1])
    send({'protocol': 1, 'event': 'ready', 'role': service.role,
          'pid': os.getpid(), 'python': platform.python_version()})
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line: return 0
        request_id = None
        try:
            if len(line) > MAX_LINE or not line.endswith(b'\n'): raise ValueError('Invalid line length')
            request = json.loads(line.decode('utf-8'), parse_constant=reject_constant)
            request_id = request['request_id']
            if request.get('protocol') != 1 or type(request_id) is not int: raise ValueError('Invalid envelope')
            if request['command'] == 'close': return 0
            with contextlib.redirect_stdout(sys.stderr):
                result = service.handle(request['command'], request['payload'])
            send({'protocol': 1, 'request_id': request_id, 'ok': True, 'result': result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({'protocol': 1, 'request_id': request_id, 'ok': False,
                  'error': type(exc).__name__ + ': ' + str(exc)})
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
