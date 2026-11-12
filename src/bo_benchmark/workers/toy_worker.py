"""Standalone stdlib worker, Python 3.7+. Never imports the host package.

Toy command handlers exercise IPC only; there is no BO loop in this process.
stdout is reserved for protocol; Python handler prints go to stderr.
"""
import contextlib
import json
import math
import os
import random
import sys
import traceback
import native_state

PROTOCOL_OUT = sys.stdout.buffer
MAX_LINE = 8 * 1024 * 1024


def send(message):
    PROTOCOL_OUT.write((json.dumps(message, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
    PROTOCOL_OUT.flush()


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


class ToyOptimizer:
    def __init__(self, task, seed):
        self.parameters = task['parameters']
        self.rng = random.Random(seed)
        self.observations = 0

    def suggest(self, n=1):
        if type(n) is not int or n < 1:
            raise ValueError('Expected positive n')
        return {'columns': [p['name'] for p in self.parameters],
                'data': [[self.rng.uniform(p['lower'], p['upper'])
                          for p in self.parameters] for _ in range(n)]}

    def observe(self, X, y):
        n = len(X['data'])
        if X['columns'] != [p['name'] for p in self.parameters]:
            raise ValueError('Wrong observation columns')
        if y['shape'] != [n, 1] or len(y['data']) != n:
            raise ValueError('Wrong observation shape')
        if any(len(row) != 1 or not math.isfinite(row[0]) for row in y['data']):
            raise ValueError('Invalid y')
        self.observations += n


class ToyService:
    def __init__(self, role):
        self.role = role
        self.initialized = False
        self.optimizer = None
        self.task = None
        self.batches = 0
        self.checkpoint_root = None
        self.rng = None
        self.options = {}
        self.compatibility = None

    def handle(self, command, payload):
        if command == 'initialize':
            if self.initialized:
                raise ValueError('Already initialized')
            options = payload['options']
            self.options = options
            self.checkpoint_root = payload.get('checkpoint_root')
            self.compatibility = native_state.identity({'role': self.role,
                'seed': payload['seed'], 'task': payload.get('task'), 'options': options})
            self.rng = random.Random(payload['seed'])
            if options.get('fail_initialize'):
                raise ValueError('Injected initialization error')
            self.task = (payload['task'] if self.role == 'optimizer' else
                         {'parameters': [{'name': 'x', 'lower': -1., 'upper': 1.},
                                         {'name': 'y', 'lower': -1., 'upper': 1.}],
                          'direction': options.get('direction', 'minimize')})
            if self.role == 'optimizer':
                self.optimizer = ToyOptimizer(self.task, payload['seed'])
            self.initialized = True
            return None
        if not self.initialized:
            raise ValueError('Not initialized')
        if command == 'save_state':
            if not self.checkpoint_root:
                raise ValueError('No checkpoint directory configured')
            state = ({'rng': self.optimizer.rng.getstate(),
                      'observations': self.optimizer.observations} if self.role == 'optimizer'
                     else {'rng': self.rng.getstate()})
            return native_state.save(self.checkpoint_root, self.role, self.compatibility,
                                     self.batches, state)
        if command == 'load_state':
            if not self.checkpoint_root:
                raise ValueError('No checkpoint directory configured')
            batch, state = native_state.load(self.checkpoint_root, self.role,
                                            self.compatibility, payload)
            if self.role == 'optimizer':
                self.optimizer.rng.setstate(native_state.tuples(state['rng']))
                self.optimizer.observations = state['observations']
            else:
                self.rng.setstate(native_state.tuples(state['rng']))
            self.batches = batch
            return None
        if command == 'describe' and self.role == 'benchmark':
            return self.task
        if command == 'suggest' and self.role == 'optimizer':
            return self.optimizer.suggest(payload['n'])
        if command == 'observe' and self.role == 'optimizer':
            self.optimizer.observe(payload['X'], payload['y'])
            self.batches += 1
            return None
        if command == 'evaluate' and self.role == 'benchmark':
            X = payload['X']
            if X['columns'] != [p['name'] for p in self.task['parameters']]:
                raise ValueError('Wrong candidate columns')
            values = []
            for row in X['data']:
                if len(row) != len(self.task['parameters']):
                    raise ValueError('Wrong point dimension')
                if any(not math.isfinite(x) or not p['lower'] <= x <= p['upper']
                       for x, p in zip(row, self.task['parameters'])):
                    raise ValueError('Candidate outside bounds')
                value = sum(x*x for x in row)
                if self.options.get('noise_std', 0):
                    value += self.rng.gauss(0., float(self.options['noise_std']))
                values.append([value])
            self.batches += 1
            return {'shape': [len(values), 1], 'data': values}
        raise ValueError('Unknown command for role: ' + command)


def main():
    service = ToyService(sys.argv[1])
    send({'protocol': 1, 'event': 'ready', 'pid': os.getpid(),
          'python': sys.version.split()[0], 'role': service.role})
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line:
            return
        request_id = None
        try:
            if len(line) > MAX_LINE or not line.endswith(b'\n'):
                raise ValueError('Oversized or incomplete line')
            request = json.loads(line.decode('utf-8'), parse_constant=reject_constant)
            request_id = request['request_id']
            if request['protocol'] != 1 or type(request_id) is not int:
                raise ValueError('Invalid request envelope')
            if request['command'] == 'close':
                return
            with contextlib.redirect_stdout(sys.stderr):
                result = service.handle(request['command'], request['payload'])
            send({'protocol': 1, 'request_id': request_id, 'ok': True, 'result': result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({'protocol': 1, 'request_id': request_id, 'ok': False,
                  'error': type(exc).__name__ + ': ' + str(exc)})
            return


if __name__ == '__main__':
    main()
