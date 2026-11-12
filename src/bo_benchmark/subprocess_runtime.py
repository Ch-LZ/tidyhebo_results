"""Real subprocess clients and explicitly toy Runtime (no environment installer)."""
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd

from .domain import (CandidateBatch, EvaluationBatch, RealParameter, SearchSpace,
                     TaskDescription, positive_int)
from .transport import JsonLinesChannel, TransportError, WorkerLaunch
from .validation import normalize_observations, validate_X


def task_payload(task):
    return {'parameters': [asdict(p) for p in task.space.parameters],
            'direction': task.direction, 'metadata': dict(task.metadata)}


class SubprocessOptimizerClient:
    """Command handlers run synchronously in worker; this client waits asynchronously.

    No native state extension yet: Controller keeps history-only checkpoints.
    """
    def __init__(self, channel, columns):
        self.channel, self.columns = channel, tuple(columns)

    async def suggest(self, n=1):
        positive_int(n)
        result = await self.channel.request('suggest', {'n': n})
        if not isinstance(result, dict) or set(result) != {'columns', 'data'}:
            raise TransportError('Expected columns/data table')
        return validate_X(pd.DataFrame(result['data'], columns=result['columns']),
                          columns=self.columns, n=n)

    async def observe(self, X, y):
        await self._observe(X, y)

    async def observe_evaluated(self, X, y, physical_X):
        await self._observe(X, y, physical_X)

    async def _observe(self, X, y, physical_X=None):
        X, y = normalize_observations(X, y, columns=self.columns)
        payload = {'X': {'columns': list(X.columns), 'data': X.values.tolist()},
                   'y': {'shape': list(y.shape), 'data': y.tolist()}}
        if physical_X is not None:
            physical = np.asarray(physical_X, dtype=float)
            if physical.ndim != 2 or len(physical) != len(X) or not np.isfinite(physical).all():
                raise ValueError('Invalid evaluated physical points')
            payload['physical_X'] = physical.tolist()
        result = await self.channel.request('observe', payload)
        if result is not None:
            raise TransportError('observe must return null')

    async def close(self):
        await self.channel.close()


class SubprocessBenchmarkClient:
    def __init__(self, channel):
        self.channel = channel
        self._task = None

    async def describe(self):
        if self._task is None:
            result = await self.channel.request('describe')
            self._task = TaskDescription(SearchSpace(tuple(
                RealParameter(**p) for p in result['parameters'])), result['direction'], result.get('metadata', {}))
        return self._task

    async def evaluate(self, batch: CandidateBatch):
        task = await self.describe()
        task.space.validate(batch.points)
        result = await self.channel.request('evaluate', {'X': {
            'columns': [p.name for p in task.space.parameters],
            'data': [list(point) for point in batch.points]}})
        if (not isinstance(result, dict) or result.get('shape') != [len(batch.points), 1]
                or not isinstance(result.get('data'), list)
                or len(result['data']) != len(batch.points)
                or any(not isinstance(row, list) or len(row) != 1 for row in result['data'])):
            raise TransportError('Expected y shape (n,1)')
        values = np.asarray(result['data'])
        if values.dtype.kind not in 'iuf':
            raise TransportError('Expected numeric y')
        physical = result.get('physical_X')
        if physical is not None:
            physical = tuple(tuple(float(x) for x in row) for row in physical)
            if len(physical) != len(batch.points) or not all(np.isfinite(row).all() for row in physical):
                raise TransportError('Invalid physical_X')
        return EvaluationBatch(batch, tuple(float(x) for x in values[:, 0]), physical)

    async def close(self):
        await self.channel.close()


class NativeStateClient:
    """Only descriptors cross the pipe; native bytes are handled by the worker."""
    async def save_state(self):
        descriptor = await self.channel.request('save_state')
        if (not isinstance(descriptor, dict) or descriptor.get('kind') != 'worker-native'
                or type(descriptor.get('batch')) is not int or descriptor['batch'] < 0):
            raise TransportError('Expected native checkpoint descriptor')
        return descriptor

    async def load_state(self, state):
        result = await self.channel.request('load_state', state)
        if result is not None:
            raise TransportError('load_state must return null')


class NativeSubprocessOptimizerClient(NativeStateClient, SubprocessOptimizerClient):
    pass


class NativeSubprocessBenchmarkClient(NativeStateClient, SubprocessBenchmarkClient):
    pass


class ToySubprocessRuntime:
    """One dedicated optimizer + benchmark process per active run.

    Ready-made Python environments only. Custom launcher gets role, environment
    path, spec; use it to select executable/env without modifying os.environ.
    Built-in launcher supports ONLY the bundled stdlib toy worker.
    """
    def __init__(self, launcher: Callable | None = None, checkpoint_root: Path | None = None):
        self.launcher = launcher or self._default_launch
        self.checkpoint_root = Path(checkpoint_root).resolve() if checkpoint_root is not None else None
        self.channels = []  # Diagnostic handles, including already closed sessions.

    @staticmethod
    def _default_launch(role, environment, spec):
        python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        if not python.is_file():
            raise FileNotFoundError(f'No ready Python at {python}; supply a launcher or create a venv')
        worker = Path(__file__).with_name('workers') / 'toy_worker.py'
        return WorkerLaunch((str(python.absolute()), '-u', str(worker), role))

    async def _open(self, role, spec, environment, seed, task=None):
        channel = await JsonLinesChannel.open(self.launcher(role, environment, spec))
        self.channels.append(channel)
        try:
            if channel.ready.get('role') != role:
                raise TransportError('Worker role mismatch')
            await channel.request('initialize', {'seed': seed, 'options': dict(spec.options),
                                                 'task': task_payload(task) if task else None,
                                                 'checkpoint_root': str(self.checkpoint_root) if self.checkpoint_root else None})
            return channel
        except BaseException:
            await channel.close()
            raise

    async def open_optimizer(self, spec, environment, task, seed):
        channel = await self._open('optimizer', spec, environment, seed, task)
        client = NativeSubprocessOptimizerClient if self.checkpoint_root else SubprocessOptimizerClient
        return client(channel, [p.name for p in task.space.parameters])

    async def open_benchmark(self, spec, environment, seed):
        client = NativeSubprocessBenchmarkClient if self.checkpoint_root else SubprocessBenchmarkClient
        return client(await self._open('benchmark', spec, environment, seed))
