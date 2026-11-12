"""Small in-process doubles for contract integration; no BO libraries needed.

The asynchronous delays emulate waiting for remote worker replies. They are
not a real subprocess transport or a production optimizer implementation.
"""
import asyncio
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import random
import pandas as pd
from .optimizer import AbstractOptimizer
from .clients import local_client
from typing import Callable
from .domain import (CandidateBatch, EvaluationBatch,
                     RealParameter, RunCheckpoint, RunOutcome, SearchSpace, TaskDescription)


class MemoryStore:
    def __init__(self):
        self.checkpoints = {}
        self.outcomes = {}
        self.committed = {}
        self.writer_pids = []

    def checkpoint(self, run_id: str) -> RunCheckpoint | None:
        return copy.deepcopy(self.checkpoints.get(run_id))

    def committed_checkpoint(self, run_id):
        return copy.deepcopy(self.committed.get(run_id))

    def save_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        if checkpoint.restore_mode == 'native':
            self.committed[checkpoint.spec.run_id] = copy.deepcopy(checkpoint)
        self.writer_pids.append(os.getpid())
        self.checkpoints[checkpoint.spec.run_id] = copy.deepcopy(checkpoint)

    def outcome(self, run_id: str) -> RunOutcome | None:
        return copy.deepcopy(self.outcomes.get(run_id))

    def save_outcome(self, outcome: RunOutcome) -> None:
        self.writer_pids.append(os.getpid())
        self.outcomes[outcome.spec.run_id] = copy.deepcopy(outcome)
        if outcome.status == 'completed':
            self.checkpoints.pop(outcome.spec.run_id, None)
            self.committed.pop(outcome.spec.run_id, None)


@dataclass(frozen=True)
class TraceEvent:
    component: str
    action: str
    pid: int
    environment: Path | None = None


class MockOptimizer(AbstractOptimizer):
    """Ready synchronous object; all random state belongs to this instance."""
    def __init__(self, task, seed, options=None):
        self.task = task
        self.options = options or {}
        super().__init__([p.name for p in task.space.parameters])
        self.rng = random.Random(seed)
        self.observed = []

    def _draw(self, n):
        return pd.DataFrame([[self.rng.uniform(p.lower, p.upper)
            for p in self.task.space.parameters] for _ in range(n)], columns=self.parameter_names)

    def _suggest_one_impl(self):
        return self._draw(1)

    def _suggest_many_impl(self, n):
        if not self.options.get('supports_many', True):
            return super()._suggest_many_impl(n)
        return self._draw(n)

    def _observe_impl(self, X, y):
        self.observed.append((X.copy(deep=True), y.copy()))

    def save_state(self):
        return dict(rng=self.rng.getstate(), observed=copy.deepcopy(self.observed))

    def load_state(self, state):
        self.rng.setstate(state['rng'])
        self.observed = copy.deepcopy(state['observed'])


class MockBenchmark:
    def __init__(self, runtime, name, environment, options):
        self.runtime, self.name, self.environment = runtime, name, environment
        self.options = options
        self.calls = 0
        self.closed = False

    async def describe(self):
        return TaskDescription(SearchSpace((RealParameter('x', -1., 1.),
                                            RealParameter('y', -1., 1.))), self.options.get('direction', 'minimize'))

    async def evaluate(self, batch):
        self.runtime.active_evaluations += 1
        self.runtime.peak_evaluations = max(self.runtime.peak_evaluations, self.runtime.active_evaluations)
        self.runtime.record(self.name, 'evaluate_started')
        try:
            await asyncio.sleep(self.options.get('delay', .001))
            self.calls += 1
            if self.options.get('fail_on_call') == self.calls:
                raise RuntimeError('injected benchmark failure')
            self.runtime.record(self.name, 'evaluate_finished')
            return EvaluationBatch(batch, tuple(sum(x*x for x in point) for point in batch.points))
        finally:
            self.runtime.active_evaluations -= 1

    async def save_state(self):
        return {'calls': self.calls}

    async def load_state(self, state):
        self.calls = state['calls']

    async def close(self):
        self.closed = True
        self.runtime.record(self.name, 'close')


class MockRuntime:
    def __init__(self, on_observe: Callable | None = None, optimizer_factory=None,
                 optimizer_objects=None):
        self.on_observe = on_observe
        self.optimizer_factory = optimizer_factory
        self.optimizer_objects = dict(optimizer_objects or {})
        self._used_objects = []
        self.events = []
        self.optimizers, self.benchmarks = [], []
        self.active_evaluations = self.peak_evaluations = 0
        self.peak_open_optimizers = 0

    def record(self, component, action, environment=None):
        self.events.append(TraceEvent(component, action, os.getpid(), environment))

    async def open_optimizer(self, spec, environment, task, seed):
        self.record(spec.name, 'open_optimizer', environment)
        if spec.name in self.optimizer_objects:
            optimizer = self.optimizer_objects[spec.name]
        elif self.optimizer_factory:
            optimizer = self.optimizer_factory(spec, task, seed)
        else:
            optimizer = MockOptimizer(task, seed, spec.options)
        if any(optimizer is old for old in self._used_objects):
            raise ValueError('Provide a fresh optimizer object for each run/attempt')
        self._used_objects.append(optimizer)
        client = local_client(optimizer, [p.name for p in task.space.parameters],
            on_event=lambda action: self.record(spec.name, action), on_observe=self.on_observe)
        client.name, client.environment = spec.name, environment
        self.optimizers.append(client)
        self.peak_open_optimizers = max(self.peak_open_optimizers,
                                      sum(not x.closed for x in self.optimizers))
        return client

    async def open_benchmark(self, spec, environment, seed):
        self.record(spec.name, 'open_benchmark', environment)
        client = MockBenchmark(self, spec.name, environment, spec.options)
        self.benchmarks.append(client)
        return client
