"""Local clients for cheap doubles only; no cross-environment transport.

Calls execute synchronously after yielding. This wrapper must not be used to
run blocking GP fits on the controller event loop. A production Runtime is a
separate design step.
"""
import asyncio
import inspect
from .domain import positive_int
from .optimizer import Checkpointable, Optimizer
from .validation import normalize_observations, validate_X


class LocalOptimizerClient:
    def __init__(self, optimizer: Optimizer, columns, on_event=None, on_observe=None):
        for name in ('suggest', 'observe'):
            method = getattr(optimizer, name, None)
            if not callable(method) or inspect.iscoroutinefunction(method):
                raise TypeError('Optimizer must provide synchronous suggest and observe')
        self.optimizer, self.columns = optimizer, tuple(columns)
        self.on_event, self.on_observe = on_event, on_observe
        self.closed = False

    def _check_open(self):
        if self.closed:
            raise RuntimeError('Optimizer client is closed')

    def _record(self, action):
        if self.on_event:
            self.on_event(action)

    async def suggest(self, n=1):
        self._check_open()
        positive_int(n)
        await asyncio.sleep(0)
        self._record('suggest')
        return validate_X(self.optimizer.suggest(n), self.columns, n)

    async def observe(self, X, y):
        self._check_open()
        X, y = normalize_observations(X, y, self.columns)
        # Retain an independent copy for observers; user code may mutate inputs.
        observed_X, observed_y = X.copy(deep=True), y.copy()
        result = self.optimizer.observe(X, y)
        if result is not None:
            raise TypeError('observe must return None')
        self._record('observe')
        if self.on_observe:
            self.on_observe(observed_X, observed_y)

    async def close(self):
        if not self.closed:
            self.closed = True
            self._record('close')


class NativeLocalOptimizerClient(LocalOptimizerClient):
    async def save_state(self):
        self._check_open()
        return self.optimizer.save_state()

    async def load_state(self, state):
        self._check_open()
        self.optimizer.load_state(state)
        self._record('restore')


def local_client(optimizer, columns, **kwargs):
    cls = NativeLocalOptimizerClient if isinstance(optimizer, Checkpointable) else LocalOptimizerClient
    return cls(optimizer, columns, **kwargs)
