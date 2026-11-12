"""One event loop and one controller process for every run in an executor.

Runs are state machines advanced one completed batch at a time. There is no
per-run scheduler/BO-loop process. Async clients must yield while external work
executes; a real CPU-bound adapter must not compute inside this event loop.
"""
import asyncio
import math
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, replace
import time
import numpy as np
from .domain import CandidateBatch, EvaluationBatch, RunCheckpoint, RunOutcome, RunSpec, TaskDescription, positive_int
from .ports import Benchmark, NativeBenchmarkClient, NativeOptimizerClient, OptimizerClient, PhysicalObservationClient, RunStore, Runtime
from .validation import validate_X
from .workspace import Workspace
from .control import ControlFile


@dataclass
class _RunState:
    spec: RunSpec
    optimizer: OptimizerClient | None = None
    benchmark: Benchmark | None = None
    description: TaskDescription | None = None
    batches: list[EvaluationBatch] = field(default_factory=list)
    saved_direction: str | None = None

    @property
    def direction(self):
        return self.description.direction if self.description else self.saved_direction

    @property
    def evaluations(self):
        return sum(len(b.values) for b in self.batches)


class Controller:
    def __init__(self, runtime: Runtime, store: RunStore, workspace: Workspace,
                 max_parallel: int = 2, control_file: Path | None = None,
                 poll_interval: float = .1):
        positive_int(max_parallel)
        self.runtime, self.store, self.workspace = runtime, store, workspace
        self.max_parallel = max_parallel
        if isinstance(poll_interval, bool) or not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError('poll_interval must be finite and positive')
        self.poll_interval = poll_interval
        self.control = ControlFile(control_file) if control_file is not None else None
        self._running = False

    @property
    def max_parallel(self):
        return self._max_parallel

    @max_parallel.setter
    def max_parallel(self, value):
        positive_int(value)
        self._max_parallel = value

    def _save_history(self, state, optimizer_state=None, restore_mode='history_only', benchmark_state=None, in_flight=False):
        if restore_mode == 'native':
            for descriptor in (optimizer_state, benchmark_state):
                if (isinstance(descriptor, dict) and descriptor.get('kind') == 'worker-native'
                        and descriptor.get('batch') != len(state.batches)):
                    raise ValueError('Native checkpoint batch mismatch')
        self.store.save_checkpoint(RunCheckpoint(
            state.spec, tuple(state.batches), optimizer_state,
            restore_mode=restore_mode, direction=state.description.direction,
            benchmark_state=benchmark_state, in_flight=in_flight))

    async def _advance(self, state: _RunState) -> None:
        spec = state.spec
        if state.optimizer is None:
            checkpoint = self.store.checkpoint(spec.run_id)
            if checkpoint:
                if checkpoint.spec != spec or checkpoint.schema_version != 3:
                    raise ValueError('Incompatible run checkpoint')
                state.batches = list(checkpoint.batches)
                if checkpoint.in_flight:
                    raise ValueError('Interrupted in-flight batch; outcome unknown, automatic replay is disabled')
                if checkpoint.restore_mode != 'native':
                    raise ValueError('History is preserved; history-only resume policy is not implemented')
            state.benchmark = await self.runtime.open_benchmark(
                spec.task, self.workspace.environment('benchmarks', spec.task.environment), spec.seed)
            description = await state.benchmark.describe()
            state.description = description
            state.optimizer = await self.runtime.open_optimizer(
                spec.optimizer, self.workspace.environment('optimizers', spec.optimizer.environment),
                description, spec.seed)
            if checkpoint:
                if checkpoint.direction != description.direction:
                    raise ValueError('Checkpoint objective direction changed')
                if not isinstance(state.optimizer, NativeOptimizerClient):
                    raise ValueError('Optimizer does not support native restore; history is preserved')
                if not isinstance(state.benchmark, NativeBenchmarkClient):
                    raise ValueError('Benchmark does not support native restore; history is preserved')
                await state.benchmark.load_state(checkpoint.benchmark_state)
                await state.optimizer.load_state(checkpoint.optimizer_state)
            # Both native states refer to the same completed batch boundary.
            # There is no shared external initialization: each optimizer owns startup.
        remaining = spec.protocol.budget-state.evaluations
        if remaining <= 0:
            return
        n = min(spec.protocol.q, remaining)
        if state.evaluations < spec.protocol.startup:
            initial_remaining = spec.protocol.startup - state.evaluations
            n = min(remaining, initial_remaining,
                    initial_remaining if spec.protocol.startup_batch else spec.protocol.q)
        started = time.monotonic()
        request_id = f'{spec.run_id}-batch-{len(state.batches):04d}'
        columns = [p.name for p in state.description.space.parameters]
        # Persist uncertainty before even suggest: it can advance native RNG/state.
        self._save_history(state, in_flight=True)
        frame = validate_X(await state.optimizer.suggest(n), columns, n)
        candidates = CandidateBatch(request_id, tuple(
            tuple(float(x) for x in row) for row in frame.to_numpy()))
        state.description.space.validate(candidates.points)
        batch = await state.benchmark.evaluate(candidates)
        if batch.candidates != candidates:
            raise ValueError('Benchmark returned a mismatched batch')
        state.batches.append(batch)
        # Preserve an expensive result even if observe or native snapshot fails.
        # Such an intermediate snapshot is explicitly history-only, not native.
        self._save_history(state)
        y = np.asarray(batch.values).reshape(-1, 1)
        if state.description.direction == 'minimize':
            y = -y
        if batch.physical_points is not None and isinstance(state.optimizer, PhysicalObservationClient):
            await state.optimizer.observe_evaluated(frame, y, batch.physical_points)
        else:
            await state.optimizer.observe(frame, y)
        if (isinstance(state.optimizer, NativeOptimizerClient)
                and isinstance(state.benchmark, NativeBenchmarkClient)):
            snapshot = await state.optimizer.save_state()
            benchmark_snapshot = await state.benchmark.save_state()
            diagnostics = snapshot.get('diagnostics', {}) if isinstance(snapshot, dict) else {}
            state.batches[-1] = replace(batch, seconds=time.monotonic()-started, diagnostics=diagnostics)
            self._save_history(state, snapshot, 'native', benchmark_snapshot)

    async def _close(self, state: _RunState) -> None:
        # Both clients must receive cleanup even when one of them fails.
        clients = [x for x in (state.optimizer, state.benchmark) if x is not None]
        results = await asyncio.gather(*(x.close() for x in clients), return_exceptions=True)
        state.optimizer = state.benchmark = None
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise RuntimeError(f'Adapter cleanup failed: {errors!r}')

    async def run(self, specs: list[RunSpec], stop: asyncio.Event | None = None) -> dict[str, RunOutcome]:
        if self._running:
            raise RuntimeError('Controller.run is already active')
        if len({s.run_id for s in specs}) != len(specs):
            raise ValueError('Duplicate run IDs')
        self._running = True
        states, outcomes, active = {}, {}, {}
        lease = None
        try:
            self.workspace.prepare()
            lease = self.workspace.controller_lease()
            lease.__enter__()
            for spec in specs:
                previous = self.store.outcome(spec.run_id)
                if previous and previous.spec != spec:
                    raise ValueError('Run ID already belongs to another specification')
                if previous and previous.status == 'completed':
                    outcomes[spec.run_id] = previous
                else:
                    state = _RunState(spec)
                    checkpoint = self.store.checkpoint(spec.run_id)
                    if checkpoint:
                        if checkpoint.spec != spec or checkpoint.schema_version != 3:
                            raise ValueError('Incompatible run checkpoint')
                        state.batches = list(checkpoint.batches)
                        state.saved_direction = checkpoint.direction
                    states[spec.run_id] = state
            pending, ready, live = deque(states.values()), deque(), set()
            paused = False
            while pending or ready or active:
                if self.control is not None:
                    self.max_parallel = self.control.read_limit(self.max_parallel)
                # A pause is latched for this invocation: remove STOP and run again to resume.
                paused = paused or self.workspace.stop_file.exists() or (stop is not None and stop.is_set())
                while pending and len(live) < self.max_parallel and not paused:
                    state = pending.popleft()
                    live.add(state.spec.run_id)
                    ready.append(state)
                while ready and not paused:
                    state = ready.popleft()
                    active[asyncio.create_task(self._advance(state))] = state
                if not active:
                    break
                done, _ = await asyncio.wait(active, timeout=self.poll_interval,
                                             return_when=asyncio.FIRST_COMPLETED)
                # Stable handling when several adapter replies arrive together.
                for future in sorted(done, key=lambda f: active[f].spec.run_id):
                    state = active.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        live.remove(state.spec.run_id)
                        outcomes[state.spec.run_id] = RunOutcome(
                            state.spec, 'failed', tuple(state.batches), str(error),
                            direction=state.direction)
                    else:
                        if state.evaluations == state.spec.protocol.budget:
                            live.remove(state.spec.run_id)
                            outcomes[state.spec.run_id] = RunOutcome(
                                state.spec, 'completed', tuple(state.batches),
                                direction=state.description.direction)
                        else:
                            ready.append(state)
                    if state.spec.run_id in outcomes:
                        try:
                            await self._close(state)
                        except Exception as error:
                            previous = outcomes[state.spec.run_id]
                            detail = '; '.join(x for x in (previous.error, str(error)) if x)
                            outcomes[state.spec.run_id] = RunOutcome(
                                state.spec, 'failed', tuple(state.batches), detail,
                                direction=state.direction)
                        self.store.save_outcome(outcomes[state.spec.run_id])
            for state in [*pending, *ready]:
                outcomes[state.spec.run_id] = RunOutcome(
                    state.spec, 'paused' if state.optimizer is not None or state.batches else 'pending', tuple(state.batches),
                    direction=state.direction)
                self.store.save_outcome(outcomes[state.spec.run_id])
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Keep uncertainty marker and last committed state; never blindly replay an unknown batch.
            self.workspace.stop_file.touch()
            raise
        finally:
            for future in active:
                future.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            cleanup = await asyncio.gather(*(self._close(s) for s in states.values()), return_exceptions=True)
            if lease is not None and lease.handle is not None:
                lease.__exit__(None, None, None)
            self._running = False
            errors = [e for e in cleanup if isinstance(e, BaseException)]
            if errors:
                raise RuntimeError(f'Controller cleanup failed: {errors!r}')
        return outcomes
