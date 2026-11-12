"""Dependency-free value objects shared by controller and adapters.

First slice: bounded real parameters and one scalar objective. This is not yet
the complete mixed-space/multiobjective schema required by the final project.
"""
from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, Mapping

Point = tuple[float, ...]


def safe_name(value: str) -> None:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Expected a safe nonempty identifier: {value!r}')


def positive_int(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError('Expected a positive integer')


@dataclass(frozen=True)
class RealParameter:
    name: str
    lower: float
    upper: float

    def __post_init__(self):
        safe_name(self.name)
        if not math.isfinite(self.lower) or not math.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError('Expected finite lower < upper')


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[RealParameter, ...]

    def __post_init__(self):
        if not self.parameters or len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError('Expected nonempty space with unique parameter names')

    def validate(self, points: tuple[Point, ...]) -> None:
        if not points:
            raise ValueError('Empty candidate batch')
        for point in points:
            if len(point) != len(self.parameters):
                raise ValueError('Point dimension does not match the space')
            if any(not math.isfinite(x) or not p.lower <= x <= p.upper
                   for x, p in zip(point, self.parameters)):
                raise ValueError('Candidate is nonfinite or outside task bounds')


@dataclass(frozen=True)
class TaskDescription:
    space: SearchSpace
    direction: Literal['minimize', 'maximize'] = 'minimize'
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.direction not in ('minimize', 'maximize'):
            raise ValueError('Unknown optimization direction')


@dataclass(frozen=True)
class CandidateBatch:
    request_id: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class EvaluationBatch:
    candidates: CandidateBatch
    values: tuple[float, ...]
    physical_points: tuple[Point, ...] | None = None
    seconds: float | None = field(default=None, compare=False)
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if len(self.values) != len(self.candidates.points) or not self.values:
            raise ValueError('One scalar value is required for every candidate')
        if not all(math.isfinite(y) for y in self.values):
            raise ValueError('Nonfinite objective')


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    environment: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        safe_name(self.name)
        safe_name(self.environment)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    environment: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        safe_name(self.name)
        safe_name(self.environment)


@dataclass(frozen=True)
class ProtocolSpec:
    budget: int
    q: int = 1
    startup: int = 0
    startup_batch: bool = False

    def __post_init__(self):
        positive_int(self.budget)
        positive_int(self.q)
        if type(self.startup) is not int or not 0 <= self.startup <= self.budget:
            raise ValueError('startup must be within the evaluation budget')
        if type(self.startup_batch) is not bool or (self.startup_batch and self.startup == 0):
            raise ValueError('startup_batch requires a positive startup budget')


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    optimizer: OptimizerSpec
    task: TaskSpec
    protocol: ProtocolSpec
    seed: int

    def __post_init__(self):
        safe_name(self.run_id)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Expected a uint32 seed')


@dataclass(frozen=True)
class RunCheckpoint:
    spec: RunSpec
    batches: tuple[EvaluationBatch, ...]
    optimizer_state: Any = None
    restore_mode: Literal['native', 'history_only'] = 'history_only'
    direction: Literal['minimize', 'maximize'] = 'minimize'
    schema_version: int = 3
    benchmark_state: Any = None
    in_flight: bool = False


@dataclass(frozen=True)
class RunOutcome:
    spec: RunSpec
    status: Literal['completed', 'paused', 'pending', 'failed']
    batches: tuple[EvaluationBatch, ...]
    error: str | None = None
    direction: Literal['minimize', 'maximize'] | None = None

    @property
    def evaluations(self) -> int:
        return sum(len(batch.values) for batch in self.batches)
