"""Synchronous mathematical API. Lifecycle and transport belong to Runtime."""
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable
import numpy as np
import pandas as pd
from .domain import positive_int
from .validation import normalize_observations, validate_X

YValues = pd.Series | np.ndarray


class Optimizer(Protocol):
    def suggest(self, n: int = 1) -> pd.DataFrame: ...
    def observe(self, X: pd.DataFrame, y: YValues) -> None: ...


@runtime_checkable
class Checkpointable(Protocol):
    """Optional native state, including the optimizer's own RNG/startup state."""
    def save_state(self) -> Any: ...
    def load_state(self, state: Any) -> None: ...


class AbstractOptimizer(ABC):
    """Optional convenience base; injected objects need not inherit from it."""
    def __init__(self, parameter_names):
        self.parameter_names = tuple(parameter_names)
        if not self.parameter_names or len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError('Expected unique parameter names')

    def suggest(self, n: int = 1) -> pd.DataFrame:
        positive_int(n)
        X = self._suggest_one_impl() if n == 1 else self._suggest_many_impl(n)
        return validate_X(X, self.parameter_names, n)

    def observe(self, X: pd.DataFrame, y: YValues) -> None:
        X, y = normalize_observations(X, y, self.parameter_names)
        self._observe_impl(X, y)

    @abstractmethod
    def _suggest_one_impl(self) -> pd.DataFrame: ...

    def _suggest_many_impl(self, n: int) -> pd.DataFrame:
        raise NotImplementedError('Batch suggestion is not implemented')

    @abstractmethod
    def _observe_impl(self, X: pd.DataFrame, y: np.ndarray) -> None: ...
