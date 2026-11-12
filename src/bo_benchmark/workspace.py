"""Executor layout and an OS lease for the single local controller."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
from .domain import safe_name


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def results(self) -> Path:
        return self.root/'results'

    @property
    def stop_file(self) -> Path:
        return self.results/'STOP'

    def environment(self, role: Literal['optimizers', 'benchmarks'], name: str) -> Path:
        if role not in ('optimizers', 'benchmarks'):
            raise ValueError('Unknown environment role')
        safe_name(name)
        return self.root/'environments'/role/name

    def prepare(self) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        for role in ('optimizers', 'benchmarks'):
            (self.root/'environments'/role).mkdir(parents=True, exist_ok=True)

    def controller_lease(self):
        return ControllerLease(self.results/'.controller.lock')


class ControllerLease:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError('A controller already owns this executor workspace') from error
        self.handle = handle
        return self

    def __exit__(self, *exc):
        self.handle.close()
        self.handle = None
