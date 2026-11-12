"""Local single-writer checkpoints for the current same-environment mock core.

Each run has ONE atomic record containing checkpoint and outcome. Native pickle
stays local; it is not the JSON transport format. Only load your own state files.
The controller's workspace lease must enclose all writes to a shared store.
"""
import hashlib
import os
from pathlib import Path
import pickle
import platform
import tempfile
import time
import numpy as np
import pandas as pd
from .domain import RunCheckpoint, RunOutcome, safe_name


def atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(.05*(attempt+1))
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compatibility():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                code=digest.hexdigest())


class FileStore:
    """RunStore backed by results/runs/<run_id>/state.pkl; no database or cache."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.compatibility = compatibility()

    def _path(self, run_id):
        safe_name(run_id)
        return self.root/'runs'/run_id/'state.pkl'

    def _read(self, run_id):
        for attempt in range(8):
            try:
                with self._path(run_id).open('rb') as stream:
                    record = pickle.load(stream)
                break
            except FileNotFoundError:
                return None
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.02*(attempt+1))
        if not isinstance(record, dict) or record.get('schema_version') != 1:
            raise ValueError('Unsupported or corrupt run record')
        if record.get('compatibility') != self.compatibility:
            raise ValueError('Run record code/Python/dependencies changed')
        if record.get('spec') is None or record['spec'].run_id != run_id:
            raise ValueError('Run record identity mismatch')
        for key, kind in [('checkpoint', RunCheckpoint), ('committed', RunCheckpoint), ('outcome', RunOutcome)]:
            value = record.get(key)
            if value is not None and (not isinstance(value, kind) or value.spec != record['spec']):
                raise ValueError('Corrupt run record '+key)
        return record

    def _record(self, spec):
        record = self._read(spec.run_id)
        if record is not None:
            if record['spec'] != spec:
                raise ValueError('Run ID already belongs to another specification')
            return record
        return dict(schema_version=1, compatibility=self.compatibility, spec=spec,
                    checkpoint=None, committed=None, outcome=None)

    def checkpoint(self, run_id):
        record = self._read(run_id)
        return record['checkpoint'] if record else None

    def committed_checkpoint(self, run_id):
        record = self._read(run_id)
        return record.get('committed') if record else None

    def outcome(self, run_id):
        record = self._read(run_id)
        return record['outcome'] if record else None

    def save_checkpoint(self, checkpoint):
        record = self._record(checkpoint.spec)
        if record['outcome'] is not None and record['outcome'].status == 'completed':
            raise ValueError('Cannot checkpoint a completed run')
        record['checkpoint'] = checkpoint
        if checkpoint.restore_mode == 'native':
            # Publish the whole consistent generation in this single atomic record.
            # A later history-only tail must not erase it.
            record['committed'] = checkpoint
        self._write(record)

    def save_outcome(self, outcome):
        record = self._record(outcome.spec)
        if record['outcome'] is not None and record['outcome'].status == 'completed':
            if outcome != record['outcome']:
                raise ValueError('Cannot overwrite a completed run')
            return
        record['outcome'] = outcome
        if outcome.status == 'completed':
            record['checkpoint'] = None
            record['committed'] = None
        self._write(record)

    def _write(self, record):
        atomic_write(self._path(record['spec'].run_id), pickle.dumps(record, protocol=4))
