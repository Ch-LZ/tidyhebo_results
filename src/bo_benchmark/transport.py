"""Async JSON-lines subprocess transport. One in-flight command per worker.

No BO-command deadline and no command replay: cancellation or broken framing
poisons this session. Only pre-ready EOF/OS errors/timeouts may be retried.
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Mapping


class TransportError(RuntimeError):
    pass


class StartupError(TransportError):
    pass


class RemoteError(TransportError):
    pass


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def encode(message):
    return (json.dumps(message, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


@dataclass(frozen=True)
class WorkerLaunch:
    """Explicit executable/argv; env overlays a COPY of controller's environment."""
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    startup_timeout: float = 30.0
    startup_attempts: int = 1
    close_timeout: float = 2.0
    max_line_bytes: int = 8 * 1024 * 1024
    stderr_path: Path | None = None

    def __post_init__(self):
        if not self.argv or any(not isinstance(x, str) for x in self.argv):
            raise ValueError('Expected nonempty argv')
        if type(self.startup_attempts) is not int or self.startup_attempts < 1:
            raise ValueError('Expected positive startup_attempts')
        if self.startup_timeout <= 0 or self.close_timeout <= 0 or self.max_line_bytes < 256:
            raise ValueError('Expected positive lifecycle limits and line size >= 256')


async def _stop_tree(process, force=False):
    """Mamba wrappers and their Python children share the worker process group."""
    if os.name == 'posix':
        try: os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError: pass
    elif process.returncode is None:
        flags = ['/F'] if force else []
        try:
            killer = await asyncio.create_subprocess_exec('taskkill', '/PID', str(process.pid),
                '/T', *flags, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await killer.wait()
        except OSError:
            try: process.kill() if force else process.terminate()
            except ProcessLookupError: pass


class JsonLinesChannel:
    def __init__(self, process, launch):
        self.process, self.launch = process, launch
        self.ready = None
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._closed = False
        self._broken = False
        self._closing = None
        # Keep a bounded diagnostic tail; always drain stderr to avoid deadlock.
        self.stderr_tail = deque(maxlen=32)
        self._log = None
        if launch.stderr_path is not None:
            path = Path(launch.stderr_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = path.open('ab', buffering=0)
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    @property
    def pid(self):
        return self.process.pid

    async def _drain_stderr(self):
        while True:
            data = await self.process.stderr.read(4096)
            if not data:
                return
            self.stderr_tail.append(data.decode('utf-8', errors='replace'))
            if self._log is not None: self._log.write(data)

    @classmethod
    async def open(cls, launch):
        for attempt in range(launch.startup_attempts):
            channel = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *launch.argv, cwd=launch.cwd, env={**os.environ, **launch.env},
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, limit=launch.max_line_bytes,
                    start_new_session=os.name == 'posix',
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
                try:
                    channel = cls(process, launch)
                except BaseException:
                    await _stop_tree(process, force=True)
                    await process.wait()
                    raise
                channel.ready = await asyncio.wait_for(channel._read(), launch.startup_timeout)
                ready = channel.ready
                if ready.get('event') == 'error':
                    raise RemoteError(f'Worker initialization failed: {ready.get("error")}')
                if ready.get('event') != 'ready' or ready.get('protocol') != 1:
                    raise TransportError('Expected protocol=1 ready event')
                return channel
            except (OSError, EOFError, asyncio.TimeoutError) as exc:
                if channel is not None:
                    channel._broken = True
                    await channel.close()
                if attempt + 1 == launch.startup_attempts:
                    raise StartupError(f'Worker failed before ready: {exc}') from exc
            except BaseException:
                if channel is not None:
                    channel._broken = True
                    await channel.close()
                raise

    async def _read(self):
        try:
            line = await self.process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise TransportError('Worker response exceeds line limit') from exc
        if not line:
            raise EOFError(f'Worker {self.pid} closed stdout')
        if len(line) > self.launch.max_line_bytes or not line.endswith(b'\n'):
            raise TransportError('Oversized or incomplete JSON line')
        try:
            message = json.loads(line.decode('utf-8'), parse_constant=_reject_constant)
        except (ValueError, UnicodeError) as exc:
            raise TransportError('Invalid UTF-8 JSON response') from exc
        if not isinstance(message, dict):
            raise TransportError('Expected JSON object')
        return message

    async def request(self, command, payload=None):
        # Encode before acquiring/sending: invalid local data cannot desynchronize IPC.
        async with self._lock:
            if self._closed or self._broken:
                raise TransportError('Worker session is closed or broken')
            self._sequence += 1
            request_id = self._sequence
            frame = encode({'protocol': 1, 'request_id': request_id,
                            'command': command, 'payload': payload})
            if len(frame) > self.launch.max_line_bytes:
                raise ValueError('Request exceeds line limit')
            try:
                self.process.stdin.write(frame)
                await self.process.stdin.drain()
                reply = await self._read()
                if (reply.get('protocol') != 1 or type(reply.get('request_id')) is not int
                        or reply['request_id'] != request_id
                        or type(reply.get('ok')) is not bool):
                    raise TransportError('Mismatched or invalid response envelope')
                if not reply['ok']:
                    # A failed mutating command may have partially changed native state.
                    raise RemoteError(str(reply.get('error', 'Worker command failed')) +
                        (f' [stderr: {self.launch.stderr_path}]' if self.launch.stderr_path else ''))
                return reply.get('result')
            except BaseException as exc:
                self._broken = True
                if isinstance(exc, (EOFError, OSError)):
                    raise TransportError(f'Command {command!r} lost its response; outcome unknown') from exc
                raise

    async def close(self):
        # Share a shielded task: caller cancellation must not orphan a worker.
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.create_task(self._cleanup())
        await asyncio.shield(self._closing)

    async def _discard_stdout(self):
        async with self._lock:
            while await self.process.stdout.read(65536):
                pass

    async def _cleanup(self):
        process = self.process
        # Drain even unread responses: asyncio Process.wait can otherwise hang
        # after kill when a full stdout StreamReader has paused the pipe.
        discard = asyncio.create_task(self._discard_stdout())
        try:
            if process.returncode is None:
                if not self._broken and not self._lock.locked():
                    try:
                        process.stdin.write(encode({'protocol': 1, 'request_id': 0,
                                                    'command': 'close', 'payload': None}))
                        await asyncio.wait_for(process.stdin.drain(), self.launch.close_timeout)
                    except (OSError, asyncio.TimeoutError):
                        pass
                else:
                    try:
                        await _stop_tree(process)
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), self.launch.close_timeout)
                except asyncio.TimeoutError:
                    try:
                        await _stop_tree(process)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), self.launch.close_timeout)
                    except asyncio.TimeoutError:
                        try:
                            await _stop_tree(process, force=True)
                        except ProcessLookupError:
                            pass
                        await process.wait()
        finally:
            if os.name == 'posix': await _stop_tree(process, force=True)
            if process.stdin is not None:
                process.stdin.close()
            discard.cancel()
            self._stderr_task.cancel()
            await asyncio.gather(discard, self._stderr_task, return_exceptions=True)
            if self._log is not None: self._log.close()
