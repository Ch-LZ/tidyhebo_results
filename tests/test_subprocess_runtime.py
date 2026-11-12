import asyncio
import ast
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bo_benchmark.controller import Controller
from bo_benchmark.domain import (OptimizerSpec, TaskSpec, ProtocolSpec, RunSpec,
                                 RealParameter, SearchSpace, TaskDescription)
from bo_benchmark.mocks import MemoryStore, MockRuntime
from bo_benchmark.subprocess_runtime import ToySubprocessRuntime
from bo_benchmark.transport import (JsonLinesChannel, WorkerLaunch, StartupError,
                                    TransportError, RemoteError)
from bo_benchmark.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
PEER = ROOT / 'tests/fixtures/protocol_worker.py'
WORKER = ROOT / 'src/bo_benchmark/workers/toy_worker.py'


def peer(mode='echo', record=None, **kwargs):
    return WorkerLaunch((sys.executable, '-u', str(PEER), mode,
                         *([str(record)] if record else [])), close_timeout=.15, **kwargs)


def launch(role, environment, spec):
    return WorkerLaunch((sys.executable, '-u', str(WORKER), role),
                        env={'TOY_ENV': role}, close_timeout=.2)


def spec(name='a', **options):
    return RunSpec(name, OptimizerSpec('random', 'opt'),
                   TaskSpec('quadratic', 'task', options), ProtocolSpec(9, 4), 42)


class SubprocessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    async def channel(self, config):
        channel = await JsonLinesChannel.open(config)
        self.addAsyncCleanup(channel.close)
        return channel

    async def test_unicode_positional_payload_pid_reuse_and_env_isolation(self):
        before = os.environ.get('TOY_ENV')
        a = await self.channel(peer(env={'TOY_ENV': 'optimizer'}))
        b = await self.channel(peer(env={'TOY_ENV': 'benchmark'}))
        payload = {'columns': ['цвет', 'имя'], 'data': [['синий', 'a\nb'], ['白', '']],
                   'array': {'shape': [2, 1], 'data': [[1.5], [-2.]]}}
        replies = await asyncio.gather(*(a.request('echo', payload) for _ in range(4)))
        self.assertEqual({r['pid'] for r in replies}, {a.pid})
        self.assertTrue(all(r['payload'] == payload for r in replies))
        self.assertEqual(replies[0]['env'], 'optimizer')
        self.assertEqual((await b.request('echo'))['env'], 'benchmark')
        self.assertEqual(os.environ.get('TOY_ENV'), before)
        self.assertEqual(len({os.getpid(), a.pid, b.pid}), 3)

    async def test_startup_eof_retries_but_structured_error_does_not(self):
        record = self.root / 'attempts'
        channel = await self.channel(peer('first_eof', record, startup_attempts=3))
        self.assertEqual(record.read_text().splitlines(), ['start', 'start'])
        await channel.request('echo')
        record.unlink()
        with self.assertRaises(RemoteError):
            await JsonLinesChannel.open(peer('startup_error', record, startup_attempts=3))
        self.assertEqual(record.read_text().splitlines(), ['start'])

    async def test_startup_timeout_and_missing_executable(self):
        with self.assertRaises(StartupError):
            await JsonLinesChannel.open(peer('startup_hang', startup_timeout=.1))
        with self.assertRaises(StartupError):
            await JsonLinesChannel.open(WorkerLaunch(('/missing/python',)))

    async def test_lost_mutating_response_is_never_replayed(self):
        record = self.root / 'commands'
        channel = await self.channel(peer('lost_response', record, startup_attempts=3))
        with self.assertRaisesRegex(TransportError, 'outcome unknown'):
            await channel.request('observe', {'some': 'state'})
        with self.assertRaises(TransportError):
            await channel.request('observe')
        self.assertEqual(record.read_text().splitlines(), ['start', 'observe'])

    async def test_protocol_and_remote_errors_poison_session(self):
        for mode in ('wrong_id', 'broken_json', 'nonfinite', 'remote_error'):
            with self.subTest(mode=mode):
                channel = await self.channel(peer(mode))
                with self.assertRaises(TransportError):
                    await channel.request('observe')
                with self.assertRaises(TransportError):
                    await channel.request('echo')
                await channel.close()
                self.assertIsNotNone(channel.process.returncode)

    async def test_command_wait_is_async_and_has_no_startup_deadline(self):
        channel = await self.channel(peer('delay', startup_timeout=.15))
        task = asyncio.create_task(channel.request('echo'))
        await asyncio.sleep(.04)
        self.assertFalse(task.done())
        reply = await task
        self.assertEqual(reply['pid'], channel.pid)

    async def test_cancel_inflight_prevents_reuse_and_reaps_process(self):
        channel = await self.channel(peer('delay'))
        task = asyncio.create_task(channel.request('observe'))
        await asyncio.sleep(.04)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(TransportError):
            await channel.request('echo')
        await channel.close()
        await channel.close()
        self.assertIsNotNone(channel.process.returncode)

    async def test_stderr_flood_and_hanging_close_do_not_deadlock(self):
        log = self.root / 'worker.log'
        channel = await self.channel(peer('stderr_flood', stderr_path=log))
        await asyncio.wait_for(channel.request('echo'), 3)
        await channel.close()
        self.assertTrue(channel.stderr_tail)
        self.assertGreaterEqual(log.stat().st_size, 900000)
        hanging = await self.channel(peer('hang_close'))
        await asyncio.wait_for(hanging.close(), 3)
        self.assertIsNotNone(hanging.process.returncode)

    async def test_local_invalid_json_does_not_send_or_poison(self):
        channel = await self.channel(peer())
        with self.assertRaises(ValueError):
            await channel.request('echo', float('nan'))
        self.assertEqual((await channel.request('echo', 'ok'))['payload'], 'ok')

    async def test_real_controller_toy_matches_mock_and_closes_all_workers(self):
        runtime, store = ToySubprocessRuntime(launch), MemoryStore()
        runs = [spec('a'), replace(spec('b'), seed=17)]
        outcomes = await Controller(runtime, store, Workspace(self.root / 'sub'), 2).run(runs)
        expected = await Controller(MockRuntime(), MemoryStore(), Workspace(self.root / 'mock'), 2).run(runs)
        self.assertEqual({o.status for o in outcomes.values()}, {'completed'})
        for name in outcomes:
            self.assertEqual(outcomes[name].batches, expected[name].batches)
            self.assertEqual([len(b.values) for b in outcomes[name].batches], [4, 4, 1])
        self.assertEqual(len(runtime.channels), 4)
        self.assertTrue(all(c.pid != os.getpid() and c.process.returncode is not None for c in runtime.channels))
        self.assertEqual(set(store.writer_pids), {os.getpid()})

    async def test_failed_initialization_does_not_stop_other_run(self):
        runtime = ToySubprocessRuntime(launch)
        outcomes = await Controller(runtime, MemoryStore(), Workspace(self.root), 2).run(
            [spec('bad', fail_initialize=True), spec('good')])
        self.assertEqual(outcomes['bad'].status, 'failed')
        self.assertEqual(outcomes['good'].status, 'completed')
        self.assertTrue(all(c.process.returncode is not None for c in runtime.channels))

    async def test_worker_is_python37_syntax_and_default_uses_environment_python(self):
        ast.parse(WORKER.read_text(), feature_version=(3, 7))
        ast.parse(WORKER.with_name("native_state.py").read_text(), feature_version=(3, 7))
        environment = self.root / 'venv'
        import venv
        await asyncio.to_thread(venv.EnvBuilder(with_pip=False).create, environment)
        runtime = ToySubprocessRuntime()
        client = await runtime.open_benchmark(spec().task, environment, 0)
        self.addAsyncCleanup(client.close)
        self.assertEqual((await client.describe()).direction, 'minimize')
        selected = Path(runtime.channels[0].launch.argv[0])
        self.assertEqual(selected.parent.parent, environment)
        channel = await self.channel(WorkerLaunch(
            (sys.executable, '-u', str(WORKER), 'benchmark'),
            env={'PYTHONIOENCODING': 'ascii'}))
        await channel.request('initialize', {'options': {}, 'seed': 0})
        with self.assertRaisesRegex(RemoteError, 'команда'):
            await channel.request('команда')

    async def test_stop_preserves_history_and_closes_dedicated_workers(self):
        stop = asyncio.Event()
        class StopRuntime(ToySubprocessRuntime):
            async def open_optimizer(self, *args):
                client = await super().open_optimizer(*args)
                original = client.observe
                async def observe(X, y):
                    await original(X, y)
                    stop.set()
                client.observe = observe
                return client
        runtime, store = StopRuntime(launch), MemoryStore()
        outcomes = await Controller(runtime, store, Workspace(self.root), 1).run(
            [spec('active'), spec('pending')], stop)
        self.assertEqual(outcomes['active'].status, 'paused')
        self.assertEqual(outcomes['active'].evaluations, 4)
        self.assertEqual(outcomes['pending'].status, 'pending')
        self.assertEqual(len(runtime.channels), 2)
        self.assertTrue(all(c.process.returncode is not None for c in runtime.channels))
        self.assertEqual(store.checkpoint('active').restore_mode, 'history_only')
        resumed = await Controller(ToySubprocessRuntime(launch), store, Workspace(self.root), 1).run([spec('active')])
        self.assertEqual(resumed['active'].status, 'failed')
        self.assertIn('history', resumed['active'].error.lower())

    async def test_python37_benchmark_and_current_python_optimizer(self):
        python37 = os.environ.get('BO_TEST_PYTHON37')
        if not python37:
            self.skipTest('Set BO_TEST_PYTHON37 to a Python 3.7 executable')
        def mixed_launch(role, environment, spec):
            python = python37 if role == 'benchmark' else sys.executable
            return WorkerLaunch((python, '-u', str(WORKER), role))
        runtime = ToySubprocessRuntime(mixed_launch)
        result = await Controller(runtime, MemoryStore(), Workspace(self.root), 1).run([spec()])
        self.assertEqual(result['a'].status, 'completed')
        self.assertEqual(result['a'].evaluations, 9)
        self.assertTrue(runtime.channels[0].ready['python'].startswith('3.7.'))
        self.assertEqual(runtime.channels[1].ready['python'], sys.version.split()[0])

    async def test_oversized_response_is_rejected_and_cleanup_drains_pipe(self):
        channel = await self.channel(peer('giant_response', max_line_bytes=1024))
        with self.assertRaisesRegex(TransportError, 'line limit'):
            await channel.request('echo')
        await asyncio.wait_for(channel.close(), 3)
        self.assertIsNotNone(channel.process.returncode)

    @unittest.skipUnless(os.name=='posix','POSIX process-group assertion; Windows requires native validation')
    async def test_close_reaps_wrapper_descendant_that_keeps_pipe_open(self):
        record=self.root/'tree'
        channel=await self.channel(peer('spawn_child',record))
        await channel.request('echo')
        child=int(next(line.split('=',1)[1] for line in record.read_text().splitlines() if line.startswith('child=')))
        await asyncio.wait_for(channel.close(),3)
        status=Path(f'/proc/{child}/status')
        if status.exists():
            self.assertIn('Z',next(line for line in status.read_text().splitlines() if line.startswith('State:')))
