"""Native state integration: fresh PIDs, stochastic oracle, failed commit windows."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bo_benchmark.controller import Controller
from bo_benchmark.domain import OptimizerSpec, TaskSpec, ProtocolSpec, RunSpec
from bo_benchmark.mocks import MemoryStore, MockRuntime
from bo_benchmark.storage import FileStore
from bo_benchmark.subprocess_runtime import ToySubprocessRuntime
from bo_benchmark.transport import RemoteError, WorkerLaunch
from bo_benchmark.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'src/bo_benchmark/workers/toy_worker.py'


def spec():
    return RunSpec('native', OptimizerSpec('random', 'opt'),
                   TaskSpec('quadratic', 'task', {'noise_std': .2}), ProtocolSpec(13, 4), 27)


def launch(role, environment, spec):
    python = os.environ.get('BO_TEST_PYTHON37', sys.executable) if role == 'benchmark' else sys.executable
    return WorkerLaunch((python, '-u', str(WORKER), role))


class NativeSubprocessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name))
        self.workspace.prepare()

    def runtime(self, cls=ToySubprocessRuntime):
        return cls(launch, checkpoint_root=self.workspace.results / 'native')

    async def pause(self, store=None):
        stop = asyncio.Event()
        class PauseStore(FileStore):
            def save_checkpoint(inner, checkpoint):
                super().save_checkpoint(checkpoint)
                if checkpoint.restore_mode == 'native':
                    stop.set()
        store = store or PauseStore(self.workspace.results)
        runtime = self.runtime()
        result = await Controller(runtime, store, self.workspace, 1).run([spec()], stop)
        self.assertEqual(result['native'].status, 'paused')
        return store, runtime

    async def test_clean_stop_restores_both_worker_rng_states(self):
        store, initial = await self.pause()
        checkpoint = store.checkpoint('native')
        self.assertEqual(checkpoint.schema_version, 3)
        self.assertEqual(checkpoint.restore_mode, 'native')
        for descriptor in (checkpoint.optimizer_state, checkpoint.benchmark_state):
            self.assertEqual(descriptor['batch'], 1)
            self.assertTrue((self.workspace.results / 'native' / descriptor['path']).is_file())
        resumed_runtime = self.runtime()
        resumed = await Controller(resumed_runtime, FileStore(self.workspace.results), self.workspace, 1).run([spec()])
        direct_runtime = self.runtime()
        direct = await Controller(direct_runtime, MemoryStore(), self.workspace, 1).run([spec()])
        self.assertEqual(resumed['native'].status, 'completed')
        self.assertEqual(resumed['native'].batches, direct['native'].batches)
        self.assertEqual([len(b.values) for b in resumed['native'].batches], [4, 4, 4, 1])
        self.assertTrue(all(c.process.returncode is not None for rt in (initial, resumed_runtime, direct_runtime) for c in rt.channels))
        self.assertIsNone(FileStore(self.workspace.results).committed_checkpoint('native'))

    async def test_failed_second_worker_save_preserves_committed_and_new_history(self):
        store, initial = await self.pause()
        committed = store.committed_checkpoint('native')
        class BrokenRuntime(ToySubprocessRuntime):
            async def open_benchmark(inner, *args):
                client = await super().open_benchmark(*args)
                async def fail_save():
                    raise OSError('injected benchmark save failure')
                client.save_state = fail_save
                return client
        runtime = self.runtime(BrokenRuntime)
        outcome = (await Controller(runtime, store, self.workspace, 1).run([spec()]))['native']
        self.assertEqual(outcome.status, 'failed')
        self.assertEqual(outcome.evaluations, 8)
        self.assertEqual(store.committed_checkpoint('native'), committed)
        self.assertEqual(store.checkpoint('native').restore_mode, 'history_only')
        self.assertEqual(len(store.checkpoint('native').batches), 2)
        # No implicit rollback or replay of the tail on the next invocation.
        retry_runtime = self.runtime()
        retry = (await Controller(retry_runtime, store, self.workspace, 1).run([spec()]))['native']
        self.assertEqual(retry.status, 'failed')
        self.assertEqual(retry.evaluations, 8)
        self.assertEqual(retry_runtime.channels, [])
        self.assertEqual(store.committed_checkpoint('native'), committed)

    async def test_failed_host_commit_preserves_old_generation_and_tail(self):
        store, _ = await self.pause()
        old = store.committed_checkpoint('native')
        original = store._write
        def fail_native(record):
            if record['checkpoint'].restore_mode == 'native':
                raise OSError('injected manifest commit failure')
            return original(record)
        with patch.object(store, '_write', side_effect=fail_native):
            result = await Controller(self.runtime(), store, self.workspace, 1).run([spec()])
        self.assertEqual(result['native'].status, 'failed')
        fresh = FileStore(self.workspace.results)
        self.assertEqual(fresh.committed_checkpoint('native'), old)
        self.assertEqual(len(fresh.checkpoint('native').batches), 2)
        self.assertEqual(fresh.checkpoint('native').restore_mode, 'history_only')

    async def test_descriptor_corruption_and_path_escape_are_rejected_by_worker(self):
        store, _ = await self.pause()
        checkpoint = store.checkpoint('native')
        for mutation in ('checksum', 'outside', 'identity'):
            with self.subTest(mutation=mutation):
                descriptor = dict(checkpoint.optimizer_state)
                if mutation == 'checksum': descriptor['sha256'] = '0' * 64
                if mutation == 'outside': descriptor['path'] = '../state.pkl'
                if mutation == 'identity': descriptor['compatibility'] = {}
                runtime = self.runtime()
                benchmark = await runtime.open_benchmark(spec().task, Path('unused'), spec().seed)
                self.addAsyncCleanup(benchmark.close)
                task = await benchmark.describe()
                optimizer = await runtime.open_optimizer(spec().optimizer, Path('unused'), task, spec().seed)
                self.addAsyncCleanup(optimizer.close)
                with self.assertRaises(RemoteError):
                    await optimizer.load_state(descriptor)
                await optimizer.close()
                await benchmark.close()

    async def test_missing_native_file_fails_without_evaluating_new_batch(self):
        store, _ = await self.pause()
        checkpoint = store.checkpoint('native')
        (self.workspace.results / 'native' / checkpoint.benchmark_state['path']).unlink()
        result = await Controller(self.runtime(), store, self.workspace, 1).run([spec()])
        self.assertEqual(result['native'].status, 'failed')
        self.assertEqual(result['native'].evaluations, 4)
        self.assertEqual(store.checkpoint('native'), checkpoint)

    async def test_descriptor_batch_mismatch_cannot_publish_native_generation(self):
        store, _ = await self.pause()
        committed = store.committed_checkpoint('native')
        class WrongBatchRuntime(ToySubprocessRuntime):
            async def open_optimizer(inner, *args):
                client = await super().open_optimizer(*args)
                original = client.save_state
                async def save():
                    descriptor = await original()
                    descriptor['batch'] -= 1
                    return descriptor
                client.save_state = save
                return client
        result = await Controller(self.runtime(WrongBatchRuntime), store, self.workspace, 1).run([spec()])
        self.assertEqual(result['native'].status, 'failed')
        self.assertIn('batch mismatch', result['native'].error)
        self.assertEqual(store.committed_checkpoint('native'), committed)
        self.assertEqual(store.checkpoint('native').restore_mode, 'history_only')

    async def test_worker_checkpoint_files_are_immutable_generations(self):
        store, _ = await self.pause()
        descriptor = store.checkpoint('native').optimizer_state
        path = self.workspace.results / 'native' / descriptor['path']
        before = path.read_bytes()
        await Controller(self.runtime(), store, self.workspace, 1).run([spec()])
        self.assertEqual(path.read_bytes(), before)
        self.assertGreater(len(list((self.workspace.results / 'native' / 'optimizer').glob('*.json'))), 1)

    def test_fresh_controller_process_resume_and_completed_skip(self):
        def run(workspace, *extra):
            args = [sys.executable, str(ROOT / 'tests/fixtures/persistent_subprocess_campaign.py'),
                    '--workspace', str(workspace), '--runs', '2', '--workers', '1',
                    '--budget', '13', '--q', '4', '--noise-std', '.2', *extra]
            if os.environ.get('BO_TEST_PYTHON37'):
                args.extend(['--benchmark-python', os.environ['BO_TEST_PYTHON37']])
            completed = subprocess.run(args, capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            return json.loads((workspace / 'results/subprocess_summary.json').read_text())
        paused = run(self.workspace.root, '--stop-after-batches', '1')
        self.assertEqual(paused['runs']['toy_0']['status'], 'paused')
        self.assertEqual(paused['runs']['toy_1']['status'], 'pending')
        self.workspace.stop_file.unlink()
        resumed = run(self.workspace.root)
        self.assertNotEqual(paused['controller_pid'], resumed['controller_pid'])
        self.assertEqual(resumed['new_batches'], 7)
        self.assertTrue({w['pid'] for w in paused['workers']}.isdisjoint(w['pid'] for w in resumed['workers']))
        direct = run(self.workspace.root / 'direct')
        self.assertEqual(resumed['runs'], direct['runs'])
        skipped = run(self.workspace.root)
        self.assertEqual(skipped['new_batches'], 0)
        self.assertEqual(skipped['workers'], [])
        self.assertEqual(skipped['runs'], direct['runs'])

    async def test_odd_batch_restores_cached_gaussian_sample(self):
        run = replace(spec(), protocol=ProtocolSpec(10, 3))
        stop = asyncio.Event()
        class PauseStore(FileStore):
            def save_checkpoint(inner, checkpoint):
                super().save_checkpoint(checkpoint)
                if checkpoint.restore_mode == 'native': stop.set()
        store = PauseStore(self.workspace.results)
        paused = await Controller(self.runtime(), store, self.workspace, 1).run([run], stop)
        self.assertEqual(paused['native'].evaluations, 3)
        resumed = await Controller(self.runtime(), FileStore(self.workspace.results), self.workspace, 1).run([run])
        direct = await Controller(self.runtime(), MemoryStore(), self.workspace, 1).run([run])
        self.assertEqual(resumed['native'].batches, direct['native'].batches)

    async def test_failed_observe_preserves_prior_native_checkpoint_and_raw_batch(self):
        store, _ = await self.pause()
        committed = store.committed_checkpoint('native')
        class BrokenObserveRuntime(ToySubprocessRuntime):
            async def open_optimizer(inner, *args):
                client = await super().open_optimizer(*args)
                async def fail(X, y):
                    raise RuntimeError('injected observe failure')
                client.observe = fail
                return client
        result = await Controller(self.runtime(BrokenObserveRuntime), store, self.workspace, 1).run([spec()])
        self.assertEqual(result['native'].evaluations, 8)
        self.assertEqual(result['native'].status, 'failed')
        self.assertEqual(store.committed_checkpoint('native'), committed)
        self.assertEqual(len(store.checkpoint('native').batches), 2)
        self.assertEqual(store.checkpoint('native').restore_mode, 'history_only')
