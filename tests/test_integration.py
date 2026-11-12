"""Contract integration using real scheduling and fake computation adapters."""
import asyncio
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bo_benchmark.controller import Controller
from bo_benchmark.domain import OptimizerSpec, ProtocolSpec, RunSpec, TaskSpec
from bo_benchmark.mocks import MemoryStore, MockRuntime
from bo_benchmark.workspace import Workspace


def spec(name='a', q=2, budget=5, seed=42, **task_options):
    return RunSpec(name, OptimizerSpec('random', 'modern'),
                   TaskSpec('quadratic', 'legacy', task_options), ProtocolSpec(budget, q), seed)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name))

    async def test_parallel_runs_have_one_controller_and_separate_environments(self):
        runtime, store = MockRuntime(), MemoryStore()
        runs = [spec(f'run{i}', seed=i) for i in range(5)]
        results = await Controller(runtime, store, self.workspace, 2).run(runs)
        self.assertEqual({r.status for r in results.values()}, {'completed'})
        self.assertEqual(runtime.peak_evaluations, 2)
        self.assertEqual(runtime.peak_open_optimizers, 2)
        self.assertEqual({e.pid for e in runtime.events}, {os.getpid()})
        self.assertEqual(set(store.writer_pids), {os.getpid()})
        self.assertTrue(all(x.closed for x in [*runtime.optimizers, *runtime.benchmarks]))
        for r in results.values():
            self.assertEqual([len(b.values) for b in r.batches], [2, 2, 1])
        self.assertEqual(runtime.optimizers[0].environment, self.workspace.root/'environments/optimizers/modern')
        self.assertEqual(runtime.benchmarks[0].environment, self.workspace.root/'environments/benchmarks/legacy')
        self.assertTrue(self.workspace.results.is_dir())

    async def test_q_modes_and_partial_final_batch(self):
        for q in [1, 2, 4, 8, 16]:
            with self.subTest(q=q):
                r = (await Controller(MockRuntime(), MemoryStore(), self.workspace).run([spec(q=q,budget=19)]))['a']
                self.assertEqual(r.evaluations, 19)
                self.assertEqual(r.status, 'completed')
                self.assertTrue(all(len(b.values)<=q for b in r.batches))

    async def test_native_mock_resume_matches_uninterrupted_run(self):
        stop, store = asyncio.Event(), MemoryStore()
        runtime = MockRuntime(on_observe=lambda X,y: stop.set())
        paused = await Controller(runtime, store, self.workspace, 1).run([spec()], stop)
        self.assertEqual(paused['a'].status, 'paused')
        self.assertEqual(paused['a'].evaluations, 2)
        checkpoint = store.checkpoint('a')
        resumed_runtime = MockRuntime()
        resumed = (await Controller(resumed_runtime, store, self.workspace).run([spec()]))['a']
        direct = (await Controller(MockRuntime(), MemoryStore(), self.workspace).run([spec()]))['a']
        self.assertEqual(resumed.batches, direct.batches)
        self.assertIn('restore', [e.action for e in resumed_runtime.events])
        self.assertEqual(len(checkpoint.batches), 1)
        self.assertIsNone(store.checkpoint('a'))

    async def test_completed_runs_are_skipped_and_changed_spec_rejected(self):
        store = MemoryStore()
        await Controller(MockRuntime(), store, self.workspace).run([spec()])
        runtime = MockRuntime()
        again = await Controller(runtime, store, self.workspace).run([spec()])
        self.assertEqual(again['a'].status, 'completed')
        self.assertEqual(runtime.events, [])
        with self.assertRaisesRegex(ValueError, 'another specification'):
            await Controller(runtime, store, self.workspace).run([spec(seed=43)])

    async def test_failed_benchmark_does_not_stop_other_runs(self):
        runtime, store = MockRuntime(), MemoryStore()
        r = await Controller(runtime, store, self.workspace).run(
            [spec('bad',fail_on_call=2),spec('good')])
        self.assertEqual(r['bad'].status, 'failed')
        self.assertEqual(r['bad'].evaluations, 2)
        self.assertEqual(r['good'].status, 'completed')
        self.assertIn('injected benchmark failure',r['bad'].error)
        self.assertIsNotNone(store.checkpoint('bad'))
        self.assertTrue(all(x.closed for x in [*runtime.optimizers,*runtime.benchmarks]))

    async def test_unsupported_many_rejected_before_any_evaluation(self):
        run = spec(q=4)
        run = replace(run,optimizer=replace(run.optimizer,options={'supports_many':False}))
        runtime = MockRuntime()
        r = await Controller(runtime, MemoryStore(), self.workspace).run([run])
        self.assertEqual(r['a'].status,'failed')
        self.assertIn('Batch suggestion is not implemented',r['a'].error)
        self.assertFalse(any(e.action=='evaluate_started' for e in runtime.events))

    async def test_seed_results_independent_of_interleaving(self):
        runs = [spec('a'),spec('b',seed=43)]
        serial = await Controller(MockRuntime(), MemoryStore(), self.workspace,1).run(runs)
        parallel = await Controller(MockRuntime(), MemoryStore(), self.workspace,2).run(runs)
        for run in runs:
            self.assertEqual(serial[run.run_id].batches,parallel[run.run_id].batches)

    async def test_stop_before_start_creates_no_clients(self):
        stop = asyncio.Event();stop.set()
        runtime=MockRuntime()
        r=await Controller(runtime,MemoryStore(),self.workspace).run([spec()],stop)
        self.assertEqual(r['a'].status,'pending')
        self.assertFalse(runtime.events)

    async def test_duplicate_ids_and_reentrant_controller_rejected(self):
        controller=Controller(MockRuntime(),MemoryStore(),self.workspace)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            await controller.run([spec(),spec()])
        task=asyncio.create_task(controller.run([spec(delay=.02)]))
        await asyncio.sleep(0)
        with self.assertRaisesRegex(RuntimeError,'already active'):
            await controller.run([spec('b')])
        await task

    async def test_another_process_cannot_own_same_executor(self):
        src=str(Path(__file__).resolve().parents[1]/'src')
        code=('from pathlib import Path\nfrom bo_benchmark.workspace import Workspace\n'
              'import sys\nwith Workspace(Path(sys.argv[1])).controller_lease(): pass\n')
        env=dict(os.environ,PYTHONPATH=src,PYTHONDONTWRITEBYTECODE='1')
        self.workspace.prepare()
        with self.workspace.controller_lease():
            result=subprocess.run([sys.executable,'-c',code,str(self.workspace.root)],
                                  env=env,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('controller already owns',result.stderr)
        result=subprocess.run([sys.executable,'-c',code,str(self.workspace.root)],
                              env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
