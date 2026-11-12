"""Live scheduling and persistent resume across real controller invocations."""
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
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bo_benchmark.control import ControlFile
from bo_benchmark.controller import Controller
from bo_benchmark.domain import OptimizerSpec, ProtocolSpec, RunSpec, TaskSpec
from bo_benchmark.mocks import MemoryStore, MockRuntime
from bo_benchmark.storage import FileStore, atomic_write
from bo_benchmark.workspace import Workspace


def spec(name='a',q=2,budget=5,seed=42):
    return RunSpec(name,OptimizerSpec('random','modern'),TaskSpec(name,'legacy'),
                   ProtocolSpec(budget,q),seed)


async def until(predicate):
    async def wait():
        while not predicate(): await asyncio.sleep(.005)
    await asyncio.wait_for(wait(),5)


class Gates(MockRuntime):
    def __init__(self,n):
        super().__init__()
        self.entered={str(i):asyncio.Event() for i in range(n)}
        self.release={str(i):asyncio.Event() for i in range(n)}

    async def open_benchmark(self,spec,environment,seed):
        client=await super().open_benchmark(spec,environment,seed)
        original=client.evaluate
        async def evaluate(batch):
            self.entered[spec.name].set()
            await self.release[spec.name].wait()
            return await original(batch)
        client.evaluate=evaluate
        return client

    def release_all(self):
        for event in self.release.values():event.set()


class ControlStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace=Workspace(Path(self.tmp.name))
        self.workspace.prepare()
        self.control=self.workspace.results/'control.json'

    def limit(self,n):
        atomic_write(self.control,json.dumps({'workers':n}).encode())

    async def test_six_to_two_drains_runs_without_interrupting(self):
        self.limit(6);runtime=Gates(8)
        controller=Controller(runtime,MemoryStore(),self.workspace,1,
                              control_file=self.control,poll_interval=.01)
        task=asyncio.create_task(controller.run([spec(str(i),q=1,budget=3) for i in range(8)]))
        try:
            await until(lambda:all(runtime.entered[str(i)].is_set() for i in range(6)))
            self.limit(2)
            await until(lambda:controller.max_parallel==2)
            self.assertEqual(sum(not c.closed for c in runtime.optimizers),6)
            for i in range(4):runtime.release[str(i)].set()
            await until(lambda:sum(c.closed for c in runtime.benchmarks)==4)
            self.assertFalse(runtime.entered['6'].is_set())
            self.assertFalse(runtime.entered['7'].is_set())
            runtime.release['4'].set()
            await until(lambda:runtime.entered['6'].is_set())
            self.assertFalse(next(c for c in runtime.benchmarks if c.name=='5').closed)
        finally:
            runtime.release_all()
            result=await asyncio.wait_for(task,5)
        self.assertTrue(all(r.status=='completed' and r.evaluations==3 for r in result.values()))
        self.assertEqual(runtime.peak_open_optimizers,6)

    async def test_increase_during_inflight_batch_and_invalid_edit(self):
        self.limit(1);runtime=Gates(3)
        controller=Controller(runtime,MemoryStore(),self.workspace,1,
                              control_file=self.control,poll_interval=.01)
        task=asyncio.create_task(controller.run([spec(str(i)) for i in range(3)]))
        try:
            await until(lambda:runtime.entered['0'].is_set())
            self.assertFalse(runtime.entered['1'].is_set())
            self.limit(3)
            await until(lambda:all(e.is_set() for e in runtime.entered.values()))
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                self.control.write_text('{unfinished',encoding='utf-8')
                await until(lambda:bool(observed))
                self.assertEqual(controller.max_parallel,3)
                self.assertFalse(task.done())
                self.assertTrue(all(not c.closed for c in runtime.optimizers))
                self.limit(2)
                await until(lambda:controller.max_parallel==2)
        finally:
            runtime.release_all()
            result=await asyncio.wait_for(task,5)
        self.assertTrue(all(r.status=='completed' for r in result.values()))

    def test_control_file_rejects_invalid_limits_and_accepts_repaired(self):
        control=ControlFile(self.control)
        self.assertEqual(control.read_limit(2),2)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            for value in [0,-1,True,1.5,'2']:
                self.control.write_text(json.dumps({'workers':value}))
                self.assertEqual(control.read_limit(2),2)
                self.assertEqual(control.read_limit(2),2)
            self.assertEqual(len(observed),1)  # The repeated validation error is not spammed.
            self.limit(4)
            self.assertEqual(control.read_limit(2),4)
            self.limit(0)
            self.assertEqual(control.read_limit(4),4)
            self.assertEqual(len(observed),2)  # A repaired file ends the previous error episode.

    def test_property_limit_validation(self):
        c=Controller(MockRuntime(),MemoryStore(),self.workspace)
        for value in [0,True,-1,1.5]:
            with self.assertRaises(ValueError):c.max_parallel=value
        c.max_parallel=3
        self.assertEqual(c.max_parallel,3)

    async def test_stop_file_finishes_batch_closes_and_resume_matches_direct(self):
        runtime=Gates(2);store=FileStore(self.workspace.results)
        runs=[spec('0',q=4,budget=9),spec('1',q=4,budget=9)]
        task=asyncio.create_task(Controller(runtime,store,self.workspace,1,poll_interval=.01).run(runs))
        await until(lambda:runtime.entered['0'].is_set())
        self.workspace.stop_file.touch()
        self.assertFalse(task.done())
        runtime.release_all()
        paused=await asyncio.wait_for(task,5)
        self.assertEqual(paused['0'].status,'paused')
        self.assertEqual(paused['0'].evaluations,4)
        self.assertEqual(paused['1'].status,'pending')
        self.assertTrue(all(c.closed for c in [*runtime.optimizers,*runtime.benchmarks]))
        cp=FileStore(self.workspace.results).checkpoint('0')
        self.assertEqual(cp.restore_mode,'native')
        self.assertEqual(len(cp.batches),1)
        self.workspace.stop_file.unlink()
        resumed=await Controller(MockRuntime(),FileStore(self.workspace.results),self.workspace,2).run(runs)
        direct=await Controller(MockRuntime(),MemoryStore(),self.workspace,2).run(runs)
        for run in runs:
            self.assertEqual(resumed[run.run_id].batches,direct[run.run_id].batches)
            self.assertEqual(resumed[run.run_id].status,'completed')
            self.assertIsNone(FileStore(self.workspace.results).checkpoint(run.run_id))

    async def test_stop_before_resume_preserves_history_without_opening_clients(self):
        store=FileStore(self.workspace.results);stop=asyncio.Event()
        await Controller(MockRuntime(on_observe=lambda X,y:stop.set()),store,self.workspace,1).run([spec() ],stop)
        self.workspace.stop_file.touch()
        runtime=MockRuntime()
        r=(await Controller(runtime,FileStore(self.workspace.results),self.workspace).run([spec()]))['a']
        self.assertEqual((r.status,r.evaluations,r.direction),('paused',2,'minimize'))
        self.assertEqual(runtime.events,[])
        self.assertEqual(len(store.checkpoint('a').batches),1)

    async def test_file_history_without_native_state_survives_and_is_not_restarted(self):
        import pandas as pd
        class Plain:
            def suggest(self,n=1):return pd.DataFrame({'x':[.25]*n,'y':[.5]*n})
            def observe(self,X,y):pass
        stop=asyncio.Event();store=FileStore(self.workspace.results)
        rt=MockRuntime(on_observe=lambda X,y:stop.set(),optimizer_factory=lambda s,t,seed:Plain())
        await Controller(rt,store,self.workspace,1).run([spec()],stop)
        cp=FileStore(self.workspace.results).checkpoint('a')
        self.assertEqual((len(cp.batches),cp.restore_mode),(1,'history_only'))
        rt2=MockRuntime(optimizer_factory=lambda s,t,seed:Plain())
        r=(await Controller(rt2,FileStore(self.workspace.results),self.workspace).run([spec()]))['a']
        self.assertEqual((r.status,r.evaluations),('failed',2))
        self.assertIn('history-only resume policy',r.error)
        self.assertFalse(any(e.action=='evaluate_started' for e in rt2.events))

    async def test_failed_atomic_replace_retains_previous_native_record(self):
        store=FileStore(self.workspace.results);stop=asyncio.Event()
        await Controller(MockRuntime(on_observe=lambda X,y:stop.set()),store,self.workspace,1).run([spec()],stop)
        path=self.workspace.results/'runs/a/state.pkl'
        previous=path.read_bytes();cp=store.checkpoint('a')
        with patch('bo_benchmark.storage.os.replace',side_effect=OSError('injected write failure')):
            with self.assertRaisesRegex(OSError,'injected'):
                store.save_checkpoint(replace(cp,restore_mode='history_only',optimizer_state=None))
        self.assertEqual(path.read_bytes(),previous)
        self.assertEqual(store.checkpoint('a').restore_mode,'native')
        self.assertEqual(list(path.parent.glob('*.tmp')),[])

    async def test_completed_skipped_spec_guard_and_corruption_not_missing(self):
        store=FileStore(self.workspace.results)
        await Controller(MockRuntime(),store,self.workspace).run([spec()])
        path=self.workspace.results/'runs/a/state.pkl';original=path.read_bytes()
        runtime=MockRuntime()
        await Controller(runtime,FileStore(self.workspace.results),self.workspace).run([spec()])
        self.assertEqual(runtime.events,[])
        self.assertEqual(path.read_bytes(),original)
        with self.assertRaisesRegex(ValueError,'another specification'):
            await Controller(MockRuntime(),store,self.workspace).run([spec(seed=43)])
        self.assertEqual(path.read_bytes(),original)
        record=pickle.loads(original);record['compatibility']['python']='different'
        path.write_bytes(pickle.dumps(record,protocol=4))
        with self.assertRaisesRegex(ValueError,'dependencies changed'):store.outcome('a')
        path.write_bytes(b'not a valid pickle')
        with self.assertRaises(pickle.UnpicklingError):store.outcome('a')

    async def test_cancellation_keeps_committed_state_and_blocks_unknown_batch_replay(self):
        second=asyncio.Event();never=asyncio.Event()
        class Runtime(MockRuntime):
            async def open_benchmark(self,s,environment,seed):
                client=await super().open_benchmark(s,environment,seed)
                original=client.evaluate
                async def evaluate(batch):
                    if client.calls==1:
                        second.set();await never.wait()
                    return await original(batch)
                client.evaluate=evaluate
                return client
        rt=Runtime();store=FileStore(self.workspace.results)
        task=asyncio.create_task(Controller(rt,store,self.workspace).run([spec()]))
        await asyncio.wait_for(second.wait(),5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(self.workspace.stop_file.exists())
        self.assertEqual(len(store.checkpoint('a').batches),1)
        self.assertTrue(store.checkpoint('a').in_flight)
        self.assertEqual(store.committed_checkpoint('a').restore_mode,'native')
        self.assertEqual(len(store.committed_checkpoint('a').batches),1)
        self.assertTrue(all(c.closed for c in [*rt.optimizers,*rt.benchmarks]))
        self.workspace.stop_file.unlink()
        retry_runtime=MockRuntime()
        resumed=(await Controller(retry_runtime,FileStore(self.workspace.results),self.workspace).run([spec()]))['a']
        self.assertEqual(resumed.status,'failed')
        self.assertIn('in-flight',resumed.error)
        self.assertEqual(retry_runtime.events,[])
        self.assertEqual(len(store.committed_checkpoint('a').batches),1)

    def test_native_resume_in_new_os_process_and_completed_skip(self):
        root=Path(__file__).resolve().parents[1]
        def launch(workspace,*extra):
            cmd=[sys.executable,str(root/'tests/fixtures/persistent_mock_campaign.py'),
                 '--workspace',str(workspace),'--runs','2','--workers','1',*extra]
            result=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
            return json.loads((workspace/'results/mock_summary.json').read_text())
        paused=launch(self.workspace.root,'--stop-after-batches','1')
        self.assertEqual(paused['runs']['mock_0']['status'],'paused')
        self.assertEqual(paused['runs']['mock_0']['evaluations'],4)
        self.workspace.stop_file.unlink()
        self.limit(2)
        resumed=launch(self.workspace.root)
        self.assertNotEqual(paused['pid'],resumed['pid'])
        self.assertEqual(resumed['native_restores'],1)
        self.assertEqual(resumed['new_batches'],5)
        direct=launch(self.workspace.root/'direct')
        self.assertEqual(resumed['runs'],direct['runs'])
        records={p:p.read_bytes() for p in (self.workspace.results/'runs').rglob('state.pkl')}
        again=launch(self.workspace.root)
        self.assertEqual(again['new_batches'],0)
        self.assertEqual(again['native_restores'],0)
        self.assertTrue(all(p.read_bytes()==data for p,data in records.items()))


if __name__=='__main__':unittest.main()
