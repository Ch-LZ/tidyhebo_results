"""User-facing API and controller boundaries, using cheap independent doubles."""
import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bo_benchmark.clients import local_client
from bo_benchmark.controller import Controller
from bo_benchmark.domain import OptimizerSpec, ProtocolSpec, RunSpec, TaskSpec
from bo_benchmark.mocks import MemoryStore, MockRuntime
from bo_benchmark.optimizer import AbstractOptimizer
from bo_benchmark.validation import normalize_observations
from bo_benchmark.workspace import Workspace


class PlainOptimizer:
    # Deliberately no framework superclass, initialize, close, or checkpoint.
    def __init__(self):
        self.observed = []
        self.calls = 0

    def suggest(self, n=1):
        self.calls += 1
        # Column order differs from task order; index carries no identity.
        return pd.DataFrame({'y': [.5]*n, 'x': [.25]*n}, index=[91]*n)

    def observe(self, X, y):
        self.observed.append((X.copy(), y.copy()))


class OneOnly(AbstractOptimizer):
    def __init__(self):
        super().__init__(['x'])
        self.history = []

    def _suggest_one_impl(self):
        return pd.DataFrame({'x':[.5]})

    def _observe_impl(self, X, y):
        self.history.append((X,y))


def spec(name='a', direction='minimize', q=2):
    return RunSpec(name, OptimizerSpec('plain','modern'),
                   TaskSpec('quadratic','legacy',{'direction':direction}),ProtocolSpec(5,q),42)


class OptimizerContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Workspace(Path(self.tmp.name))

    def test_positional_pairing_and_column_order(self):
        X = pd.DataFrame({'y':[7,8], 'x':[1,2]}, index=[100,100])
        y = pd.Series([3.,4.],index=[9,-2])
        a,b = normalize_observations(X,y,['x','y'])
        self.assertEqual(a.columns.tolist(),['x','y'])
        self.assertEqual(a.index.tolist(),[0,1])
        np.testing.assert_array_equal(a.to_numpy(),[[1,7],[2,8]])
        np.testing.assert_array_equal(b,[[3],[4]])
        self.assertEqual(X.index.tolist(),[100,100])

    def test_y_containers_and_shapes(self):
        X = pd.DataFrame({'x':[0,1]})
        for y in [pd.Series([2.,3.]),np.array([2.,3.]),np.array([[2.],[3.]])]:
            with self.subTest(y=type(y)):
                a,b=normalize_observations(X,y)
                self.assertEqual(b.shape,(2,1))
                b[0]=99
                self.assertEqual(np.asarray(y).flat[0],2.)

    def test_observation_validation(self):
        X = pd.DataFrame({'x':[0,1]})
        cases=[(X,[1.,2.],TypeError), (X.to_numpy(),np.array([1,2]),TypeError),
               (X,np.ones((1,2)),ValueError),(X,np.array([np.nan,2]),ValueError),
               (X,np.array([np.inf,2]),ValueError),(X,np.array(['a','b']),ValueError),
               (X,np.ones((2,1,1)),ValueError),
               (pd.DataFrame([[1,2]],columns=['x','x']),np.ones(1),ValueError)]
        for a,b,error in cases:
            with self.subTest(a=type(a),b=np.shape(b)):
                with self.assertRaises(error): normalize_observations(a,b)

    def test_abc_single_and_explicit_many_failure(self):
        obj=OneOnly()
        self.assertEqual(obj.suggest().shape,(1,1))
        with self.assertRaises(NotImplementedError): obj.suggest(2)
        for n in [True,0,-1,1.5]:
            with self.subTest(n=n):
                with self.assertRaises(ValueError): obj.suggest(n)

    async def test_initial_history_before_first_suggest(self):
        obj=PlainOptimizer();client=local_client(obj,['x','y'])
        await client.observe(pd.DataFrame({'x':[0.,1.],'y':[.1,.2]},index=[8,7]),
                             pd.Series([10.,20.],index=[7,8]))
        self.assertEqual(obj.calls,0)
        np.testing.assert_array_equal(obj.observed[0][1],[[10],[20]])
        self.assertEqual((await client.suggest()).shape,(1,2))
        await client.close()
        # Client cleanup never requires a close method on the mathematical object.
        self.assertFalse(hasattr(obj,'close'))

    async def test_ready_injected_object_and_raw_direction(self):
        for direction in ['minimize','maximize']:
            obj=PlainOptimizer();store=MemoryStore()
            rt=MockRuntime(optimizer_objects={'plain':obj})
            result=(await Controller(rt,store,self.workspace).run([spec(direction=direction)]))['a']
            self.assertEqual(result.status,'completed')
            self.assertEqual(result.direction,direction)
            self.assertEqual([len(b.values) for b in result.batches],[2,2,1])
            raw=np.concatenate([b.values for b in result.batches])
            received=np.concatenate([y.ravel() for X,y in obj.observed])
            np.testing.assert_array_equal(received,-raw if direction=='minimize' else raw)
            self.assertTrue((raw>=0).all())
            self.assertEqual(obj.observed[0][0].columns.tolist(),['x','y'])
            self.assertFalse(hasattr(obj,'initialize'))
            self.assertFalse(hasattr(obj,'capabilities'))

    async def test_history_without_native_state_is_preserved_and_resume_explicit(self):
        stop=asyncio.Event();store=MemoryStore();obj=PlainOptimizer()
        rt=MockRuntime(on_observe=lambda X,y:stop.set(),optimizer_objects={'plain':obj})
        paused=(await Controller(rt,store,self.workspace).run([spec()],stop))['a']
        self.assertEqual(paused.evaluations,2)
        cp=store.checkpoint('a')
        self.assertEqual(cp.restore_mode,'history_only')
        self.assertEqual(len(cp.batches),1)
        self.assertEqual(cp.direction,'minimize')
        resumed_rt=MockRuntime(optimizer_objects={'plain':PlainOptimizer()})
        result=(await Controller(resumed_rt,store,self.workspace).run([spec()]))['a']
        self.assertEqual(result.status,'failed')
        self.assertEqual(result.evaluations,2)
        self.assertIn('history-only resume policy',result.error)
        self.assertFalse(any(e.action=='evaluate_started' for e in resumed_rt.events))
        self.assertEqual(store.checkpoint('a'),cp)

    async def test_observe_failure_keeps_evaluations_and_other_run_completes(self):
        class Broken(PlainOptimizer):
            def observe(self,X,y): raise RuntimeError('observe failed')
        store=MemoryStore()
        def factory(s,t,seed): return Broken() if s.name=='bad' else PlainOptimizer()
        bad=replace(spec('bad'),optimizer=OptimizerSpec('bad','modern'))
        r=await Controller(MockRuntime(optimizer_factory=factory),store,self.workspace).run([bad,spec('good')])
        self.assertEqual(r['bad'].status,'failed')
        self.assertEqual(r['bad'].evaluations,2)
        self.assertEqual(store.checkpoint('bad').restore_mode,'history_only')
        self.assertEqual(r['good'].status,'completed')

    async def test_snapshot_failure_keeps_result_as_history_only(self):
        class BrokenState(PlainOptimizer):
            def save_state(self): raise RuntimeError('snapshot failed')
            def load_state(self,state): pass
        store=MemoryStore();rt=MockRuntime(optimizer_objects={'plain':BrokenState()})
        r=(await Controller(rt,store,self.workspace).run([spec()]))['a']
        self.assertEqual(r.status,'failed')
        self.assertEqual(r.evaluations,2)
        self.assertEqual(store.checkpoint('a').restore_mode,'history_only')

    async def test_bad_suggest_is_rejected_before_evaluation(self):
        for output in [np.ones((2,2)),pd.DataFrame({'x':[0],'y':[0]}),
                       pd.DataFrame({'wrong':[0,0],'y':[0,0]}),
                       pd.DataFrame({'x':[np.nan,0],'y':[0,0]}),
                       pd.DataFrame({'x':[5,0],'y':[0,0]})]:
            with self.subTest(output=type(output)):
                class Bad(PlainOptimizer):
                    def suggest(self,n=1): return output
                rt=MockRuntime(optimizer_objects={'plain':Bad()})
                r=(await Controller(rt,MemoryStore(),self.workspace).run([spec()]))['a']
                self.assertEqual(r.status,'failed')
                self.assertFalse(any(e.action=='evaluate_started' for e in rt.events))

    async def test_mutating_observe_cannot_modify_recorded_results(self):
        class Mutates(PlainOptimizer):
            def observe(self,X,y):
                X.iloc[:,:]=99
                y[:]=99
        r=(await Controller(MockRuntime(optimizer_objects={'plain':Mutates()}),
                            MemoryStore(),self.workspace).run([spec()]))['a']
        self.assertEqual(r.status,'completed')
        self.assertEqual(r.batches[0].values,(.3125,.3125))
        self.assertEqual(r.batches[0].candidates.points,((.25,.5),(.25,.5)))

    async def test_shared_object_not_reset_for_another_repeat(self):
        obj=PlainOptimizer();rt=MockRuntime(optimizer_objects={'plain':obj})
        r=await Controller(rt,MemoryStore(),self.workspace).run([spec('a'),spec('b')])
        self.assertEqual(sorted(x.status for x in r.values()),['completed','failed'])
        self.assertTrue(any('fresh optimizer' in (x.error or '') for x in r.values()))

    async def test_cleanup_failure_does_not_abort_other_runs(self):
        class Runtime(MockRuntime):
            async def open_benchmark(self,spec,environment,seed):
                client=await super().open_benchmark(spec,environment,seed)
                if spec.name=='broken':
                    async def close():
                        client.closed=True
                        raise RuntimeError('cleanup failed')
                    client.close=close
                return client
        bad=replace(spec('bad'),task=TaskSpec('broken','legacy'))
        rt=Runtime();r=await Controller(rt,MemoryStore(),self.workspace).run([bad,spec('good')])
        self.assertEqual(r['bad'].status,'failed')
        self.assertIn('cleanup failed',r['bad'].error)
        self.assertEqual(r['good'].status,'completed')
        self.assertTrue(all(c.closed for c in [*rt.optimizers,*rt.benchmarks]))

    async def test_async_object_is_not_the_sync_protocol(self):
        class Async(PlainOptimizer):
            async def suggest(self,n=1): return super().suggest(n)
        with self.assertRaisesRegex(TypeError,'synchronous'):
            local_client(Async(),['x','y'])

    async def test_non_none_observe_is_rejected(self):
        class Bad(PlainOptimizer):
            def observe(self,X,y): return 42
        client=local_client(Bad(),['x','y'])
        with self.assertRaisesRegex(TypeError,'return None'):
            await client.observe(pd.DataFrame({'x':[0],'y':[0]}),np.array([0.]))


if __name__=='__main__': unittest.main()
