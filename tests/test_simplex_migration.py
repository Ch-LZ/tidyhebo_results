"""Legacy simplex numerical semantics and actual physical feedback across IPC."""
import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORKERS=ROOT/'src/bo_benchmark/workers'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(WORKERS));sys.path.insert(0,str(ROOT/'vendor/tidyhebo'))
from simplex import SimplexTransformer
SCIENCE=all(importlib.util.find_spec(name) for name in ('torch','botorch','gpytorch','pymoo'))


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def task(k=5):
    return {'parameters':[{'name':f'x{i}','lower':-2.,'upper':2.} for i in range(k-1)],
            'direction':'minimize','metadata':{'simplex':True,'benchmark':'olympus','task':'colors_bob'}}


class SimplexNumericalTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'tests/fixtures/legacy/reference_optimizers/simplex_transformer.py').exists(),'Full bundle legacy reference required')
    def test_transform_and_inverse_match_legacy_for_all_simplex_tasks_and_q16(self):
        path=ROOT/'tests/fixtures/legacy/reference_optimizers/simplex_transformer.py'
        self.assertEqual((WORKERS/'simplex.py').read_bytes(),path.read_bytes())
        old=load(path,'legacy_simplex')
        catalog=json.loads((ROOT/'src/bo_benchmark/assets/olympus_tasks.json').read_text())
        count=0
        for name,config in catalog.items():
            if not config.get('on_simplex'):continue
            count+=1;k=len(config['params']);dim=k-1
            rng=np.random.default_rng(k)
            values=rng.uniform(-2,2,(16,dim));values[0]=2;values[1]=-2
            values[2]=np.linspace(-2,2,dim);values[3]=np.nextafter(2.,-2.)
            current=SimplexTransformer(k);previous=old.SimplexTransformer(k)
            with self.subTest(task=name):
                physical=current.transform(values)
                np.testing.assert_array_equal(physical,previous.transform(values))
                np.testing.assert_array_equal(current.inverse_transform(physical),previous.inverse_transform(physical))
                self.assertTrue(np.isfinite(physical).all());self.assertTrue((physical>=0).all())
                np.testing.assert_allclose(physical.sum(axis=1),1.,rtol=0,atol=5e-16)
                self.assertEqual(current.transform(values[0]).shape,(k,))
                self.assertEqual(current.inverse_transform(physical[0]).shape,(dim,))
        self.assertEqual(count,8)

    def test_interior_roundtrip_and_legacy_boundary_guard_are_distinct(self):
        mapper=SimplexTransformer(5)
        inside=np.random.default_rng(1).uniform(-1.8,1.8,(100,4))
        np.testing.assert_allclose(mapper.inverse_transform(mapper.transform(inside)),inside,rtol=0,atol=3e-14)
        edge=np.array([[2.,0.,-1.,1.]])
        physical=mapper.transform(edge);back=mapper.inverse_transform(physical)
        self.assertLess(physical[0,-1],1e-10)
        self.assertGreater(np.abs(back-edge).max(),2.9)  # Existing guarded inverse, not a new regression.


@unittest.skipUnless(SCIENCE,'Pinned model dependencies required')
class SimplexAdapterTests(unittest.TestCase):
    def adapter(self,startup='internal',adaptive=False):
        from modern_adapters import TidyHEBOAdapter
        options={'n_init':4,'startup':startup,'model':{'max_attempts':1}}
        if adaptive:options['search']={'_target_':'layered'}
        return TidyHEBOAdapter(task(),42,options)

    def test_internal_startup_observes_legacy_guarded_coordinates(self):
        import pandas as pd
        adapter=self.adapter();x=pd.DataFrame([[2.,0.,-1.,1.]],columns=adapter.columns)
        physical=adapter.simplex.transform(x.to_numpy())
        adapter.observe_evaluated(x,np.array([[-1.]]),physical)
        expected=adapter.core.design_space.from_DS_to_GA(pd.DataFrame(adapter.simplex.inverse_transform(physical),columns=adapter.columns))
        np.testing.assert_array_equal(adapter.core.train_x.numpy(),expected.numpy())

    def test_legacy_qmace_initial_frame_is_not_inverse_roundtripped(self):
        import pandas as pd
        adapter=self.adapter('legacy_qmace');x=pd.DataFrame([[2.,0.,-1.,1.]],columns=adapter.columns)
        physical=adapter.simplex.transform(x.to_numpy())
        expected=adapter.core.design_space.from_DS_to_GA(x).numpy().copy()
        adapter.observe_evaluated(x,np.array([[-1.]]),physical)
        np.testing.assert_array_equal(adapter.core.train_x.numpy(),expected)
        adapter.observe_evaluated(x,np.array([[-2.]]),physical)
        inverse=adapter.simplex.inverse_transform(physical)
        expected_second=adapter.core.design_space.from_DS_to_GA(pd.DataFrame(inverse,columns=adapter.columns)).numpy()
        np.testing.assert_array_equal(adapter.core.train_x.numpy()[-1:],expected_second)

    def test_authoritative_physical_values_are_used_and_wrong_composition_rejected(self):
        import pandas as pd
        adapter=self.adapter();x=pd.DataFrame([[0.,.4,-.8,1.]],columns=adapter.columns)
        physical=adapter.simplex.transform(x.to_numpy());physical[0,0]+=5e-14;physical[0,1]-=5e-14
        adapter.observe_evaluated(x,np.array([[1.]]),physical)
        expected=adapter.core.design_space.from_DS_to_GA(pd.DataFrame(adapter.simplex.inverse_transform(physical),columns=adapter.columns)).numpy()
        np.testing.assert_array_equal(adapter.core.train_x.numpy(),expected)
        before=adapter.core.train_x.clone()
        foreign=physical[:,::-1].copy()
        with self.assertRaisesRegex(ValueError,'does not match'):adapter.observe_evaluated(x,np.array([[1.]]),foreign)
        self.assertTrue(before.equal(adapter.core.train_x))

    def test_whole_q16_batch_uses_guarded_coordinates(self):
        import pandas as pd
        adapter=self.adapter(adaptive=True)
        x=np.random.default_rng(2).uniform(-2,2,(16,4));x[0]=[2,0,-1,1]
        frame=pd.DataFrame(x,columns=adapter.columns)
        physical=adapter.simplex.transform(x)
        adapter.observe_evaluated(frame,np.arange(16,dtype=float).reshape(-1,1),physical)
        expected=adapter.core.design_space.from_DS_to_GA(pd.DataFrame(
            adapter.simplex.inverse_transform(physical),columns=adapter.columns)).numpy()
        np.testing.assert_array_equal(adapter.core.train_x.numpy(),expected)

    def test_reordered_physical_batch_is_rejected_without_observing(self):
        import pandas as pd
        adapter=self.adapter(adaptive=True)
        frame=pd.DataFrame(np.random.default_rng(3).uniform(-1,1,(16,4)),columns=adapter.columns)
        with self.assertRaisesRegex(ValueError,'does not match'):
            adapter.observe_evaluated(frame,np.zeros((16,1)),adapter.simplex.transform(frame.to_numpy())[::-1])
        self.assertEqual(len(adapter.core.train_y),0)


class PhysicalFeedbackRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_passes_exact_benchmark_physical_points_without_changing_math_X(self):
        from bo_benchmark.controller import Controller
        from bo_benchmark.domain import RunSpec,OptimizerSpec,TaskSpec,ProtocolSpec
        from bo_benchmark.mocks import MockRuntime,MemoryStore
        from bo_benchmark.workspace import Workspace
        received=[]
        class Runtime(MockRuntime):
            async def open_optimizer(self,*args):
                client=await super().open_optimizer(*args);original=client.observe
                async def observe_evaluated(X,y,physical):
                    received.append((X.copy(),np.asarray(physical),y.copy()))
                    await original(X,y)
                client.observe_evaluated=observe_evaluated
                return client
            async def open_benchmark(self,*args):
                client=await super().open_benchmark(*args);original=client.evaluate
                async def evaluate(batch):
                    result=await original(batch)
                    return replace(result,physical_points=tuple(tuple(v+10 for v in row) for row in batch.points))
                client.evaluate=evaluate
                return client
        with tempfile.TemporaryDirectory() as directory:
            run=RunSpec('physical',OptimizerSpec('random','opt'),TaskSpec('quadratic','bench'),ProtocolSpec(3,2),42)
            result=(await Controller(Runtime(),MemoryStore(),Workspace(Path(directory))).run([run]))['physical']
        self.assertEqual(result.status,'completed');self.assertEqual(len(received),2)
        for (frame,physical,y),batch in zip(received,result.batches):
            np.testing.assert_array_equal(physical,np.asarray(batch.physical_points))
            np.testing.assert_array_equal(frame,np.asarray(batch.candidates.points))
            np.testing.assert_array_equal(y,-np.asarray(batch.values).reshape(-1,1))


@unittest.skipUnless(SCIENCE,'Pinned model dependencies required')
class SimplexProcessParityTests(unittest.TestCase):
    def test_q16_simplex_pipeline_and_resume_match_legacy_loop_with_fake_oracle(self):
        import os,subprocess
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fake=folder/'framework';fake.mkdir()
            (fake/'tensorflow.py').write_text("__version__='FAKE-contract-only'\nclass Graph:\n def __init__(self): self.seed=None\ndef set_random_seed(seed): pass\n")
            (fake/'olympus.py').write_text("import numpy as np\nimport tensorflow as tf\n__version__='FAKE-contract-only'\nclass Emulator:\n def __init__(self,*args): self.graph=tf.Graph()\n def run(self,x):\n  x=np.asarray(x);np.testing.assert_allclose(x.sum(axis=1),1.)\n  return x@np.arange(1,x.shape[1]+1),None,None\n")
            options={'startup':'legacy_qmace','n_init':4,'model':{'max_attempts':1},
                'acquisition':{'mc_samples':16,'quadrature_size':16},
                'search':{'_target_':'layered',
                    'point_init':{'anchors':False,'directions':0,'restarts':2,'local_steps':1,'use_point_pool':False},
                    'batch_init':{'anchors':False,'directions':0,'restarts':2,'local_steps':1,'use_point_pool':False},
                    'points':{'algorithm':'initial_only','population':12,'pool_size':16},
                    'batches':{'population':12,'generations':1,'reference_partitions':2,'memetic':'none'}}}
            config={'benchmark':'olympus','tasks':['colors_bob'],'q':[16],'repeats':1,'n_trials':21,'n_init':4,
                'workers':1,'master_seed':42,'models':[{'name':'adaptive','options':options}],
                'environments':{'model':{'python':sys.executable,'sources':[str(ROOT/'vendor/tidyhebo')],'threads':1},
                    'benchmark':{'python':sys.executable,'sources':[str(fake)],'threads':1}}}
            path=folder/'config.json';path.write_text(json.dumps(config))
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TIDYHEBO_CPU_PROFILE='off')
            def run(action,*extra):
                result=subprocess.run([sys.executable,str(ROOT/'run_experiments.py'),action,'--config',str(path),
                    '--workspace',str(folder/'campaign'),'--quiet',*extra],capture_output=True,text=True,env=env,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                return json.loads(next((folder/'campaign/results/trajectories').glob('*.json')).read_text())
            paused=run('run','--stop-after-batches','2');self.assertEqual(paused['status'],'paused')
            actual=run('resume');self.assertEqual(actual['status'],'completed');self.assertEqual(actual['batches'],[4,16,1])
            catalog=json.loads((ROOT/'src/bo_benchmark/assets/olympus_tasks.json').read_text())
            request={'seed':1608637542,'columns':[p['name'] for p in catalog['colors_bob']['params'][:-1]],'options':options}
            req=folder/'request.json';output=folder/'reference.json';req.write_text(json.dumps(request))
            direct=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/direct_simplex_trajectory.py'),str(req),str(output)],
                capture_output=True,text=True,env=env,timeout=180)
            self.assertEqual(direct.returncode,0,direct.stderr)
            expected=json.loads(output.read_text())
            for key in ('x','y','batches'):self.assertEqual(actual[key],expected[key],key)
            self.assertNotEqual(paused['provenance']['optimizer']['pid'],actual['provenance']['optimizer']['pid'])
