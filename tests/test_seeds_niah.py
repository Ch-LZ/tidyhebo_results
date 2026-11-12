import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests/fixtures'))
from bo_benchmark.seeds import SeedManager
from bo_benchmark.campaign import plan
from bo_benchmark.cli import prepare
from niah_data import prepared

SCIENCE=all(importlib.util.find_spec(x) for x in ('torch','botorch','gpytorch','pymoo'))


class SeedTests(unittest.TestCase):
    def test_exact_legacy_schedule_and_saved_roundtrip(self):
        reference=np.random.RandomState(42)
        manager=SeedManager(42,30)
        self.assertEqual(manager.seeds,[int(reference.randint(0,2**31-1)) for _ in range(30)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'seeds.json';manager.save(path)
            self.assertEqual(SeedManager.load(path).state_dict(),manager.state_dict())
        for master in (-1,True,2**32):
            with self.assertRaises(ValueError):SeedManager(master)
        modified=manager.state_dict();modified['seeds'][0]+=1
        with self.assertRaises(ValueError):SeedManager.from_state(modified)

    def test_framework_rng_capture_and_restore(self):
        SeedManager.apply(42,include_torch=False)
        state=SeedManager.capture_rng(include_torch=False)
        expected=(random.random(),np.random.random(5))
        SeedManager.restore_rng(state)
        self.assertEqual(random.random(),expected[0]);np.testing.assert_array_equal(np.random.random(5),expected[1])
        if SCIENCE:
            import torch
            SeedManager.apply(42);state=SeedManager.capture_rng();expected=torch.rand(4)
            SeedManager.restore_rng(state);self.assertTrue(torch.equal(torch.rand(4),expected))

    def test_one_schedule_for_suites_methods_q_and_repeat_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';workspace=Path(directory)/'campaign'
            base={'benchmark':'synthetic','tasks':['ackley','hartmann'],'q':[1,4,16],
                  'repeats':3,'n_init':4,'n_trials':6,'master_seed':42,
                  'models':[{'name':'gp'},{'name':'random','kind':'random'}]}
            path.write_text(json.dumps(base));original=plan(path,workspace);prepare(original)
            manager=SeedManager(42,3)
            for spec in original['specs']:
                self.assertEqual(spec.seed,manager.seed_for(original['labels'][spec.run_id]['repeat']))
            base['repeats']=1;path.write_text(json.dumps(base));prepare(plan(path,workspace))
            saved=json.loads((workspace/'results/seeds.json').read_text())
            self.assertEqual(saved,manager.state_dict())

    def test_niah_plan_and_finite_protocol_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            base={'benchmark':'niah','tasks':'all','q':[1],'repeats':1,'n_trials':7,'models':[{'name':'gp'}]}
            path.write_text(json.dumps(base));planned=plan(path)
            self.assertEqual({s.task.name for s in planned['specs']},{'poisson_ratio','thermoelectric','wildfire'})
            base['q']=[4];path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError,'q=1'):plan(path)

    def test_candidates_are_feature_only_and_integrity_checked(self):
        from niah_assets import load_candidates,load_asset
        with tempfile.TemporaryDirectory() as directory:
            data=prepared(directory)
            arrays,meta=load_candidates(directory,data['wildfire'][1]['asset_id'])
            self.assertNotIn('y',arrays)
            oracle=load_asset(directory,'poisson_ratio')
            np.testing.assert_array_equal(oracle([[.2,.1],[.9,.1]]),[-2.2,.4])
            arrays['y']=np.arange(len(arrays['x']))
            np.savez_compressed(Path(directory)/'wildfire_candidates.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'no targets'):load_candidates(directory,data['wildfire'][1]['asset_id'])


@unittest.skipUnless(SCIENCE,'Pinned scientific dependencies required')
class NIAHRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.assets=prepared(self.root/'data')
        self.options={'model':{'max_attempts':1},'acquisition':{'mc_samples':16,'quadrature_size':16},
                      'search':{'_target_':'pymoo','pop_size':12,'n_offsprings':12,'max_generations':2,'reference_partitions':2}}

    def config(self,tasks,models=None,budget=7):
        data={'benchmark':'niah','tasks':tasks,'data_dir':str(self.root/'data'),'candidate_limit':0,
              'q':[1],'repeats':1,'n_trials':budget,'n_init':4,'master_seed':42,'workers':1,'threads':1,
              'models':models or [{'name':'gp','options':self.options}]}
        path=self.root/'config.json';path.write_text(json.dumps(data));return path

    def run_cli(self,config,workspace,action='run',extra=()):
        result=subprocess.run([sys.executable,str(ROOT/'run_experiments.py'),action,'--config',str(config),
                               '--workspace',str(workspace),'--quiet',*extra],capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return [json.loads(p.read_text()) for p in sorted((workspace/'results/trajectories').glob('*.json'))]

    def test_wildfire_real_gp_resume_never_repeats_and_matches_direct_trajectory(self):
        config=self.config(['wildfire'])
        paused=self.run_cli(config,self.root/'resume',extra=('--stop-after-batches','5'))[0]
        self.assertEqual(len(paused['y']),5)
        resumed=self.run_cli(config,self.root/'resume','resume')[0]
        direct=self.run_cli(config,self.root/'direct')[0]
        for key in ('optimizer_x','x','y','candidate_ids'):self.assertEqual(resumed[key],direct[key])
        ids=resumed['candidate_ids'];self.assertEqual(len(set(ids)),7)
        arrays=self.assets['wildfire'][0]
        np.testing.assert_array_equal(resumed['y'],arrays['y'][ids])
        np.testing.assert_array_equal(np.asarray(resumed['x'])[:,0,:],arrays['x'][ids]*arrays['span']+arrays['low'])

    def test_all_three_niah_tasks_run_with_shared_seeds(self):
        records=self.run_cli(self.config(['poisson_ratio','thermoelectric','wildfire'],budget=5),self.root/'all')
        self.assertEqual(len(records),3)
        self.assertEqual({r['status'] for r in records},{'completed'})
        self.assertEqual({r['spec']['seed'] for r in records},{1608637542})
        self.assertTrue(all(r['minimize'] for r in records))
        self.assertEqual({r['provenance']['benchmark']['identity']['adapter']['kind'] for r in records},{'forest','table'})

    def test_finite_random_has_same_initial_rows_as_gp_without_targets(self):
        models=[{'name':'gp','options':self.options},{'name':'random','kind':'random'}]
        records=self.run_cli(self.config(['wildfire'],models=models,budget=5),self.root/'paired')
        self.assertEqual(records[0]['candidate_ids'][:4],records[1]['candidate_ids'][:4])
        for record in records:self.assertEqual(len(set(record['candidate_ids'])),5)

    def test_ackley_hartmann_are_real_six_dimensional_functions(self):
        from botorch.test_functions.synthetic import Ackley,Hartmann
        import torch
        data={'benchmark':'synthetic','tasks':['ackley','hartmann'],'q':[2],'repeats':1,
              'n_trials':6,'n_init':4,'master_seed':42,'workers':1,
              'models':[{'name':'gp','options':self.options}]}
        path=self.root/'synthetic.json';path.write_text(json.dumps(data))
        records=self.run_cli(path,self.root/'synthetic')
        for record in records:
            x=np.asarray([point for batch in record['optimizer_x'] for point in batch]);self.assertEqual(x.shape,(6,6))
            task=Ackley(dim=6) if record['spec']['task']=='ackley' else Hartmann(dim=6)
            expected=task.double()(torch.tensor(x,dtype=torch.double)).detach().numpy()
            np.testing.assert_array_equal(record['y'],expected)
