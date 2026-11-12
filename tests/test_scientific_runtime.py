"""Real CPU GP, direct baseline parity, native resume and standalone export."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
HAS_SCIENCE=all(importlib.util.find_spec(x) is not None for x in ('torch','botorch','gpytorch','pymoo'))


@unittest.skipUnless(HAS_SCIENCE,'Requires pinned scientific model environment')
class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.options={'model':{'max_attempts':1},'acquisition':{'mc_samples':32,'quadrature_size':32},
                      'search':{'_target_':'pymoo','pop_size':12,'max_generations':2,'n_offsprings':12,'reference_partitions':2}}

    def config(self, q=2, startup='internal', budget=8):
        data={'benchmark':'synthetic','tasks':['branin'],'q':[q],'repeats':1,'n_trials':budget,
              'n_init':4,'master_seed':42,'workers':1,'threads':1,
              'models':[{'name':'gp','kind':'tidyhebo','options':dict(self.options,startup=startup)}]}
        path=self.root/f'config-{q}-{startup}.json';path.write_text(json.dumps(data));return path

    def run_cli(self,config,workspace,action='run',*extra,root=ROOT):
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TIDYHEBO_CPU_PROFILE='off')
        result=subprocess.run([sys.executable,str(root/'run_experiments.py'),action,'--config',str(config),
                               '--workspace',str(workspace),'--quiet',*extra],capture_output=True,text=True,env=env,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        files=list((workspace/'results/trajectories').glob('*.json'));self.assertEqual(len(files),1)
        return json.loads(files[0].read_text())

    def direct(self,config):
        data=json.loads(config.read_text());options=dict(data['models'][0]['options'],n_init=4)
        request=self.root/'direct-request.json';result=self.root/'direct-result.json'
        request.write_text(json.dumps({'q':data['q'][0],'seed':1608637542,'budget':data['n_trials'],'options':options}))
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TIDYHEBO_CPU_PROFILE='off')
        completed=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/direct_trajectory.py'),str(request),str(result)],
                                 capture_output=True,text=True,env=env,timeout=180)
        self.assertEqual(completed.returncode,0,completed.stderr)
        return json.loads(result.read_text())

    def test_real_gp_q1_matches_direct_supplied_library(self):
        config=self.config(q=1,budget=5)
        paused=self.run_cli(config,self.root/'q1','run','--stop-after-batches','1')
        self.assertEqual(len(paused['y']),1)
        actual=self.run_cli(config,self.root/'q1','resume');expected=self.direct(config)
        self.assertEqual(actual['status'],'completed')
        for key in ('x','y','batches'):self.assertEqual(actual[key],expected[key])
        self.assertTrue(actual['provenance']['optimizer']['identity']['dependencies']['torch'].startswith('2.7.1'))

    def test_real_qmace_matches_direct_and_resumes_after_fitted_gp(self):
        config=self.config(q=2,budget=8)
        paused=self.run_cli(config,self.root/'resume','run','--stop-after-batches','3')
        self.assertEqual(paused['status'],'paused');self.assertEqual(len(paused['y']),6)
        resumed=self.run_cli(config,self.root/'resume','resume')
        expected=self.direct(config)
        for key in ('x','y','batches'):self.assertEqual(resumed[key],expected[key])
        self.assertEqual(resumed['status'],'completed')
        self.assertIsNotNone(resumed['timings'][-1])
        self.assertNotEqual(paused['provenance']['optimizer']['pid'],resumed['provenance']['optimizer']['pid'])
        # Completed skip preserves provenance and leaves no new worker logs.
        logs=list((self.root/'resume/results/logs').glob('*'))
        skipped=self.run_cli(config,self.root/'resume')
        self.assertEqual(skipped['provenance'],resumed['provenance'])
        self.assertEqual(list((self.root/'resume/results/logs').glob('*')),logs)

    def test_explicit_legacy_qmace_startup_matches_direct_reference(self):
        config=self.config(q=2,startup='legacy_qmace',budget=6)
        actual=self.run_cli(config,self.root/'legacy');expected=self.direct(config)
        self.assertEqual(actual['batches'],[4,2])
        for key in ('x','y','batches'):self.assertEqual(actual[key],expected[key])

    def test_exported_repository_runs_without_old_version(self):
        import shutil
        destination=self.root/'standalone'
        shutil.copytree(ROOT,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc','results','ready*.json','local.json','source_before037.zip'))
        self.assertFalse((destination/'old_version').exists())
        config=self.config(q=2,budget=6)
        data=json.loads(config.read_text());data['models']=[{'name':'random','kind':'random'}]
        config.write_text(json.dumps(data))
        actual=self.run_cli(config,self.root/'standalone-run',root=destination)
        self.assertEqual(actual['status'],'completed')
        self.assertEqual(len(actual['y']),6)

    def test_sobol_startup_boundary_q16_and_partial_final_batch(self):
        config=self.config(q=16,budget=6)
        actual=self.run_cli(config,self.root/'q16')
        self.assertEqual(actual['batches'],[4,2])
        self.assertEqual(len(actual['y']),6)

    def test_real_q16_batch_and_partial_final_q1_match_direct(self):
        config=self.config(q=16,budget=21)
        actual=self.run_cli(config,self.root/'joint16');expected=self.direct(config)
        self.assertEqual(actual['batches'],[4,16,1])
        for key in ('x','y','batches'):self.assertEqual(actual[key],expected[key])

    def test_current_layered_leader_runs_through_joint_api(self):
        config=self.config(q=2,budget=6)
        data=json.loads(config.read_text())
        data['models'][0]['options']['search']={
            '_target_':'layered','points':{'algorithm':'initial_only','population':12,'pool_size':16},
            'batch_init':{'anchors':False,'directions':0,'use_point_pool':False,'restarts':2,'local_steps':1},
            'batches':{'population':12,'generations':1,'reference_partitions':2,'memetic':'none'}}
        config.write_text(json.dumps(data))
        actual=self.run_cli(config,self.root/'qlogei')
        self.assertEqual(actual['status'],'completed')
        self.assertEqual(actual['batches'],[2,2,2])


class StandaloneWorkerSyntaxTests(unittest.TestCase):
    def test_oracle_outer_worker_and_simplex_remain_python37_compatible(self):
        for name in ('scientific_worker.py','olympus_adapter.py','simplex.py'):
            ast.parse((ROOT/'src/bo_benchmark/workers'/name).read_text(),feature_version=(3,7))
