"""Contract test with fake framework modules, NOT an actual TensorFlow smoke."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class OlympusContractTests(unittest.TestCase):
    def test_graph_seed_simplex_raw_sign_and_replay_contract(self):
        if importlib.util.find_spec('scipy') is None:self.skipTest('Requires scipy')
        path=ROOT/'src/bo_benchmark/workers'
        sys.path.insert(0,str(path));self.addCleanup(lambda:sys.path.remove(str(path)))
        spec=importlib.util.spec_from_file_location('ported_olympus',path/'olympus_adapter.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        graphs=[]
        class Graph:
            def __init__(self):self.seed=None;graphs.append(self)
        fake_tf=types.ModuleType('tensorflow');fake_tf.Graph=Graph
        fake_tf.set_random_seed=lambda seed:None;fake_tf.__version__='FAKE-contract-only'
        fake_olympus=types.ModuleType('olympus');fake_olympus.__version__='FAKE-contract-only'
        fake_olympus.__file__=str(path/'olympus_adapter.py')
        class Emulator:
            def __init__(self,name,kind):
                self.graph=Graph();self.name=name
            def run(self,x):
                self_x=np.asarray(x)
                if self.name=='colors_bob':
                    np.testing.assert_allclose(self_x.sum(axis=1),1.)
                    assert self_x.shape[1]==5
                return self_x.sum(axis=1)+np.random.normal(0,.01,len(x)),None,None
        fake_olympus.Emulator=Emulator
        with patch.dict(sys.modules,{'tensorflow':fake_tf,'olympus':fake_olympus}):
            first=module.OlympusBenchmark('colors_bob',42,{})
            self.assertEqual(graphs[-1].seed,42)
            self.assertEqual(len(first.description['parameters']),4)
            self.assertEqual(first.description['direction'],'minimize')
            x=np.zeros((3,4));y,physical=first.evaluate(x)
            self.assertTrue((y>0).all())  # Adapter returns RAW objective; no sign flip here.
            state=first.save_state()
            expected,_=first.evaluate(x+.2)
            self.assertEqual(len(state['history']),1)
            resumed=module.OlympusBenchmark('colors_bob',42,{})
            resumed.load_state(state)
            actual,_=resumed.evaluate(x+.2)
            np.testing.assert_array_equal(actual,expected)
