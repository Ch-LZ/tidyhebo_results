import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from bo_benchmark.analysis.metrics import scale_utility,incumbent_utility,calculate_metrics,regret_auc


class AnalysisMetricsTests(unittest.TestCase):
    def test_minimization_reference_order(self):
        np.testing.assert_array_equal(scale_utility([10.,5.,0.],0.,10.),[0.,.5,1.])

    def test_batch_feedback_preserves_historical_initialization(self):
        observed=incumbent_utility(np.array([[.1,.2,.3,.4,.9,.5,.8,.6]]),batch_sizes=[[4,2,2]],sequential_initial_evaluations=4)
        np.testing.assert_array_equal(observed,[[.1,.2,.3,.4,.4,.9,.9,.9]])

    def test_strict_feedback_and_legacy_convention_are_distinct(self):
        values=np.array([[.1,.2,.3,.4]])
        np.testing.assert_array_equal(incumbent_utility(values,batch_sizes=[[4]],sequential_initial_evaluations=0),[[0.,0.,0.,.4]])
        self.assertGreater(regret_auc(values,1,0,batch_sizes=[[4]],sequential_initial_evaluations=0)[0],
                           regret_auc(values,1,0,batch_sizes=[[4]],sequential_initial_evaluations=4)[0])

    def test_manuscript_fields(self):
        metrics=calculate_metrics(np.array([[0.,.5,1.],[0.,.25,.5]]),1.,0.)
        self.assertEqual(set(metrics),{'median_nAUC','IQR_nAUC','CVaR_nAUC'})
