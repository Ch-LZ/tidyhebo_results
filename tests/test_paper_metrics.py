"""Numerical contracts that must survive the publication cleanup."""
import unittest
import numpy as np
from bo_benchmark.analysis.metrics import (
    average_ranks, best_so_far, trajectory_metrics, upper_tail_mean,
    paired_mean_interval, olympus_run_table, olympus_statistics,
)
from bo_benchmark.task_catalog import task_reference


class PaperMetricTests(unittest.TestCase):
    def test_area_scale_and_batch_feedback(self):
        y = [0.,1.,3.,2.,5.]
        sequential = trajectory_metrics(y,5.,0.,n_init=2)
        batch = trajectory_metrics(y,5.,0.,batch_sizes=[[2,2,1]],n_init=2)
        self.assertAlmostEqual(sequential['nAUC'],2.1)
        self.assertAlmostEqual(batch['nAUC'],2.5)
        self.assertAlmostEqual(batch['mean_regret_auc'],2.5/4)
        self.assertEqual(batch['final_regret'],0.)
        np.testing.assert_array_equal(best_so_far([3.,2.,4.]),[3.,2.,2.])

    def test_tail_includes_ties_and_uses_three_worst_of_thirty(self):
        self.assertEqual(upper_tail_mean([0.,0.,1.,1.]),1.)
        self.assertEqual(upper_tail_mean(np.arange(1.,31.)),29.)
        np.testing.assert_array_equal(average_ranks([3.,1.,1.,2.]),[4.,1.5,1.5,3.])

    def test_common_bootstrap_draws_and_fixed_seed(self):
        x = np.arange(30.)
        interval = paired_mean_interval(np.column_stack((x,x+2)),draws=100,seed=42)
        np.testing.assert_allclose(interval[:,1]-interval[:,0],2.,atol=1e-14)
        np.testing.assert_array_equal(interval,paired_mean_interval(
            np.column_stack((x,x+2)),draws=100,seed=42))

    def records(self):
        ref = task_reference('hplc')['normalization']
        result = {}
        for q in (1,4):
            for i in range(30):
                for method,final in [('a',float(i)),('b',float((i+4)%30))]:
                    y = ref['y_best']+(ref['y_worst']-ref['y_best'])*final
                    s = dict(task='hplc',variant=method,q=q,seed=i,repeat=i,n_init=4,n_trials=100,startup='legacy_qmace')
                    result['hplc',method,q,i] = dict(spec=s,status='completed',y=[y]*100,
                        batches=[4]+([q]*((100-4)//q)),initial_x=[[0.1]]*4,minimize=ref['y_best'] < ref['y_worst'])
        return result

    def test_tail_difference_is_not_tail_of_seed_differences(self):
        records = self.records()
        runs = olympus_run_table(records,['hplc'],['a','b'],[1,4],list(range(30)))
        _,summary = olympus_statistics(runs,['a','b'],[1,4],baseline='b',draws=100)
        differences = summary[(summary.panel=='cvar_delta') & (summary.optimizer=='a')]
        np.testing.assert_allclose(differences['mean'],0.,atol=1e-10)
        self.assertGreater(upper_tail_mean(np.arange(30.)-np.roll(np.arange(30.),-4)),0.)

    def test_missing_seed_and_unpaired_initial_design_fail(self):
        records = self.records()
        records.pop(('hplc','a',1,0))
        with self.assertRaisesRegex(ValueError,'coverage'):
            olympus_run_table(records,['hplc'],['a','b'],[1,4],list(range(30)))
        records = self.records()
        records['hplc','a',4,0]['initial_x'] = [[0.2]]*4
        with self.assertRaisesRegex(ValueError,'Unpaired initial'):
            olympus_run_table(records,['hplc'],['a','b'],[1,4],list(range(30)))


if __name__ == '__main__':
    unittest.main()
