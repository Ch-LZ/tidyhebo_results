import numpy as np

from reproducibility.metrics import calculate_metrics, incumbent_utility, scale_utility


def test_scale_utility_handles_minimization_by_reference_order():
    values = np.array([10.0, 5.0, 0.0])
    np.testing.assert_allclose(scale_utility(values, y_best=0.0, y_worst=10.0), [0.0, 0.5, 1.0])


def test_sequential_incumbent_is_cumulative_maximum():
    utility = np.array([[0.1, 0.4, 0.2, 0.7]])
    np.testing.assert_allclose(incumbent_utility(utility), [[0.1, 0.4, 0.4, 0.7]])


def test_adaptive_incumbent_updates_at_batch_end_after_initialization():
    utility = np.array([[0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.8, 0.6]])
    observed = incumbent_utility(
        utility,
        batch_sizes=[[4, 2, 2]],
        sequential_initial_evaluations=4,
    )
    np.testing.assert_allclose(observed, [[0.1, 0.2, 0.3, 0.4, 0.4, 0.9, 0.9, 0.9]])


def test_metrics_return_expected_fields():
    values = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.5]])
    metrics = calculate_metrics(values, y_best=1.0, y_worst=0.0)
    assert set(metrics) == {"median_nAUC", "IQR_nAUC", "CVaR_nAUC"}
