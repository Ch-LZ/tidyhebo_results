import pytest
import numpy as np
import yaml
import tempfile
from pathlib import Path

from src.optimizer_adaptors.model_lib import Model, make_y_point

pytestmark = pytest.mark.env_tidyhebo


@pytest.fixture(scope="function")
def logei_optimizer():
    pytest.importorskip("tidyhebo.optimizer")
    from src.optimizer_adaptors.logei_opt import LogEIOptimizer
    from tidyhebo.mixed_params import RealP

    params = [
        RealP(0, 0.0, 1.0, name="red"),
        RealP(1, 0.0, 1.0, name="orange"),
        RealP(2, 0.0, 1.0, name="yellow"),
        RealP(3, 0.0, 1.0, name="blue"),
        RealP(4, 0.0, 1.0, name="green"),
    ]
    return LogEIOptimizer(params=params, task_name='testing_task', task_minimization=True, on_simplex=False, n_init=4)


def test_registration():
    pytest.importorskip("tidyhebo.optimizer")
    from src.optimizer_adaptors.logei_opt import LogEIOptimizer
    assert 'logei' in Model._MODEL_REGISTRY
    assert Model._MODEL_REGISTRY['logei'] == 'src.optimizer_adaptors.logei_opt.LogEIOptimizer'


def test_from_config():
    pytest.importorskip("tidyhebo.optimizer")
    from src.optimizer_adaptors.logei_opt import LogEIOptimizer

    params = [{'name': 'red', 'type': 'continuous', 'low': 0.0, 'high': 1.0}]
    config = {
        'task_name': 'testing_task',
        'params': params,
        'less_is_better': True,
        'on_simplex': False
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        cfg_path = f.name

    optimizer = Model.from_config('logei', cfg_path)
    assert isinstance(optimizer, LogEIOptimizer)
    assert optimizer.task_minimization is True
    assert optimizer.n_init == 4
    Path(cfg_path).unlink()


def test_suggest_shape(logei_optimizer):
    optimizer = logei_optimizer
    point = optimizer.suggest()
    assert point.shape == (1, len(optimizer.params))


def test_suggest_values_within_bounds(logei_optimizer):
    optimizer = logei_optimizer
    for _ in range(optimizer.n_init):
        point = optimizer.suggest()
        for i, p in enumerate(optimizer.params):
            assert p.ds_lb <= point[0, i] <= p.ds_ub
        optimizer.observe(point, make_y_point(np.array([[0.5]])))
    point = optimizer.suggest()
    for i, p in enumerate(optimizer.params):
        assert p.ds_lb <= point[0, i] <= p.ds_ub


def test_observe_and_suggest(logei_optimizer):
    optimizer = logei_optimizer
    x = optimizer.suggest()
    y = make_y_point(np.array([[0.5]]))
    optimizer.observe(x, y)
    assert optimizer.optimizer._init_design_generator is not None
    for _ in range(optimizer.n_init - 1):
        x = optimizer.suggest()
        y = make_y_point(np.array([[0.3]]))
        optimizer.observe(x, y)
    x = optimizer.suggest()
    assert x.shape == (1, len(optimizer.params))


def test_observe_with_1d_y(logei_optimizer):
    optimizer = logei_optimizer
    x = optimizer.suggest()
    y = make_y_point(np.array([[0.7]]))
    optimizer.observe(x, y)  # should not raise


def test_minimization_sign():
    pytest.importorskip("tidyhebo.optimizer")
    from src.optimizer_adaptors.logei_opt import LogEIOptimizer
    from tidyhebo.mixed_params import RealP

    params = [RealP(0, 0.0, 1.0, name="red")]
    opt_max = LogEIOptimizer(task_name='testing_task', params=params, task_minimization=False, on_simplex=False, n_init=1)
    x = opt_max.suggest()
    y = make_y_point(np.array([[0.7]]))

    original_observe = opt_max.optimizer.observe
    captured_y = None

    def mock_observe(df, y_tensor):
        nonlocal captured_y
        captured_y = y_tensor.item()
        original_observe(df, y_tensor)

    opt_max.optimizer.observe = mock_observe
    opt_max.observe(x, y)
    assert captured_y == 0.7


def test_invalid_param_type():
    from src.optimizer_adaptors.logei_opt import LogEIOptimizer
    invalid_params = [
        {'name': 'red', 'type': 'integer', 'low': 0, 'high': 10},
    ]
    with pytest.raises(ValueError, match="Only 'continuous' parameters are supported"):
        LogEIOptimizer._from_config_impl(invalid_params, task_name='testing_task', on_simplex=False, task_minimization=True)


def test_get_history(logei_optimizer):
    optimizer = logei_optimizer
    history = optimizer.get_history()
    assert history.shape == (1,0)
    x1 = optimizer.suggest()
    y1 = make_y_point(np.array([[0.5]]))
    optimizer.observe(x1, y1)
    x2 = optimizer.suggest()
    y2 = make_y_point(np.array([[0.7]]))
    optimizer.observe(x2, y2)
    history = optimizer.get_history()
    np.testing.assert_allclose(history, [[0.5, 0.7],])