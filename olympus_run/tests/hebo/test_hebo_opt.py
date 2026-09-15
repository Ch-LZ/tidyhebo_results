import pytest
import numpy as np
import yaml
import tempfile
from pathlib import Path

from src.optimizer_adaptors.model_lib import Model, make_y_point

pytestmark = pytest.mark.env_hebo

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="hebo.optimizers.hebo")


@pytest.fixture(scope="function")
def hebo_optimizer():
    pytest.importorskip("hebo")
    from src.optimizer_adaptors.hebo_opt import HeboOptimizer
    params = [
        {'name': 'red',    'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'orange', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'yellow', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'blue',   'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'green',  'type': 'continuous', 'low': 0.0, 'high': 1.0},
    ]
    return HeboOptimizer(task_name='testing_task', params=params, task_minimization=True, on_simplex=False)


def test_registration():
    pytest.importorskip("hebo")
    from src.optimizer_adaptors.hebo_opt import HeboOptimizer
    assert 'hebo' in Model._MODEL_REGISTRY
    assert Model._MODEL_REGISTRY['hebo'] == 'src.optimizer_adaptors.hebo_opt.HeboOptimizer'


def test_from_config():
    pytest.importorskip("hebo")
    from src.optimizer_adaptors.hebo_opt import HeboOptimizer
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
    optimizer = Model.from_config('hebo', cfg_path)
    assert isinstance(optimizer, HeboOptimizer)
    assert optimizer.task_minimization is True
    Path(cfg_path).unlink()


def test_suggest_shape(hebo_optimizer):
    optimizer = hebo_optimizer
    point = optimizer.suggest()
    expected_dim = len(optimizer.design_space.para_names)
    assert point.shape == (1, expected_dim)


def test_suggest_values_within_bounds(hebo_optimizer):
    optimizer = hebo_optimizer
    lb = optimizer.design_space.opt_lb.numpy()
    ub = optimizer.design_space.opt_ub.numpy()
    for _ in range(5):
        point = optimizer.suggest()
        for i in range(len(lb)):
            assert lb[i] <= point[0, i] <= ub[i]


def test_observe_updates_optimizer(hebo_optimizer):
    optimizer = hebo_optimizer
    point = optimizer.suggest()
    y = make_y_point(np.array([[0.5]]))
    optimizer.observe(point, y)
    assert len(optimizer.optimizer.X) == 1
    assert len(optimizer.optimizer.y) == 1
    np.testing.assert_array_equal(optimizer.optimizer.X.values, point)
    y_obs = optimizer.optimizer.y[0]
    if hasattr(y_obs, 'item'):
        y_obs = y_obs.item()
    np.testing.assert_almost_equal(y_obs, 0.5)


def test_observe_minimization_sign():
    pytest.importorskip("hebo")
    from src.optimizer_adaptors.hebo_opt import HeboOptimizer
    params = [{'name': 'red', 'type': 'continuous', 'low': 0.0, 'high': 1.0}]
    opt_min = HeboOptimizer(params=params, task_name='testing_task', task_minimization=True, on_simplex=False)
    point_min = opt_min.suggest()
    y_min = make_y_point(np.array([[0.7]]))
    opt_min.observe(point_min, y_min)
    y_obs_min = opt_min.optimizer.y[0]
    if hasattr(y_obs_min, 'item'):
        y_obs_min = y_obs_min.item()
    assert y_obs_min == 0.7

    opt_max = HeboOptimizer(task_name='testing_task', params=params, task_minimization=False, on_simplex=False)
    point_max = opt_max.suggest()
    y_max = make_y_point(np.array([[0.7]]))
    opt_max.observe(point_max, y_max)
    y_obs_max = opt_max.optimizer.y[0]
    if hasattr(y_obs_max, 'item'):
        y_obs_max = y_obs_max.item()
    assert y_obs_max == -0.7


def test_suggest_and_observe_cycle(hebo_optimizer):
    optimizer = hebo_optimizer
    for i in range(3):
        point = optimizer.suggest()
        y = make_y_point(np.array([[i * 0.1]]))
        optimizer.observe(point, y)
    y_observed = optimizer.optimizer.y
    if hasattr(y_observed, 'detach'):
        y_observed = y_observed.detach().numpy()
    else:
        y_observed = np.asarray(y_observed)
    y_observed = y_observed.flatten()
    expected = np.array([0.0, 0.1, 0.2])
    np.testing.assert_almost_equal(y_observed, expected, decimal=6)


def test_observe_without_suggest_creates_cols(hebo_optimizer):
    optimizer = hebo_optimizer
    x0 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    y0 = make_y_point(np.array([[0.9]]))
    optimizer.observe(x0, y0)
    assert optimizer._cols is not None
    assert len(optimizer.optimizer.X) == 1


def test_invalid_param_type():
    from src.optimizer_adaptors.hebo_opt import HeboOptimizer
    invalid_params = [
        {'name': 'red', 'type': 'integer', 'low': 0, 'high': 10},
        {'name': 'green', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
    ]
    with pytest.raises(AssertionError):
        HeboOptimizer(task_name='testing_task', params=invalid_params, task_minimization=True, on_simplex=False)


def test_get_history(hebo_optimizer):
    optimizer = hebo_optimizer
    history = optimizer.get_history()
    assert history.shape == (1, 0)
    x1 = optimizer.suggest()
    y1 = make_y_point(np.array([[0.5]]))
    optimizer.observe(x1, y1)
    x2 = optimizer.suggest()
    y2 = make_y_point(np.array([[0.7]]))
    optimizer.observe(x2, y2)
    history = optimizer.get_history()
    np.testing.assert_allclose(history, [[0.5, 0.7],])