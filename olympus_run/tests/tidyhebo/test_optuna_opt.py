# tests/tidyhebo/test_optuna_opt.py

import pytest
import numpy as np
import yaml
from pathlib import Path

from src.optimizer_adaptors.model_lib import make_x_point, make_y_point

pytestmark = pytest.mark.env_tidyhebo


# ---------- Fixtures ----------
@pytest.fixture
def config_continuous(tmp_path):
    config = {
        'task_name': 'testing_task',
        'params': [
            {'name': 'x1', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
            {'name': 'x2', 'type': 'continuous', 'low': -2.0, 'high': 2.0}
        ],
        'on_simplex': False,
        'less_is_better': False
    }
    path = tmp_path / 'config.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def config_with_min_max(tmp_path):
    config = {
        'task_name': 'testing_task',
        'params': [
            {'name': 'a', 'type': 'continuous', 'min': 0, 'max': 10},
            {'name': 'b', 'type': 'continuous', 'min': -5, 'max': 5}
        ],
        'less_is_better': True
    }
    path = tmp_path / 'config2.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


# ---------- Tests ----------
def test_from_config_with_low_high(config_continuous):
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    model = OptunaOptimizer.from_config('optuna', config_continuous)
    assert isinstance(model, OptunaOptimizer)
    assert model.on_simplex is False
    assert model.task_minimization is False
    assert model.dim == 2
    assert model.bounds == [(0.0, 1.0), (-2.0, 2.0)]
    assert model.names == ['x1', 'x2']


def test_from_config_with_min_max(config_with_min_max):
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    with pytest.raises(KeyError):
        OptunaOptimizer.from_config('optuna', config_with_min_max)


def test_from_config_missing_params(tmp_path):
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    config = {'on_simplex': False}
    path = tmp_path / 'bad.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="Missing 'params' section in config."):
        OptunaOptimizer.from_config('optuna', path)


def test_from_config_unsupported_type(tmp_path):
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    config = {
        'task_name': 'testing_task',
        'params': [{'name': 'x', 'type': 'categorical', 'values': ['a', 'b']}]
    }
    path = tmp_path / 'bad.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="only 'continuous' is supported"):
        OptunaOptimizer.from_config('optuna', path)


def test_suggest_and_observe():
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    params = [
        {'name': 'a', 'low': 0.0, 'high': 1.0},
        {'name': 'b', 'low': -1.0, 'high': 1.0}
    ]
    model = OptunaOptimizer(params=params, task_name='testing_task', on_simplex=False, task_minimization=False)

    x1 = model.suggest()
    assert isinstance(x1, np.ndarray)
    assert x1.shape == (1, 2)
    assert np.all(x1 >= [0.0, -1.0]) and np.all(x1 <= [1.0, 1.0])

    y1 = make_y_point(np.array([[2.5]]))
    model.observe(x1, y1)
    assert model._last_trial_number is not None

    x2 = model.suggest()
    assert x2.shape == (1, 2)
    y2 = make_y_point(np.array([[1.0]]))
    model.observe(x2, y2)

    assert len(model.study.trials) == 2
    values = [t.value for t in model.study.trials]
    assert values == [2.5, 1.0]


def test_minimization_conversion():
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    params = [{'name': 'x', 'low': 0, 'high': 1}]
    model = OptunaOptimizer(params=params, task_name='testing_task', on_simplex=False, task_minimization=True)

    x = model.suggest()
    y_orig = make_y_point(np.array([[5.0]]))
    model.observe(x, y_orig)
    trial = model.study.trials[0]
    assert trial.value == -5.0


def test_observe_without_suggest_raises():
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    params = [{'name': 'x', 'low': 0, 'high': 1}]
    model = OptunaOptimizer(params=params, task_name='testing_task')
    y = make_y_point(np.array([[0.5]]))
    x = np.array([[0.0]])
    with pytest.raises(ValueError, match=r'suggest\(\) must be called before observe\(\)'):
        model.observe(x, y)


def test_simplex_mode():
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    params = [{'name': 'x', 'low': 0, 'high': 1}]
    model = OptunaOptimizer(params=params, task_name='testing_task', on_simplex=True, task_minimization=False)
    assert model.on_simplex is True
    assert model.to_simplex_transformer is None


def test_get_history():
    pytest.importorskip("optuna")
    from src.optimizer_adaptors.optuna_opt import OptunaOptimizer

    params = [{'name': 'x', 'low': 0, 'high': 1}]
    model = OptunaOptimizer(params=params, task_name='testing_task', on_simplex=False, task_minimization=False)

    history = model.get_history()
    assert isinstance(history, np.ndarray)
    assert history.shape == (1, 0)

    for i in range(3):
        x = model.suggest()
        y = make_y_point(np.array([[i + 1.0]]))
        model.observe(x, y)

    history = model.get_history()
    assert history.shape == (1, 3)
    np.testing.assert_allclose(history, [[1.0, 2.0, 3.0],])

    model_min = OptunaOptimizer(params=params, task_name='testing_task', on_simplex=False, task_minimization=True)
    for i in range(3):
        x = model_min.suggest()
        y = make_y_point(np.array([[i + 1.0]]))
        model_min.observe(x, y)

    history_min = model_min.get_history()
    np.testing.assert_allclose(history_min, [[1.0, 2.0, 3.0],])