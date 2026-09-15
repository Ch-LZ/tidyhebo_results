import pytest
import numpy as np
import yaml
from pathlib import Path

from src.optimizer_adaptors.random_opt import RandomOptimizer
from src.optimizer_adaptors.model_lib import Model, make_x_point, make_y_point

pytestmark = pytest.mark.env_tidyhebo


# ---------- Fixtures ----------
@pytest.fixture
def params_continuous():
    return [
        {'name': 'x1', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'x2', 'type': 'continuous', 'low': -2.0, 'high': 2.0},
    ]


@pytest.fixture
def optimizer(params_continuous):
    return RandomOptimizer(params=params_continuous, task_name='testing_task', on_simplex=False, task_minimization=False)


@pytest.fixture
def config_yaml(tmp_path, params_continuous):
    config = {
        'task_name': 'testing_task',
        'params': params_continuous,
        'on_simplex': False,
        'less_is_better': False
    }
    path = tmp_path / 'config.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


# ---------- Tests ----------
def test_registration():
    """Check that RandomOptimizer is registered."""
    assert 'random' in Model._MODEL_REGISTRY
    # The registry stores a class path string, so check the name.
    assert Model._MODEL_REGISTRY['random'] == 'src.optimizer_adaptors.random_opt.RandomOptimizer'


def test_from_config(config_yaml):
    """Create an instance through from_config."""
    model = RandomOptimizer.from_config('random', config_yaml)
    assert isinstance(model, RandomOptimizer)
    assert model.on_simplex is False
    assert model.task_minimization is False
    assert model.dim == 2
    assert model.bounds == [(0.0, 1.0), (-2.0, 2.0)]


def test_from_config_with_minimization(tmp_path, params_continuous):
    """Check that the minimization flag is propagated correctly."""
    config = {
        'task_name': 'testing_task',
        'params': params_continuous,
        'less_is_better': True
    }
    path = tmp_path / 'config_min.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    model = RandomOptimizer.from_config('random', path)
    assert model.task_minimization is True


def test_suggest_shape(optimizer):
    """suggest returns an XPoint with shape (1, dim)."""
    point = optimizer.suggest()
    assert isinstance(point, np.ndarray)
    assert point.shape == (1, optimizer.dim)


def test_suggest_values_within_bounds(optimizer):
    """Suggested points lie within the configured bounds."""
    for _ in range(50):
        point = optimizer.suggest()
        for i, (low, high) in enumerate(optimizer.bounds):
            assert low <= point[0, i] <= high


def test_suggest_randomness(optimizer):
    """Check that independently sampled points differ with high probability."""
    points = [optimizer.suggest().ravel() for _ in range(20)]
    unique = np.unique(np.vstack(points), axis=0)
    # Independently sampled points should not all be identical.
    assert len(unique) > 1, "All generated points are identical"


def test_observe_does_not_affect_suggest(optimizer):
    """observe does not change the random sampling mechanism."""
    # Fix the seed for reproducibility.
    np.random.seed(42)
    points_before = [optimizer.suggest().ravel() for _ in range(5)]
    np.random.seed(42)  # reset
    optimizer.observe(np.array([[0.0, 0.0]]), np.array([[0.5]]))
    points_after = [optimizer.suggest().ravel() for _ in range(5)]
    np.testing.assert_array_equal(points_before, points_after)

def test_get_history_empty(optimizer):
    """get_history returns an empty array when there are no observations."""
    history = optimizer.get_history()
    assert isinstance(history, np.ndarray)
    assert history.shape == (1, 0)


def test_get_history_after_observations(optimizer):
    """get_history returns observed y values."""
    y_values = [0.3, 0.7, 1.2]
    for y in y_values:
        x = optimizer.suggest()
        optimizer.observe(x, make_y_point(np.array([[y]])))
    history = optimizer.get_history()
    np.testing.assert_allclose(history, [y_values,])


def test_get_history_with_minimization(params_continuous):
    """With task_minimization=True, history values should be sign-inverted."""
    opt = RandomOptimizer(params=params_continuous, task_name='testing_task', on_simplex=False, task_minimization=True)
    x = opt.suggest()
    y_orig = 2.5
    opt.observe(x, make_y_point(np.array([[y_orig]])))
    # History should contain -2.5 because observe flips the sign.
    history = opt.get_history()
    assert history[0][0] == 2.5


def test_simplex_mode(params_continuous):
    """Check operation in simplex mode."""
    # Create a temporary config with on_simplex=True.
    config = {
        'task_name': 'testing_task',
        'params': params_continuous,
        'on_simplex': True,
        'less_is_better': False
    }
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        cfg_path = Path(f.name)

    try:
        model = RandomOptimizer.from_config('random', cfg_path)
        assert model.on_simplex is True
        assert model.to_simplex_transformer is not None
        assert model.to_simplex_transformer.K == len(params_continuous)  # K=2

        # suggest should return a simplex point whose coordinates sum to one.
        point = model.suggest()
        assert point.shape == (1, 2)
        assert np.isclose(np.sum(point), 1.0)

        # Observation should succeed.
        x_simplex = np.array([0.2, 0.8])  # sum=1
        y = make_y_point(np.array([[0.5]]))
        model.observe(x_simplex, y)  # should succeed

    finally:
        # Remove the temporary file.
        cfg_path.unlink()


def test_init_invalid_param_type():
    """Check that non-continuous parameters raise an error."""
    invalid_params = [
        {'name': 'red', 'type': 'integer', 'low': 0, 'high': 10},
        {'name': 'green', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
    ]
    with pytest.raises(ValueError, match="only 'continuous'"):
        RandomOptimizer(params=invalid_params, task_name='testing_task')


def test_init_missing_type_adds_default():
    """If type is omitted, it is added as 'continuous'."""
    params = [{'name': 'x', 'low': 0, 'high': 1}]
    opt = RandomOptimizer(params=params, task_name='testing_task')
    # Check that type was added.
    assert opt.params[0]['type'] == 'continuous'