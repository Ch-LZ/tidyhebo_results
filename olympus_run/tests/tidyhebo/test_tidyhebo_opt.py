# tests/tidyhebo/test_tidyhebo_opt.py

from pathlib import Path

import pytest
import numpy as np
import yaml
import tempfile

from src.optimizer_adaptors.model_lib import Model, make_y_point, make_x_point

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="tidyhebo")

pytestmark = pytest.mark.env_tidyhebo


@pytest.fixture(scope="function")
def tidyhebo_optimizer():
    pytest.importorskip("tidyhebo")
    from src.optimizer_adaptors.tidyhebo_opt import TidyheboOptimizer
    params = [
        {'name': 'red',    'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'orange', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'yellow', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'blue',   'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'green',  'type': 'continuous', 'low': 0.0, 'high': 1.0},
    ]
    return TidyheboOptimizer(
        params=params,
        task_name='testing_task',
        task_minimization=True,
        on_simplex=False,
        n_init=4,
        return_df=True
    )


def test_registration():
    pytest.importorskip("tidyhebo")
    from src.optimizer_adaptors.tidyhebo_opt import TidyheboOptimizer
    assert 'tidyhebo' in Model._MODEL_REGISTRY
    assert Model._MODEL_REGISTRY['tidyhebo'] == 'src.optimizer_adaptors.tidyhebo_opt.TidyheboOptimizer'


def test_from_config():
    pytest.importorskip("tidyhebo")
    from src.optimizer_adaptors.tidyhebo_opt import TidyheboOptimizer
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

    optimizer = Model.from_config('tidyhebo', cfg_path)
    assert isinstance(optimizer, TidyheboOptimizer)
    assert optimizer.task_minimization is True
    assert optimizer.n_init == 4
    Path(cfg_path).unlink()


def test_suggest_shape(tidyhebo_optimizer):
    optimizer = tidyhebo_optimizer
    point = optimizer.suggest()
    assert point.shape == (1, len(optimizer.params))


def test_suggest_values_within_bounds(tidyhebo_optimizer):
    optimizer = tidyhebo_optimizer
    for _ in range(optimizer.n_init):
        point = optimizer.suggest()
        for i, p in enumerate(optimizer.params):
            assert p['low'] <= point[0, i] <= p['high']
        optimizer.observe(point, make_y_point(np.array([[0.5]])))
    point = optimizer.suggest()
    for i, p in enumerate(optimizer.params):
        assert p['low'] <= point[0, i] <= p['high']


def test_observe_and_suggest(tidyhebo_optimizer):
    optimizer = tidyhebo_optimizer
    x = optimizer.suggest()
    y = make_y_point(np.array([[0.5]]))
    optimizer.observe(x, y)
    x2 = optimizer.suggest()
    assert x2.shape == (1, len(optimizer.params))


def test_observe_with_1d_y(tidyhebo_optimizer):
    optimizer = tidyhebo_optimizer
    x = optimizer.suggest()
    y = make_y_point(np.array([[0.7]]))
    optimizer.observe(x, y)
    x2 = optimizer.suggest()
    assert x2.shape == (1, len(optimizer.params))


def test_observe_with_dataframe_conversion(tidyhebo_optimizer):
    optimizer = tidyhebo_optimizer
    x = optimizer.suggest()
    y = make_y_point(np.array([[0.5]]))
    original_observe = optimizer.optimizer.observe
    captured_df = None

    def mock_observe(df, y_arr):
        nonlocal captured_df
        captured_df = df
        original_observe(df, y_arr)

    optimizer.optimizer.observe = mock_observe
    optimizer.observe(x, y)
    assert captured_df is not None
    assert list(captured_df.columns) == [p['name'] for p in optimizer.params]
    np.testing.assert_array_equal(captured_df.values, x)


def test_minimization_sign():
    pytest.importorskip("tidyhebo")
    from src.optimizer_adaptors.tidyhebo_opt import TidyheboOptimizer
    params = [{'name': 'red', 'type': 'continuous', 'low': 0.0, 'high': 1.0}]

    # For maximization, the base class does not invert y.
    opt_max = TidyheboOptimizer(
        params=params,
        task_name='testing_task',
        task_minimization=False,
        on_simplex=False,
        n_init=1,
        return_df=True
    )
    x = opt_max.suggest()
    y = make_y_point(np.array([[0.7]]))

    original_observe = opt_max.optimizer.observe
    captured_y = None

    def mock_observe(df, y_arr):
        nonlocal captured_y
        captured_y = y_arr[0, 0]
        original_observe(df, y_arr)

    opt_max.optimizer.observe = mock_observe
    opt_max.observe(x, y)
    # Pass y unchanged for maximization.
    assert captured_y == 0.7


def test_invalid_param_type():
    from src.optimizer_adaptors.tidyhebo_opt import TidyheboOptimizer
    invalid_params = [
        {'name': 'red', 'type': 'integer', 'low': 0, 'high': 10},
        {'name': 'green', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
    ]
    with pytest.raises(ValueError, match="Only 'continuous' parameters are supported"):
        TidyheboOptimizer._from_config_impl(invalid_params, task_name='testing_task', on_simplex=False, task_minimization=True)


def test_get_history(tidyhebo_optimizer):
    """Check that the inherited get_history method works correctly."""
    optimizer = tidyhebo_optimizer
    # History starts empty.
    history = optimizer.get_history()
    assert history.shape == (1, 0)

    # Add observations.
    x1 = optimizer.suggest()
    y1 = make_y_point(np.array([[0.5]]))
    optimizer.observe(x1, y1)

    x2 = optimizer.suggest()
    y2 = make_y_point(np.array([[0.7]]))
    optimizer.observe(x2, y2)

    history = optimizer.get_history()
    np.testing.assert_allclose(history, [[0.5, 0.7],])

# region adaptive batching


def test_suggest_adaptive_batch_returns_numpy(tidyhebo_optimizer):
    """Check that adaptive batching returns a NumPy array."""
    optimizer = tidyhebo_optimizer
    # Add observations so a Pareto front is available.
    for _ in range(5):
        x = optimizer.suggest()
        y = make_y_point(np.array([[np.random.rand()]]))
        optimizer.observe(x, y)

    points = optimizer.suggest(adaptive_ceil=3)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[0] <= 3
    assert points.shape[1] == len(optimizer.params)


def test_suggest_adaptive_batch_limits_points(tidyhebo_optimizer):
    """Check that no more than adaptive_ceil points are returned."""
    optimizer = tidyhebo_optimizer
    # Add enough observations to obtain a larger front.
    for _ in range(20):
        x = optimizer.suggest()
        y = make_y_point(np.array([[np.random.rand()]]))
        optimizer.observe(x, y)

    ceil = 5
    points = optimizer.suggest(adaptive_ceil=ceil)
    assert len(points) <= ceil


def test_suggest_adaptive_batch_uses_pareto_front(tidyhebo_optimizer):
    """Check that points are taken from the Pareto front via return_all=True."""
    optimizer = tidyhebo_optimizer
    # Replace the inner suggest method to inspect how it is called.
    original_suggest = optimizer.optimizer.suggest
    call_args = []

    def mock_suggest(return_all=False):
        # Return an empty DataFrame for the simple path.
        if return_all:
            call_args.append(('return_all', return_all))
            # Create a DataFrame with five points.
            import pandas as pd
            data = np.random.rand(5, len(optimizer.params))
            return pd.DataFrame(data, columns=[p['name'] for p in optimizer.params])
        else:
            # Ordinary suggest returns one point.
            return pd.DataFrame([np.random.rand(len(optimizer.params))],
                                columns=[p['name'] for p in optimizer.params])

    optimizer.optimizer.suggest = mock_suggest

    # Request an adaptive batch.
    optimizer.suggest(adaptive_ceil=3)
    # Check that return_all=True was used.
    assert ('return_all', True) in call_args

    # Restore the original method.
    optimizer.optimizer.suggest = original_suggest


def test_suggest_adaptive_batch_with_simplex(tidyhebo_optimizer):
    """Check that adaptive batches are transformed correctly when on_simplex=True."""
    # Create an optimizer in simplex mode.
    params = [
        {'name': 'p1', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'p2', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
        {'name': 'p3', 'type': 'continuous', 'low': 0.0, 'high': 1.0},
    ]
    # Use from_config so SimplexTransformer is initialized correctly.
    config = {
        'task_name': 'simplex_test',
        'params': params,
        'on_simplex': True,
        'less_is_better': True,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        cfg_path = f.name

    optimizer = Model.from_config('tidyhebo', cfg_path)
    assert optimizer.on_simplex is True

    # Add observations on the simplex.
    for _ in range(5):
        x = np.random.dirichlet(np.ones(3)).reshape(1, -1)
        y = make_y_point(np.array([[np.random.rand()]]))
        optimizer.observe(x, y)

    points = optimizer.suggest(adaptive_ceil=2)
    # Transformed points must lie on the simplex and sum to one.
    assert points.shape[1] == 3
    sums = points.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    Path(cfg_path).unlink()


def test_observe_batch_tidyhebo(tidyhebo_optimizer):
    """Check that batched observations are passed through correctly."""
    optimizer = tidyhebo_optimizer
    # Generate a batch of three points.
    x_batch = np.random.rand(3, len(optimizer.params))
    y_batch = np.random.rand(3, 1)  # shape (3,1)

    # Replace optimizer.observe to inspect the DataFrame passed to it.
    original_observe = optimizer.optimizer.observe
    captured_df = None

    def mock_observe(df, y_arr):
        nonlocal captured_df
        captured_df = df
        original_observe(df, y_arr)

    optimizer.optimizer.observe = mock_observe

    # Call batched observe.
    optimizer.observe(make_x_point(x_batch), make_y_point(y_batch))

    assert captured_df is not None
    assert captured_df.shape == (3, len(optimizer.params))
    np.testing.assert_array_equal(captured_df.values, x_batch)

    # Restore the original method.
    optimizer.optimizer.observe = original_observe
# end region