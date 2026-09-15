import pytest
import yaml
import numpy as np
from pathlib import Path
from unittest.mock import patch
import types

from src.optimizer_adaptors.model_lib import Model, XPoint, YPoint, make_x_point, make_y_point

pytestmark = pytest.mark.env_common

# ---------- Fixtures ----------
@pytest.fixture
def dummy_model_class():
    original_registry = Model._MODEL_REGISTRY.copy()
    original_loaded = Model._LOADED_CLASSES.copy()

    Model._MODEL_REGISTRY['dummy'] = 'tests.dummy.DummyModel'

    class DummyModel(Model):
        _model_name = 'dummy'

        def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
            self._last_x0 = x0
            self._last_y0 = y0

        def _suggest_impl(self) -> XPoint:
            return make_x_point(np.array([[0.5, 0.5]]))

        @classmethod
        def _from_config_impl(cls, params, task_name, on_simplex, task_minimization):
            return cls(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

    Model._LOADED_CLASSES['dummy'] = DummyModel

    yield DummyModel

    Model._MODEL_REGISTRY = original_registry
    Model._LOADED_CLASSES = original_loaded


@pytest.fixture
def temp_config(tmp_path):
    config = {
        'task_name': 'testing_task',
        'params': [
            {'name': 'x1', 'type': 'continuous', 'min': 0, 'max': 1},
            {'name': 'x2', 'type': 'continuous', 'min': 0, 'max': 1}
        ],
        'on_simplex': False,
        'less_is_better': False
    }
    path = tmp_path / 'config.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def temp_config_simplex(tmp_path):
    config = {
        'task_name': 'testing_task',
        'params': [
            {'name': 'p1', 'type': 'continuous'},
            {'name': 'p2', 'type': 'continuous'},
            {'name': 'p3', 'type': 'continuous'}
        ],
        'on_simplex': True,
        'less_is_better': True
    }
    path = tmp_path / 'config_simplex.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


# ---------- make_x_point / make_y_point tests ----------
def test_make_x_point_valid():
    data = np.array([[1.0, 2.0, 3.0]])
    point = make_x_point(data)
    assert point is data
    assert isinstance(point, np.ndarray)
    assert point.shape == (1, 3)


def test_make_x_point_invalid_shape():
    # Shape (2, 2) is valid, so it is not included as an invalid case.
    with pytest.raises(ValueError, match="XPoint must have shape"):
        make_x_point(np.array([1, 2, 3]))          # 1D
    with pytest.raises(ValueError, match="XPoint must have shape"):
        make_x_point(np.array([[[1]]]))            # 3D
    with pytest.raises(ValueError, match="XPoint must have shape"):
        make_x_point(np.empty((0, 3)))             # shape[0] = 0


def test_make_y_point_valid():
    data = np.array([[1], [2], [3]])
    point = make_y_point(data)
    assert point is data
    assert isinstance(point, np.ndarray)
    assert point.shape == (3, 1)


def test_make_y_point_invalid_shape():
    with pytest.raises(ValueError, match="YPoint must have shape"):
        make_y_point(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="YPoint must have shape"):
        make_y_point(np.array([[1, 2]]))


# ---------- Model tests ----------
def test_available_models():
    available = Model.available_models()
    expected = list(Model._MODEL_REGISTRY.keys())
    assert sorted(available) == sorted(expected)


def test_from_config_loads_dummy(dummy_model_class, temp_config):
    model = Model.from_config('dummy', temp_config)
    assert isinstance(model, dummy_model_class)
    assert model.on_simplex is False
    assert model.task_minimization is False
    assert model.to_simplex_transformer is None


def test_from_config_simplex(dummy_model_class, temp_config_simplex):
    model = Model.from_config('dummy', temp_config_simplex)
    assert model.on_simplex is True
    assert model.task_minimization is True
    assert model.to_simplex_transformer is not None
    assert model.to_simplex_transformer.K == 3


def test_from_config_missing_params(dummy_model_class, tmp_path):
    config = {'on_simplex': False}
    path = tmp_path / 'bad.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError, match="Missing 'params' section"):
        Model.from_config('dummy', path)


def test_from_config_unsupported_param_type(dummy_model_class, tmp_path):
    config = {
        'task_name': 'testing_task',
        'params': [
            {'name': 'p1', 'type': 'continuous'},
            {'name': 'p2', 'type': 'categorical', 'values': ['a', 'b']}
        ],
        'on_simplex': True
    }
    path = tmp_path / 'bad.yaml'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    with pytest.raises(NotImplementedError, match="Simplex transformation supports only continuous"):
        Model.from_config('dummy', path)


def test_observe_minimization(dummy_model_class, temp_config):
    model = Model.from_config('dummy', temp_config)  # task_minimization=False
    x = make_x_point(np.array([[1.0, 2.0]]))
    y = make_y_point(np.array([[3.0]]))
    model.observe(x, y)
    assert np.array_equal(model._last_y0, y)

    model_min = dummy_model_class(task_name='testing_task', on_simplex=False, task_minimization=True)
    y_neg = make_y_point(np.array([[-3.0]]))
    model_min.observe(x, y)
    assert np.array_equal(model_min._last_y0, y_neg)


def test_observe_simplex(dummy_model_class, temp_config_simplex):
    model = Model.from_config('dummy', temp_config_simplex)
    x_simplex = np.array([0.2, 0.5, 0.3])  # one-dimensional array
    y = make_y_point(np.array([[1.0]]))
    model.observe(x_simplex, y)
    recovered = model._last_x0
    # Expected values under the softmax/logit mapping.
    expected = np.array([[-0.4, 0.5],])
    np.testing.assert_allclose(recovered, expected)


def test_suggest_no_simplex(dummy_model_class, temp_config):
    model = Model.from_config('dummy', temp_config)
    suggestion = model.suggest()
    assert isinstance(suggestion, np.ndarray)
    # suggestion must be an XPoint with shape (1, 2).
    assert suggestion.shape == (1, 2)
    np.testing.assert_allclose(suggestion, [[0.5, 0.5]])

def test_suggest_simplex(dummy_model_class, temp_config_simplex):
    model = Model.from_config('dummy', temp_config_simplex)
    suggestion = model.suggest()
    assert isinstance(suggestion, np.ndarray)
    assert suggestion.shape == (1, 3)
    # Expected values under softmax(logit(0.625, 0.625)).
    expected = np.array([[0.38461538, 0.38461538, 0.23076923]])
    np.testing.assert_allclose(suggestion, expected, rtol=1e-6, atol=1e-6)


def test_suggest_without_transformer_raises(dummy_model_class):
    model = dummy_model_class(task_name= 'testing_task', on_simplex=True, task_minimization=False)
    model.to_simplex_transformer = None
    with pytest.raises(RuntimeError, match="Simplex transformer not initialized"):
        model.suggest()


def test_observe_minimization_and_simplex(dummy_model_class, temp_config_simplex):
    model = Model.from_config('dummy', temp_config_simplex)
    x_simplex = np.array([0.2, 0.5, 0.3])  # one-dimensional
    y = make_y_point(np.array([[4.0]]))
    model.observe(x_simplex, y)
    np.testing.assert_allclose(model._last_y0, [[-4.0]])
    # Expected values under the softmax/logit mapping.
    expected_x = np.array([[-0.4, 0.5,]])
    np.testing.assert_allclose(model._last_x0, expected_x)


def test_lazy_loading():
    original_registry = Model._MODEL_REGISTRY.copy()
    original_loaded = Model._LOADED_CLASSES.copy()
    Model._MODEL_REGISTRY['mock'] = 'some.module.MockOptimizer'

    class MockOptimizer(Model):
        _model_name = 'mock'

        def _observe_impl(self, x0, y0):
            pass

        def _suggest_impl(self):
            return make_x_point(np.array([[0]]))

        @classmethod
        def _from_config_impl(cls, params, task_name, on_simplex, task_minimization):
            return cls(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

    mock_module = types.ModuleType('some.module')
    mock_module.MockOptimizer = MockOptimizer

    with patch('importlib.import_module') as mock_import:
        mock_import.return_value = mock_module
        Model._LOADED_CLASSES.clear()

        loaded = Model._get_model_class('mock')
        assert loaded is MockOptimizer
        mock_import.assert_called_once_with('some.module')

        mock_import.reset_mock()
        loaded2 = Model._get_model_class('mock')
        assert loaded2 is MockOptimizer
        mock_import.assert_not_called()

    Model._MODEL_REGISTRY = original_registry
    Model._LOADED_CLASSES = original_loaded


def test_get_model_class_invalid_name():
    with pytest.raises(ValueError, match="Unknown model type 'unknown'"):
        Model._get_model_class('unknown')


def test_get_model_class_import_error():
    with patch('importlib.import_module', side_effect=ImportError("No module")):
        original_registry = Model._MODEL_REGISTRY.copy()
        Model._MODEL_REGISTRY['bad'] = 'bad.module.BadClass'
        with pytest.raises(ImportError, match="Could not import model 'bad' from 'bad.module.BadClass'"):
            Model._get_model_class('bad')
        Model._MODEL_REGISTRY = original_registry


def test_subclass_must_define_model_name():
    with pytest.raises(ValueError, match="must define class attribute '_model_name'"):
        class BadModel(Model):
            pass


def test_subclass_with_wrong_model_name():
    with pytest.raises(ValueError, match="Model name 'wrong' not found in _MODEL_REGISTRY"):
        class WrongModel(Model):
            _model_name = 'wrong'

def test_get_history(dummy_model_class, temp_config):
    """Check that get_history returns the accumulated y values."""
    model = Model.from_config('dummy', temp_config)
    x = make_x_point(np.array([[0.1, 0.2]]))
    y1 = make_y_point(np.array([[1.0]]))
    y2 = make_y_point(np.array([[2.5]]))
    model.observe(x, y1)
    model.observe(x, y2)
    history = model.get_history()
    assert isinstance(history, np.ndarray)
    np.testing.assert_allclose(history, [[1.0, 2.5],])


def test_check_init_raises():
    """Check that base-class methods require super().__init__()."""
    class BadModel(Model):
        _model_name = "optuna"
        def __init__(self):
            # Deliberately omit super().__init__().
            pass

        def _observe_impl(self, x0, y0):
            pass

        def _suggest_impl(self):
            return make_x_point(np.array([[0]]))

        @classmethod
        def _from_config_impl(cls, params, task_name, on_simplex, task_minimization):
            return cls()

    model = BadModel()
    with pytest.raises(RuntimeError, match=r"Subclass must call super\(\).__init__"):
        model.suggest()
    with pytest.raises(RuntimeError, match=r"Subclass must call super\(\).__init__"):
        model.observe(np.array([[0]]), np.array([[0]]))
    with pytest.raises(RuntimeError, match=r"Subclass must call super\(\).__init__"):
        model.get_history()

# ---------- Batch-operation tests ----------

@pytest.fixture
def batch_dummy_model_class():
    """Register a temporary class that implements batch methods."""
    original_registry = Model._MODEL_REGISTRY.copy()
    original_loaded = Model._LOADED_CLASSES.copy()

    # Register a temporary model name.
    Model._MODEL_REGISTRY['batch_dummy'] = 'dummy'

    class BatchDummyModel(Model):
        _model_name = 'batch_dummy'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_x_batch = None
            self.last_y_batch = None
            self.suggest_batch_called = False
            self.suggest_batch_size = None
            self.suggest_batch_result = None

        def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
            # Preserve the standard single-point behavior; not used by these tests.
            self._last_x0 = x0
            self._last_y0 = y0

        def _suggest_impl(self) -> XPoint:
            return make_x_point(np.array([[0.5, 0.5]]))

        def _observe_batch_impl(self, x_batch: XPoint, y_batch: YPoint) -> None:
            self.last_x_batch = x_batch
            self.last_y_batch = y_batch

        def _suggest_batch_impl(self, batch_size: int) -> np.ndarray:
            self.suggest_batch_called = True
            self.suggest_batch_size = batch_size
            # Return a (batch_size, 2) array with deterministic values.
            result = np.full((batch_size, 2), 0.7)
            self.suggest_batch_result = result
            return result

        @classmethod
        def _from_config_impl(cls, params, task_name, on_simplex, task_minimization):
            return cls(task_name=task_name, on_simplex=on_simplex,
                       task_minimization=task_minimization)

    Model._LOADED_CLASSES['batch_dummy'] = BatchDummyModel

    yield BatchDummyModel

    # Restore the model registry.
    Model._MODEL_REGISTRY = original_registry
    Model._LOADED_CLASSES = original_loaded


def test_suggest_batch_no_simplex(batch_dummy_model_class, temp_config):
    """Check that batch_size > 1 calls _suggest_batch_impl and returns a valid array."""
    model = Model.from_config('batch_dummy', temp_config)
    model.on_simplex = False  # Explicitly disable simplex mode.

    batch_size = 3
    suggestion = model.suggest(batch_size=batch_size)

    # Check that the batch implementation was called.
    assert model.suggest_batch_called is True
    assert model.suggest_batch_size == batch_size

    # Check shape and values.
    assert isinstance(suggestion, np.ndarray)
    assert suggestion.shape == (batch_size, 2)
    np.testing.assert_allclose(suggestion, np.full((batch_size, 2), 0.7))


def test_suggest_batch_with_simplex(batch_dummy_model_class, temp_config_simplex):
    """Check that simplex transformation is applied to every point in a batch."""
    model = Model.from_config('batch_dummy', temp_config_simplex)
    assert model.on_simplex is True

    batch_size = 2
    suggestion = model.suggest(batch_size=batch_size)

    # _suggest_batch_impl returns a (2, 2) array filled with 0.7.
    # Expect transform to be applied row-wise (K=3, dim=2).
    # The transform uses a softmax-style mapping for each row.
    # For x=[0.7, 0.7], exp(0.7)=2.01375.
    # K=3 uses a two-dimensional unconstrained representation plus the reference component.
    # Exact transformed values are implementation-specific; test invariant properties instead.
    # Earlier single-point tests cover exact reference values.
    # Here we check only that every transformed row sums to one and is positive.
    assert suggestion.shape == (batch_size, 3)
    row_sums = suggestion.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    assert np.all(suggestion > 0)


def test_suggest_batch_not_implemented(dummy_model_class, temp_config):
    """Check that suggest with batch_size > 1 raises NotImplementedError when _suggest_batch_impl is absent."""
    model = Model.from_config('dummy', temp_config)  # dummy_model_class does not override _suggest_batch_impl.
    with pytest.raises(NotImplementedError, match="Batch suggestion not implemented"):
        model.suggest(batch_size=2)


def test_observe_batch_no_simplex(batch_dummy_model_class, temp_config):
    """Check that a point batch is passed to _observe_batch_impl with correct x and y."""
    model = Model.from_config('batch_dummy', temp_config)
    model.on_simplex = False
    model.task_minimization = False

    x_batch = make_x_point(np.array([[0.1, 0.2], [0.3, 0.4]]))
    y_batch = make_y_point(np.array([[5.0], [6.0]]))
    model.observe(x_batch, y_batch)

    # Check that the batch implementation was called.
    assert model.last_x_batch is not None
    assert model.last_y_batch is not None
    np.testing.assert_allclose(model.last_x_batch, x_batch)
    np.testing.assert_allclose(model.last_y_batch, y_batch)
    # Check that history was stored.
    history = model.get_history()
    np.testing.assert_allclose(history, np.array([[5.0, 6.0]]))


def test_observe_batch_with_simplex(batch_dummy_model_class, temp_config_simplex):
    """Check that simplex inputs are inverse-transformed for the whole batch and that the mapping is reversible."""
    model = Model.from_config('batch_dummy', temp_config_simplex)
    model.task_minimization = False

    # Points on a K=3 simplex.
    x_simplex = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])  # (2,3)
    y_batch = make_y_point(np.array([[1.0], [2.0]]))

    x_simplex = make_x_point(x_simplex)  # Valid because shape is (2, 3).
    model.observe(x_simplex, y_batch)

    recovered_x = model.last_x_batch

    # Check that transforming recovered values reconstructs the original simplex points.
    transformed_back = model.to_simplex_transformer.transform(recovered_x)
    np.testing.assert_allclose(transformed_back, x_simplex, rtol=1e-5)

    # Also check the recovered shape.
    assert recovered_x.shape == (2, 2)  # Dimension K-1.


def test_observe_batch_minimization(batch_dummy_model_class, temp_config):
    """Check that task_minimization=True negates y for the entire batch."""
    model = Model.from_config('batch_dummy', temp_config)
    model.on_simplex = False
    model.task_minimization = True

    x_batch = make_x_point(np.array([[0.1, 0.2], [0.3, 0.4]]))
    y_batch = make_y_point(np.array([[5.0], [6.0]]))
    model.observe(x_batch, y_batch)

    # Check that _observe_batch_impl receives negated y.
    np.testing.assert_allclose(model.last_y_batch, np.array([[-5.0], [-6.0]]))
    # History should retain the original positive values.
    history = model.get_history()
    np.testing.assert_allclose(history, np.array([[5.0, 6.0]]))


def test_observe_batch_not_implemented(dummy_model_class, temp_config):
    """Check that batched observe raises NotImplementedError when _observe_batch_impl is absent."""
    model = Model.from_config('dummy', temp_config)
    x_batch = make_x_point(np.array([[0.1, 0.2], [0.3, 0.4]]))
    y_batch = make_y_point(np.array([[1.0], [2.0]]))
    with pytest.raises(NotImplementedError, match="Batch observation not implemented"):
        model.observe(x_batch, y_batch)


def test_observe_single_point_does_not_use_batch(batch_dummy_model_class, temp_config):
    """Check that a one-point array uses _observe_impl rather than the batch implementation."""
    model = Model.from_config('batch_dummy', temp_config)
    model.on_simplex = False
    model.task_minimization = False

    x_single = make_x_point(np.array([[0.1, 0.2]]))
    y_single = make_y_point(np.array([[5.0]]))
    model.observe(x_single, y_single)

    # The batch method must not be called.
    assert model.last_x_batch is None
    assert model.last_y_batch is None
    # The single-point method is used and stores _last_x0/_last_y0.
    assert hasattr(model, '_last_x0')
    np.testing.assert_allclose(model._last_x0, x_single)
    np.testing.assert_allclose(model._last_y0, y_single)

# region adaptive batching
@pytest.fixture
def adaptive_dummy_model_class():
    original_registry = Model._MODEL_REGISTRY.copy()
    original_loaded = Model._LOADED_CLASSES.copy()

    # Register the name before defining the class.
    Model._MODEL_REGISTRY['adaptive_dummy'] = 'dummy'

    class AdaptiveDummy(Model):
        _model_name = 'adaptive_dummy'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.adaptive_called = False
            self.adaptive_ceil_received = None

        def _observe_impl(self, x0, y0):
            pass

        def _suggest_impl(self):
            return make_x_point(np.array([[0.5, 0.5]]))

        def _suggest_adaptive_batch_impl(self, adaptive_ceil: int):
            self.adaptive_called = True
            self.adaptive_ceil_received = adaptive_ceil
            return np.full((adaptive_ceil, 2), 0.7)

        @classmethod
        def _from_config_impl(cls, params, task_name, on_simplex, task_minimization):
            return cls(task_name=task_name, on_simplex=on_simplex,
                       task_minimization=task_minimization)

    # Cache the class so from_config can resolve it.
    Model._LOADED_CLASSES['adaptive_dummy'] = AdaptiveDummy

    yield AdaptiveDummy

    # Restore the original registry state.
    Model._MODEL_REGISTRY = original_registry
    Model._LOADED_CLASSES = original_loaded
# endregion