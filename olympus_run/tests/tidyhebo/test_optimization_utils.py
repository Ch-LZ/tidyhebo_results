import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.optimizer_adaptors.model_lib import Model, XPoint, YPoint

pytestmark = pytest.mark.env_tidyhebo

# Fixture for temporary dummy-model registration.
@pytest.fixture
def dummy_model_class():
    original_registry = Model._MODEL_REGISTRY.copy()
    original_loaded = Model._LOADED_CLASSES.copy()

    Model._MODEL_REGISTRY['dummy'] = 'tests.dummy.DummyModel'

    class DummyModel(Model):
        _model_name = 'dummy'

        def _observe_impl(self, x0: XPoint, y0: YPoint) -> None:
            pass

        def _suggest_impl(self) -> XPoint:
            return np.array([[0.0]])

        @classmethod
        def _from_config_impl(cls, params, task_name, on_simplex, task_minimization):
            return cls(task_name=task_name, on_simplex=on_simplex, task_minimization=task_minimization)

        def get_history(self):
            return np.array([0.1, 0.2, 0.3]).reshape(1, -1)

    Model._LOADED_CLASSES['dummy'] = DummyModel

    yield DummyModel

    Model._MODEL_REGISTRY = original_registry
    Model._LOADED_CLASSES = original_loaded


def test_find_dataset_config():
    """Check task-config discovery."""
    from src.run_optimization import find_dataset_config

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        config_dir = base / "configs" / "tasks" / "olympus_2021"
        config_dir.mkdir(parents=True)
        # Create a .yaml config file.
        (config_dir / "test_task.yaml").touch()

        found = find_dataset_config("test_task", base_path=base / "configs" / "tasks", subdirs=["olympus_2021"])
        assert found == config_dir / "test_task.yaml"

        with pytest.raises(FileNotFoundError):
            find_dataset_config("unknown", base_path=base / "configs" / "tasks", subdirs=["olympus_2021"])


def test_append_history_to_file(dummy_model_class):
    """Check that optimization history is written to disk."""
    from src.run_optimization import append_history_to_file

    model = dummy_model_class(task_name='testing_task', on_simplex=False, task_minimization=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        append_history_to_file(model, out_dir, "test_task", 0)

        expected_file = out_dir / "test_task" / "dummy_values.txt"
        assert expected_file.exists()
        data = np.loadtxt(expected_file)
        # After the first call the file contains one row with three values.
        np.testing.assert_array_almost_equal(data, [0.1, 0.2, 0.3])

        # The second call should append another row.
        append_history_to_file(model, out_dir, "test_task", 1)
        data2 = np.loadtxt(expected_file)
        # The file now contains two rows of three values.
        assert data2.shape == (2, 3)
        np.testing.assert_array_almost_equal(data2[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(data2[1], [0.1, 0.2, 0.3])
