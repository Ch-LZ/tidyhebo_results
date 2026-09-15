import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src import run_optimization
from optimizer_adaptors.model_lib import Model

pytestmark = pytest.mark.env_o11


# ---------- batch_sizes ----------
def test_batch_sizes():
    total = 10
    init = 3
    batch = 4
    sizes = list(run_optimization.batch_sizes(total, init, batch))
    assert sizes == [3, 4, 3]

    sizes = list(run_optimization.batch_sizes(total=5, n_init=10, n_batch=2))
    assert sizes == [5]

    sizes = list(run_optimization.batch_sizes(total=0, n_init=5, n_batch=3))
    assert sizes == []

    sizes = list(run_optimization.batch_sizes(total=7, n_init=3, n_batch=5))
    assert sizes == [3, 4]


# ---------- find_dataset_config ----------
def test_find_dataset_config(tmp_path):
    base = tmp_path / "configs" / "tasks"
    (base / "olympus 2021").mkdir(parents=True)
    (base / "olympus 2023").mkdir(parents=True)
    file1 = base / "olympus 2021" / "test.yaml"
    file1.touch()
    file2 = base / "olympus 2023" / "test2.yaml"
    file2.touch()

    found = run_optimization.find_dataset_config("test", base_path=base)
    assert found == file1

    found = run_optimization.find_dataset_config("test2", base_path=base)
    assert found == file2

    with pytest.raises(FileNotFoundError):
        run_optimization.find_dataset_config("nonexistent", base_path=base)


# ---------- append_history_to_file ----------
def test_append_history_to_file(tmp_path):
    mock_opt = Mock(spec=Model)
    mock_opt.get_history.return_value = np.array([[1.0, 2.0, 3.0]])
    mock_opt._model_name = "test_opt"

    output_dir = tmp_path / "results"
    task_name = "task1"
    run_idx = 5

    run_optimization.append_history_to_file(mock_opt, output_dir, task_name, run_idx)

    expected_file = output_dir / task_name / "test_opt_values.txt"
    assert expected_file.exists()
    content = np.loadtxt(expected_file)
    np.testing.assert_allclose(content, [1.0, 2.0, 3.0])

    # Append behavior.
    mock_opt.get_history.return_value = np.array([[4.0, 5.0]])
    run_optimization.append_history_to_file(mock_opt, output_dir, task_name, run_idx)
    with open(expected_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 2
    line1 = np.fromstring(lines[0], sep=' ')
    np.testing.assert_allclose(line1, [1.0, 2.0, 3.0])
    line2 = np.fromstring(lines[1], sep=' ')
    np.testing.assert_allclose(line2, [4.0, 5.0])


# ---------- adaptive_ceil tests ----------
# These tests check that adaptive_ceil limits the number of returned points,
# without requiring the main code to shrink batch_size at the final call.

@patch('src.run_optimization.find_dataset_config')
@patch('src.run_optimization.Model')
@patch('src.run_optimization.append_history_to_file')
def test_main_basic(mock_append, mock_model_class, mock_find_config, tmp_path):
    """Basic test without adaptive_ceil; only the number of calls is checked."""
    mock_optimizer = Mock()
    mock_model_class.from_config.return_value = mock_optimizer

    def suggest_side_effect(batch_size=None, adaptive_ceil=None):
        if adaptive_ceil is not None:
            n = adaptive_ceil
        else:
            n = batch_size if batch_size is not None else 1
        return np.ones((n, 2))
    mock_optimizer.suggest.side_effect = suggest_side_effect

    dummy_config = tmp_path / "dummy.yaml"
    dummy_config.touch()
    mock_find_config.return_value = dummy_config

    from_m = tmp_path / "fromM"
    from_b = tmp_path / "fromB"
    inp = tmp_path / "in"
    out = tmp_path / "out"
    from_b.touch()
    np.savetxt(out, [[42.0]])

    num_iters = 5
    init_size = 2
    batch_size = 2
    dataset = "test"
    model_type = "dummy"
    run_idx = 7
    output_dir = "some_dir"

    run_optimization.main(
        from_M_to_B=str(from_m),
        from_B_to_M=str(from_b),
        dataset=dataset,
        inp_file=str(inp),
        output_file=str(out),
        num_iterations=num_iters,
        model_type=model_type,
        run_idx=run_idx,
        init_size=init_size,
        batch_size=batch_size,
        output_dir=output_dir,
        seed=None
    )

    mock_find_config.assert_called_once_with(dataset)
    mock_model_class.from_config.assert_called_once_with(model_type, dummy_config)

    # Expect three calls; the implementation does not shrink the final fixed batch.
    # The final call still uses batch_size=2, producing six proposed points in total.
    # Only the number of calls is checked here.
    expected_calls = len(list(run_optimization.batch_sizes(num_iters, init_size, batch_size)))
    assert mock_optimizer.suggest.call_count == expected_calls

    # adaptive_ceil should remain None.
    for call in mock_optimizer.suggest.call_args_list:
        kwargs = call[1]
        assert kwargs.get('adaptive_ceil') is None

    assert mock_optimizer.observe.call_count == expected_calls
    mock_append.assert_called_once()
    args, _ = mock_append.call_args
    assert args[0] is mock_optimizer
    assert str(args[1]).endswith(output_dir)
    assert args[2] == dataset
    assert args[3] == run_idx
    assert inp.exists()


@patch('src.run_optimization.find_dataset_config')
@patch('src.run_optimization.Model')
@patch('src.run_optimization.append_history_to_file')
def test_main_seed(mock_append, mock_model, mock_find, tmp_path):
    with patch('numpy.random.seed') as mock_seed:
        dummy_config = tmp_path / "dummy.yaml"
        dummy_config.touch()
        mock_find.return_value = dummy_config
        mock_opt = Mock()
        mock_opt.suggest.return_value = np.ones((1, 2))
        mock_model.from_config.return_value = mock_opt

        from_m = tmp_path / "f1"
        from_b = tmp_path / "f2"
        inp = tmp_path / "in"
        out = tmp_path / "out"
        from_b.touch()
        np.savetxt(out, [[0.0]])

        run_optimization.main(
            from_M_to_B=str(from_m),
            from_B_to_M=str(from_b),
            dataset="test",
            inp_file=str(inp),
            output_file=str(out),
            num_iterations=1,
            model_type="dummy",
            run_idx=0,
            init_size=1,
            batch_size=1,
            output_dir=".",
            seed=123
        )
        mock_seed.assert_called_once_with(123)


def test_main_batch_size_one(tmp_path):
    with patch('src.run_optimization.find_dataset_config') as mock_find, \
         patch('src.run_optimization.Model') as mock_model, \
         patch('src.run_optimization.append_history_to_file'):
        dummy_config = tmp_path / "dummy.yaml"
        dummy_config.touch()
        mock_find.return_value = dummy_config

        mock_opt = Mock()
        mock_opt.suggest.return_value = np.ones((1, 2))
        mock_model.from_config.return_value = mock_opt

        from_m = tmp_path / "f1"
        from_b = tmp_path / "f2"
        inp = tmp_path / "in"
        out = tmp_path / "out"
        from_b.touch()
        np.savetxt(out, [[0.0]])

        run_optimization.main(
            from_M_to_B=str(from_m),
            from_B_to_M=str(from_b),
            dataset="test",
            inp_file=str(inp),
            output_file=str(out),
            num_iterations=3,
            model_type="dummy",
            run_idx=0,
            init_size=1,
            batch_size=1,
            output_dir=".",
            seed=None
        )

        assert mock_opt.suggest.call_count == 3
        for call in mock_opt.suggest.call_args_list:
            # Both arguments should be forwarded.
            assert call[1] == {'batch_size': 1, 'adaptive_ceil': None}

        assert mock_opt.observe.call_count == 3


# ---------- adaptive_ceil tests ----------

@patch('src.run_optimization.find_dataset_config')
@patch('src.run_optimization.Model')
@patch('src.run_optimization.append_history_to_file')
def test_main_adaptive_ceil_less_than_remaining(mock_append, mock_model_class, mock_find_config, tmp_path):
    """Check that adaptive_ceil limits the number of points per call."""
    dummy_config = tmp_path / "dummy.yaml"
    dummy_config.touch()
    mock_find_config.return_value = dummy_config

    mock_opt = Mock()
    def suggest_side_effect(batch_size=None, adaptive_ceil=None):
        if adaptive_ceil is not None:
            n = adaptive_ceil
        else:
            n = batch_size if batch_size is not None else 1
        return np.ones((n, 2))
    mock_opt.suggest.side_effect = suggest_side_effect
    mock_model_class.from_config.return_value = mock_opt

    from_m = tmp_path / "f1"
    from_b = tmp_path / "f2"
    inp = tmp_path / "in"
    out = tmp_path / "out"
    from_b.touch()
    np.savetxt(out, [[42.0]])

    num_iters = 10
    batch_size = 3
    adaptive_ceil = 4

    run_optimization.main(
        from_M_to_B=str(from_m),
        from_B_to_M=str(from_b),
        dataset="test",
        inp_file=str(inp),
        output_file=str(out),
        num_iterations=num_iters,
        model_type="dummy",
        run_idx=0,
        init_size=1,
        batch_size=batch_size,
        adaptive_ceil=adaptive_ceil,
        output_dir=".",
        seed=None
    )

    calls = mock_opt.suggest.call_args_list
    # Expect three calls with caps 4, 4, and 2.
    assert len(calls) == 3
    expected_ceils = [4, 4, 2]
    for call, expected in zip(calls, expected_ceils):
        kwargs = call[1]
        assert kwargs.get('adaptive_ceil') == expected
        assert kwargs.get('batch_size') == batch_size  # forwarded but ignored
    assert mock_opt.observe.call_count == 3


@patch('src.run_optimization.find_dataset_config')
@patch('src.run_optimization.Model')
@patch('src.run_optimization.append_history_to_file')
def test_main_adaptive_ceil_greater_than_remaining(mock_append, mock_model_class, mock_find_config, tmp_path):
    """Check that adaptive_ceil does not exceed the remaining evaluation budget."""
    dummy_config = tmp_path / "dummy.yaml"
    dummy_config.touch()
    mock_find_config.return_value = dummy_config

    mock_opt = Mock()
    def suggest_side_effect(batch_size=None, adaptive_ceil=None):
        if adaptive_ceil is not None:
            n = adaptive_ceil
        else:
            n = batch_size if batch_size is not None else 1
        return np.ones((n, 2))
    mock_opt.suggest.side_effect = suggest_side_effect
    mock_model_class.from_config.return_value = mock_opt

    from_m = tmp_path / "f1"
    from_b = tmp_path / "f2"
    inp = tmp_path / "in"
    out = tmp_path / "out"
    from_b.touch()
    np.savetxt(out, [[42.0]])

    num_iters = 5
    batch_size = 3
    adaptive_ceil = 10

    run_optimization.main(
        from_M_to_B=str(from_m),
        from_B_to_M=str(from_b),
        dataset="test",
        inp_file=str(inp),
        output_file=str(out),
        num_iterations=num_iters,
        model_type="dummy",
        run_idx=0,
        init_size=1,
        batch_size=batch_size,
        adaptive_ceil=adaptive_ceil,
        output_dir=".",
        seed=None
    )

    calls = mock_opt.suggest.call_args_list
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs.get('adaptive_ceil') == 5  # limited by num_iters
    assert mock_opt.observe.call_count == 1


@patch('src.run_optimization.find_dataset_config')
@patch('src.run_optimization.Model')
@patch('src.run_optimization.append_history_to_file')
def test_main_adaptive_ceil_none_uses_batch_size(mock_append, mock_model_class, mock_find_config, tmp_path):
    """Check that adaptive_ceil=None uses the fixed batch_size."""
    dummy_config = tmp_path / "dummy.yaml"
    dummy_config.touch()
    mock_find_config.return_value = dummy_config

    mock_opt = Mock()
    def suggest_side_effect(batch_size=None, adaptive_ceil=None):
        if adaptive_ceil is not None:
            n = adaptive_ceil
        else:
            n = batch_size if batch_size is not None else 1
        return np.ones((n, 2))
    mock_opt.suggest.side_effect = suggest_side_effect
    mock_model_class.from_config.return_value = mock_opt

    from_m = tmp_path / "f1"
    from_b = tmp_path / "f2"
    inp = tmp_path / "in"
    out = tmp_path / "out"
    from_b.touch()
    np.savetxt(out, [[42.0]])

    num_iters = 10
    batch_size = 3
    adaptive_ceil = None

    run_optimization.main(
        from_M_to_B=str(from_m),
        from_B_to_M=str(from_b),
        dataset="test",
        inp_file=str(inp),
        output_file=str(out),
        num_iterations=num_iters,
        model_type="dummy",
        run_idx=0,
        init_size=1,
        batch_size=batch_size,
        adaptive_ceil=adaptive_ceil,
        output_dir=".",
        seed=None
    )

    calls = mock_opt.suggest.call_args_list
    # Expect four calls; fixed batch_size remains 3 even on the final call.
    assert len(calls) == 4
    for call in calls:
        kwargs = call[1]
        assert kwargs.get('adaptive_ceil') is None
        # batch_size remains 3; it is not reduced at the end.
        assert kwargs.get('batch_size') == batch_size
    assert mock_opt.observe.call_count == 4