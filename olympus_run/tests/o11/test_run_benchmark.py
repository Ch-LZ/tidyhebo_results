import pytest
import numpy as np
from pathlib import Path
from src import run_benchmark
from src.tasks import OLYMPUS_TASKS

pytestmark = pytest.mark.env_o11


# ---------- Emulator test double ----------
class CapturingEmulator:
    instances = []
    call_count = 0

    def __init__(self, dataset, strategy):
        self.dataset = dataset
        self.strategy = strategy
        CapturingEmulator.instances.append((dataset, strategy))

    def run(self, x):
        CapturingEmulator.call_count += 1
        # x should be a 2D array with shape (N, D).
        # Convert accidental 1D inputs to (N, 1); tests normally provide 2D inputs.
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return np.zeros((len(x), 1)), None, None


# ---------- batch_sizes tests ----------
def test_batch_sizes():
    total = 10
    init = 3
    batch = 4
    sizes = list(run_benchmark.batch_sizes(total, init, batch))
    assert sizes == [3, 4, 3]

    sizes = list(run_benchmark.batch_sizes(total=5, n_init=10, n_batch=2))
    assert sizes == [5]

    sizes = list(run_benchmark.batch_sizes(total=0, n_init=5, n_batch=3))
    assert sizes == []

    sizes = list(run_benchmark.batch_sizes(total=7, n_init=3, n_batch=5))
    assert sizes == [3, 4]


# ---------- main tests ----------
def test_main_uses_correct_emulator_params(tmp_path):
    """Check that Emulator receives the requested dataset and strategy."""
    run_benchmark.Emulator = CapturingEmulator
    CapturingEmulator.instances.clear()
    CapturingEmulator.call_count = 0

    signal1 = tmp_path / "p1"
    signal2 = tmp_path / "p2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0.5 0.5\n")
    signal2.touch()
    # Two-dimensional point.
    np.savetxt(inp, [[0.0, 0.0]])

    run_benchmark.main(
        from_M_to_B=str(signal1),
        from_B_to_M=str(signal2),
        dataset=OLYMPUS_TASKS[0],
        inp_file=str(inp),
        init_size=1,
        output_file=str(out),
        num_iterations=5,
        seed=None
    )

    assert len(CapturingEmulator.instances) == 1
    dataset, strategy = CapturingEmulator.instances[0]
    assert dataset == OLYMPUS_TASKS[0]
    assert strategy == "BayesNeuralNet"


def test_main_handles_missing_signal_file(tmp_path):
    """Check that reading the signal file does not fail."""
    run_benchmark.Emulator = CapturingEmulator
    CapturingEmulator.call_count = 0

    signal1 = tmp_path / "pipe1"
    signal2 = tmp_path / "pipe2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0.1 0.2\n")
    signal2.touch()
    np.savetxt(inp, [[0.0, 0.0]])

    run_benchmark.main(
        from_M_to_B=str(signal1),
        from_B_to_M=str(signal2),
        dataset="alkox",
        inp_file=str(inp),
        init_size=1,
        output_file=str(out),
        num_iterations=3,
        seed=None
    )

    assert out.exists()


def test_main_iteration_count_matches_file_reads(tmp_path):
    """
    Check that each iteration reads the file and decrements iter_remained by the number of rows.
    """
    run_benchmark.Emulator = CapturingEmulator
    CapturingEmulator.call_count = 0

    signal1 = tmp_path / "p1"
    signal2 = tmp_path / "p2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0 0\n")
    signal2.touch()
    np.savetxt(inp, [[0.0, 0.0]])

    run_benchmark.main(
        from_M_to_B=str(signal1),
        from_B_to_M=str(signal2),
        dataset="alkox",
        inp_file=str(inp),
        init_size=1,
        output_file=str(out),
        num_iterations=3,
        seed=None
    )

    assert CapturingEmulator.call_count == 3


def test_main_handles_multiple_points_in_file(tmp_path):
    """
    Check that multiple points are processed in one call and decrement iter_remained accordingly.
    """
    run_benchmark.Emulator = CapturingEmulator
    CapturingEmulator.call_count = 0

    signal1 = tmp_path / "p1"
    signal2 = tmp_path / "p2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0 0\n")
    signal2.touch()
    # Five points with two coordinates each.
    points = np.random.rand(5, 2)
    np.savetxt(inp, points)

    num_iters = 10  # More iterations than points in one input batch.
    run_benchmark.main(
        from_M_to_B=str(signal1),
        from_B_to_M=str(signal2),
        dataset="alkox",
        inp_file=str(inp),
        init_size=1,
        output_file=str(out),
        num_iterations=num_iters,
        seed=None
    )

    # Expect two calls: each consumes five evaluations.
    assert CapturingEmulator.call_count == 2


def test_main_seed_setting(tmp_path):
    """Check that seed is passed to np.random.seed."""
    run_benchmark.Emulator = CapturingEmulator
    CapturingEmulator.call_count = 0

    signal1 = tmp_path / "p1"
    signal2 = tmp_path / "p2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0.5 0.5\n")
    signal2.touch()
    np.savetxt(inp, [[0.0, 0.0]])

    test_seed = 12345
    original_seed = np.random.seed
    seed_called_with = [None]

    def mock_seed(s):
        seed_called_with[0] = s
        original_seed(s)

    np.random.seed = mock_seed

    try:
        run_benchmark.main(
            from_M_to_B=str(signal1),
            from_B_to_M=str(signal2),
            dataset="alkox",
            inp_file=str(inp),
            init_size=1,
            output_file=str(out),
            num_iterations=1,
            seed=test_seed
        )
    finally:
        np.random.seed = original_seed

    assert seed_called_with[0] == test_seed


def test_main_output_file_overwritten_each_iteration(tmp_path):
    """Check that output_file is overwritten on every iteration."""
    class VariedEmulator:
        call_idx = 0
        def __init__(self, dataset, strategy):
            pass
        def run(self, x):
            VariedEmulator.call_idx += 1
            # x must be 2D.
            return np.array([[float(VariedEmulator.call_idx)] for _ in range(len(x))]), None, None

    run_benchmark.Emulator = VariedEmulator

    signal1 = tmp_path / "p1"
    signal2 = tmp_path / "p2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0.5 0.5\n")
    signal2.touch()
    np.savetxt(inp, [[0.0, 0.0]])

    run_benchmark.main(
        from_M_to_B=str(signal1),
        from_B_to_M=str(signal2),
        dataset="alkox",
        inp_file=str(inp),
        init_size=1,
        output_file=str(out),
        num_iterations=3,
        seed=None
    )

    data = np.loadtxt(out)
    # data should be a scalar or a one-element array.
    assert float(data) == 3.0
    assert out.exists()


def test_main_with_variable_points_adaptive_ceil(tmp_path):
    """
    Check a variable-size input batch, mimicking adaptive_ceil, and complete the loop correctly.
    """
    run_benchmark.Emulator = CapturingEmulator
    CapturingEmulator.call_count = 0

    signal1 = tmp_path / "p1"
    signal2 = tmp_path / "p2"
    inp = tmp_path / "in"
    out = tmp_path / "out"

    signal1.write_text("0 0\n")
    signal2.touch()
    # Three points with two coordinates each.
    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.savetxt(inp, points)

    num_iters = 5  # More iterations than points in one input batch.
    run_benchmark.main(
        from_M_to_B=str(signal1),
        from_B_to_M=str(signal2),
        dataset="alkox",
        inp_file=str(inp),
        init_size=1,
        output_file=str(out),
        num_iterations=num_iters,
        seed=None
    )

    # First call consumes three evaluations; the second finishes the budget.
    assert CapturingEmulator.call_count == 2
    assert out.exists()