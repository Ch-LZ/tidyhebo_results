import pytest
import numpy as np
from src.optimizer_adaptors.simplex_transformer import SimplexTransformer  # Import the transformer under test.

pytestmark = pytest.mark.env_common

# ------------------ Fixtures ------------------
@pytest.fixture
def trans3():
    """Transformer with K=3, L=-1, U=2."""
    return SimplexTransformer(K=3, L=-1.0, U=2.0)


@pytest.fixture
def trans2():
    """Transformer with K=2, L=0, U=1."""
    return SimplexTransformer(K=2, L=0.0, U=1.0)


@pytest.fixture
def trans4():
    """Transformer with K=4, L=-2, U=2."""
    return SimplexTransformer(K=4, L=-2.0, U=2.0)


# ------------------ Initialization tests ------------------
def test_initialization_errors():
    with pytest.raises(ValueError, match="K must be >= 2"):
        SimplexTransformer(K=1)
    with pytest.raises(ValueError, match="L must be smaller than U"):
        SimplexTransformer(K=3, L=2.0, U=1.0)


# ------------------ transform tests ------------------
def test_transform_shape_error(trans3):
    with pytest.raises(ValueError, match="Expected an array of length"):
        trans3.transform(np.array([0.0]))                 # length 1
    with pytest.raises(ValueError, match="Expected an array of length"):
        trans3.transform(np.array([0.0, 1.0, 2.0]))       # length 3


def test_transform_output_properties(trans3):
    """Check that transformed values lie on the simplex (sum=1, all >=0)."""
    for _ in range(20):
        y = np.random.uniform(trans3.L, trans3.U, size=trans3.dim)
        x = trans3.transform(y)
        assert np.sum(x) == pytest.approx(1.0, abs=1e-10)
        assert np.all(x >= 0.0) and np.all(x <= 1.0)


# ------------------ inverse_transform tests ------------------
def test_inverse_transform_recovery(trans3):
    """transform(y) -> inverse_transform -> y for interior points."""
    for _ in range(50):
        y = np.random.uniform(trans3.L, trans3.U, size=trans3.dim)
        x = trans3.transform(y)
        y_rec = trans3.inverse_transform(x)
        np.testing.assert_allclose(y, y_rec, rtol=1e-9, atol=1e-9)


def test_inverse_transform_boundary(trans3):
    """Hypercube boundary points should be reconstructed correctly."""
    # y1 = L, y2 = U
    y1 = np.array([trans3.L, trans3.U])
    x1 = trans3.transform(y1)
    y1_rec = trans3.inverse_transform(x1)
    np.testing.assert_allclose(y1, y1_rec)

    # y1 = U, y2 = L
    y2 = np.array([trans3.U, trans3.L])
    x2 = trans3.transform(y2)
    y2_rec = trans3.inverse_transform(x2)
    np.testing.assert_allclose(y2, y2_rec)

    # All values at L.
    y_all_L = np.full(trans3.dim, trans3.L)
    x_all_L = trans3.transform(y_all_L)
    y_all_L_rec = trans3.inverse_transform(x_all_L)
    np.testing.assert_allclose(y_all_L, y_all_L_rec)

    # All values at U.
    y_all_U = np.full(trans3.dim, trans3.U)
    x_all_U = trans3.transform(y_all_U)
    y_all_U_rec = trans3.inverse_transform(x_all_U)
    np.testing.assert_allclose(y_all_U, y_all_U_rec)


# ------------------ validate tests ------------------
def test_validate(trans3):
    x_good = np.array([0.2, 0.5, 0.3])
    assert trans3.validate(x_good) is True

    x_bad_sum = np.array([0.2, 0.5, 0.2])
    assert trans3.validate(x_bad_sum) is False

    x_bad_neg = np.array([-0.1, 0.6, 0.5])
    assert trans3.validate(x_bad_neg) is False

    x_bad_gt = np.array([0.2, 1.2, -0.4])
    assert trans3.validate(x_bad_gt) is False

    x_wrong_len = np.array([0.2, 0.8])
    assert trans3.validate(x_wrong_len) is False


# ------------------ inverse_transform error tests ------------------
def test_inverse_transform_invalid(trans3):
    x_bad1 = np.array([0.2, 0.5, 0.4])   # sum is 1.1
    with pytest.raises(ValueError, match="do not lie on the simplex"):
        trans3.inverse_transform(x_bad1)

    x_bad2 = np.array([-0.1, 0.6, 0.5])
    with pytest.raises(ValueError, match="do not lie on the simplex"):
        trans3.inverse_transform(x_bad2)

    # A boundary point with x_K=0 is valid and should not raise.
    x_edge = np.array([0.2, 0.8, 0.0])
    y = trans3.inverse_transform(x_edge)   # should succeed
    assert len(y) == trans3.dim


# ------------------ K=2 tests ------------------
def test_k2(trans2):
    y = np.array([0.3])
    x = trans2.transform(y)
    np.testing.assert_allclose(x, [0.3, 0.7])
    y_rec = trans2.inverse_transform(x)
    np.testing.assert_allclose(y, y_rec)

    y0 = np.array([0.0])
    x0 = trans2.transform(y0)
    # Use a slightly looser tolerance at the boundary.
    np.testing.assert_allclose(x0, [0.0, 1.0], atol=1e-14)
    y0_rec = trans2.inverse_transform(x0)
    np.testing.assert_allclose(y0, y0_rec, atol=1e-14)

    y1 = np.array([1.0])
    x1 = trans2.transform(y1)
    np.testing.assert_allclose(x1, [1.0, 0.0], atol=1e-14)
    y1_rec = trans2.inverse_transform(x1)
    np.testing.assert_allclose(y1, y1_rec, atol=1e-14)


# ------------------ K=4 tests with many random points ------------------
def test_k4_random(trans4):
    for _ in range(30):
        y = np.random.uniform(trans4.L, trans4.U, size=trans4.dim)
        x = trans4.transform(y)
        assert np.sum(x) == pytest.approx(1.0, abs=1e-10)
        y_rec = trans4.inverse_transform(x)
        np.testing.assert_allclose(y, y_rec, rtol=1e-9, atol=1e-9)


# ------------------ Numerical-stability test for tiny components ------------------
def test_tiny_components(trans3):
    x_tiny = np.array([1e-15, 0.5, 1.0 - 1e-15 - 0.5])
    y = trans3.inverse_transform(x_tiny)
    x_rec = trans3.transform(y)
    np.testing.assert_allclose(x_tiny, x_rec, rtol=1e-8, atol=1e-8)

# ------------------ Batch-processing (2D) tests ------------------
def test_batch_transform_shape(trans3):
    """Check that a 2D input produces a 2D array with the expected shape."""
    y_batch = np.random.uniform(trans3.L, trans3.U, size=(10, trans3.dim))
    x_batch = trans3.transform(y_batch)
    assert x_batch.shape == (10, trans3.K)
    # A single row is still a 2D input.
    y_single = np.random.uniform(trans3.L, trans3.U, size=(1, trans3.dim))
    x_single = trans3.transform(y_single)
    assert x_single.shape == (1, trans3.K)

def test_batch_transform_properties(trans4):
    """Check simplex properties for every point in a batch."""
    y_batch = np.random.uniform(trans4.L, trans4.U, size=(20, trans4.dim))
    x_batch = trans4.transform(y_batch)
    assert np.allclose(np.sum(x_batch, axis=1), 1.0, atol=1e-10)
    assert np.all(x_batch >= 0.0) and np.all(x_batch <= 1.0)

def test_batch_inverse_recovery(trans3):
    """transform -> inverse_transform for a batch of points."""
    y_batch = np.random.uniform(trans3.L, trans3.U, size=(15, trans3.dim))
    x_batch = trans3.transform(y_batch)
    y_rec = trans3.inverse_transform(x_batch)
    np.testing.assert_allclose(y_batch, y_rec, rtol=1e-9, atol=1e-9)

def test_batch_validate(trans3):
    """Check validation on 2D arrays."""
    good = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert trans3.validate(good) is True

    bad_sum = np.array([[0.2, 0.2, 0.5], [0.1, 0.1, 0.7]])   # second row does not sum to 1
    assert trans3.validate(bad_sum) is False

    bad_neg = np.array([[0.2, 0.3, 0.5], [-0.1, 0.6, 0.5]])
    assert trans3.validate(bad_neg) is False

    bad_shape = np.random.rand(2, 4)
    assert trans3.validate(bad_shape) is False

def test_batch_inverse_invalid(trans3):
    """Check that inverse_transform rejects an invalid batch."""
    x_bad = np.array([[0.2, 0.3, 0.5], [0.2, 0.5, 0.4]])   # second row sums to 1.1
    with pytest.raises(ValueError, match="do not lie on the simplex"):
        trans3.inverse_transform(x_bad)

def test_backward_compatibility_single_point(trans3):
    """Check that a 1D input returns a 1D output for backward compatibility."""
    y = np.random.uniform(trans3.L, trans3.U, size=trans3.dim)
    x = trans3.transform(y)
    assert x.ndim == 1
    assert x.shape == (trans3.K,)
    y_rec = trans3.inverse_transform(x)
    assert y_rec.ndim == 1
    assert y_rec.shape == (trans3.dim,)
    np.testing.assert_allclose(y, y_rec)

def test_mixed_input_types(trans3):
    """Check that list, tuple, and similar inputs are accepted."""
    y_list = [0.5, -0.3]
    x = trans3.transform(y_list)
    assert x.shape == (trans3.K,)
    y_rec = trans3.inverse_transform(x.tolist())
    np.testing.assert_allclose(y_list, y_rec)

    # Batch supplied as a list of lists.
    y_batch_list = [[0.5, -0.3], [-0.5, 0.3]]
    x_batch = trans3.transform(y_batch_list)
    assert x_batch.shape == (2, trans3.K)
    y_batch_rec = trans3.inverse_transform(x_batch)
    np.testing.assert_allclose(y_batch_list, y_batch_rec)
