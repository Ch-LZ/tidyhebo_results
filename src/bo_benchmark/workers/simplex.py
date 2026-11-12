import numpy as np
from scipy.special import logit, softmax, expit

class SimplexTransformer:
    """
    Bijective transformation between the hypercube [L, U]^(K-1) and the K-dimensional simplex.

    Supports both single points (1D arrays) and batches of points (2D arrays, each row is a point).
    For a 1D input, a 1D output is returned; for a 2D input, a 2D output with the same batch size.

    The transformation uses the following scheme:
        1. Normalize the input vector y ∈ [L, U]^(K-1) to [0, 1]:
               t_i = (y_i - L) / (U - L),   i = 1..K-1
        2. Apply logit to map t to ℝ:
               z_i = logit(t_i) = log(t_i / (1 - t_i))
        3. Append a fixed z_K = 0
        4. Apply softmax to obtain x ∈ simplex Δ^K:
               x = softmax(z)

    The inverse recovers y from x on the simplex:
        1. Compute logits relative to the last component:
               z_i = log(x_i / x_K),   i = 1..K-1
        2. Inverse logit:
               t_i = expit(z_i) = 1 / (1 + exp(-z_i))
        3. Scale back to [L, U]:
               y_i = L + t_i * (U - L)

    The transformation is bijective on the open hypercube (L, U)^(K-1)
    and the open simplex (x_i > 0, sum x_i = 1). An eps guard is used
    near boundaries to avoid numerical issues.

    Parameters
    ----------
    K : int
        Dimension of the simplex (number of components). Must be >= 2.
    L : float, default=-2.0
        Lower bound for each of the K-1 parameters.
    U : float, default=2.0
        Upper bound for each of the K-1 parameters. Must satisfy L < U.
    eps : float, default=1e-15
        Guard against t leaving [0, 1] before logit. t is clipped to [eps, 1-eps].
    """

    def __init__(self, K, L=-2.0, U=2.0, eps=1e-15):
        if K < 2:
            raise ValueError("K must be >= 2")
        if L >= U:
            raise ValueError("L must be smaller than U")
        self.K = K
        self.L = L
        self.U = U
        self.dim = K - 1
        self.eps = eps

    def _prepare_input(self, arr, expected_cols, name="input"):
        """
        Convert input to a 2D array and validate its shape.

        Parameters
        ----------
        arr : array_like
            Input array (1D or 2D).
        expected_cols : int
            Expected number of columns (features) for each point.
        name : str, optional
            Name of the input for error messages (not used in the final message,
            but kept for clarity).

        Returns
        -------
        arr_2d : ndarray, shape (N, expected_cols)
            Input reshaped to 2D (if necessary).
        input_ndim : int
            Original number of dimensions (1 or 2) – used later to restore shape.

        Raises
        ------
        ValueError
            If shape is incompatible.
        """
        arr = np.asarray(arr)
        ndim = arr.ndim
        if ndim == 1:
            if arr.shape[0] != expected_cols:
                raise ValueError(f"Expected an array of length {expected_cols}, got {arr.shape}")
            arr_2d = arr.reshape(1, -1)
        elif ndim == 2:
            if arr.shape[1] != expected_cols:
                raise ValueError(f"Expected an array of shape (N, {expected_cols}), got {arr.shape}")
            arr_2d = arr
        else:
            raise ValueError("Input must be a 1D or 2D array")
        return arr_2d, ndim

    def _restore_output(self, arr_2d, input_ndim):
        """
        Restore the output shape to match the input's original number of dimensions.

        Parameters
        ----------
        arr_2d : ndarray, shape (N, ...)
            Output in 2D form (each row is one result).
        input_ndim : int
            Original dimensionality of the input (1 or 2).

        Returns
        -------
        out : ndarray
            If input_ndim == 1, returns a 1D array (first row);
            otherwise returns the 2D array unchanged.
        """
        return arr_2d[0] if input_ndim == 1 else arr_2d

    def transform(self, y):
        """
        Transform point(s) y from the hypercube [L, U]^(K-1) to point(s) on the simplex Δ^K.

        Parameters
        ----------
        y : array_like, shape (K-1,) or (N, K-1)
            Input point or batch of points in the hypercube.

        Returns
        -------
        x : ndarray, shape (K,) or (N, K)
            Point or batch of points on the simplex.
        """
        y_2d, ndim = self._prepare_input(y, self.dim, "y")

        # Normalize to [0,1]
        t = (y_2d - self.L) / (self.U - self.L)
        t = np.clip(t, self.eps, 1.0 - self.eps)
        # Logit
        z = logit(t)
        # Append zero for the last component
        z = np.concatenate([z, np.zeros((z.shape[0], 1))], axis=1)
        # Softmax
        x_2d = softmax(z, axis=1)

        return self._restore_output(x_2d, ndim)

    def inverse_transform(self, x, tol=1e-10):
        """
        Recover coordinates y in the hypercube [L, U]^(K-1) from point(s) x on the simplex.

        Parameters
        ----------
        x : array_like, shape (K,) or (N, K)
            Point or batch of points on the simplex.
        tol : float, default=1e-10
            Tolerance for simplex membership check and handling of boundary cases.

        Returns
        -------
        y : ndarray, shape (K-1,) or (N, K-1)
            Recovered point or batch of points in the hypercube.
        """
        x_2d, ndim = self._prepare_input(x, self.K, "x")

        if not self.validate(x_2d, tol=tol):
            raise ValueError("Input point(s) do not lie on the simplex within tolerance")

        # Guard against x_K = 0 (boundary) for each row
        x_fixed = x_2d.copy()
        mask = x_fixed[:, -1] <= tol
        if np.any(mask):
            x_fixed[mask, -1] = np.maximum(x_fixed[mask, -1], tol)
            sums = np.sum(x_fixed[mask], axis=1, keepdims=True)
            x_fixed[mask] = x_fixed[mask] / sums

        xK = x_fixed[:, -1]  # (N,)
        # Compute log-ratios relative to last component
        z = np.log(x_fixed[:, :-1] / xK[:, None])  # (N, K-1)
        # Inverse logit
        t = expit(z)
        # Scale back to [L, U]
        y_2d = self.L + t * (self.U - self.L)
        y_2d = np.clip(y_2d, self.L, self.U)

        return self._restore_output(y_2d, ndim)

    def validate(self, x, tol=1e-8):
        """
        Check whether point(s) x belong to the simplex Δ^K.

        Parameters
        ----------
        x : array_like, shape (K,) or (N, K)
            Point or batch of points to check.
        tol : float, default=1e-8
            Tolerance: components must be >= -tol, and sum of each row must be 1 within tol.

        Returns
        -------
        bool
            True if all points belong to the simplex, False otherwise.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            if len(x) != self.K:
                return False
            if np.any(x < -tol) or np.any(x > 1 + tol):
                return False
            if not np.isclose(np.sum(x), 1.0, atol=tol):
                return False
            return True
        elif x.ndim == 2:
            if x.shape[1] != self.K:
                return False
            if np.any(x < -tol) or np.any(x > 1 + tol):
                return False
            sums = np.sum(x, axis=1)
            if not np.all(np.isclose(sums, 1.0, atol=tol)):
                return False
            return True
        else:
            return False
