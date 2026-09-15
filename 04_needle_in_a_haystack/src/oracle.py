from pathlib import Path
import joblib
import numpy as np


class PickleOracle:
    """Thin wrapper around the random-forest emulators used by ZoMBI."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Oracle file not found: {self.path}")
        self.model = joblib.load(self.path)
        if not callable(getattr(self.model, "predict", None)):
            raise TypeError(f"Object in {self.path} does not expose predict(X)")
        self.dim = getattr(self.model, "n_features_in_", None)
        if self.dim is None:
            raise TypeError(
                "Cannot infer oracle dimensionality (missing n_features_in_). "
                "Use a fitted sklearn-compatible model."
            )

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, self.dim)
        y = np.asarray(self.model.predict(x), dtype=float).reshape(-1)
        return y
