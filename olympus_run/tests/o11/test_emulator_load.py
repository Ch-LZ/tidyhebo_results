# tests/test_emulator_load.py

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
import pytest

from src.tasks import OLYMPUS_TASKS

pytestmark = pytest.mark.env_o11

@pytest.mark.parametrize("dataset", OLYMPUS_TASKS)
def test_emulator_loads(dataset):
    """Check that the emulator loads and predicts one point."""
    from olympus import Emulator

    # 1. Create the emulator.
    em = Emulator(dataset, "BayesNeuralNet")
    
    # 2. Check that it is trained.
    assert em.is_trained, f"Emulator for {dataset} is not trained"
    
    # 3. Obtain parameter bounds.
    # Prefer param_bounds when available.
    if hasattr(em.param_space, 'param_bounds'):
        bounds = em.param_space.param_bounds  # List of (low, high) pairs.
    else:
        # Fallback to individual parameter definitions.
        bounds = []
        for param in em.param_space.parameters:
            if hasattr(param, 'low') and hasattr(param, 'high'):
                bounds.append((param.low, param.high))
            elif hasattr(param, 'categories'):
                # Categorical parameter.
                bounds.append((param.categories, param.categories))
            else:
                bounds.append((0.0, 1.0))  # Fallback bound.

    # 4. Generate a random point within bounds.
    x = []
    for low, high in bounds:
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            x.append(np.random.uniform(low, high))
        elif isinstance(low, (list, tuple)):
            # For a categorical parameter, use the first category.
            x.append(low[0])
        else:
            x.append(0.0)
    x = np.array(x).reshape(1, -1)
    
    # 5. Run prediction.
    y = em.run(x)
    
    # 6. Debug output.
    print(f"\n[DEBUG] dataset={dataset}")
    print(f"  x = {x}")
    print(f"  y = {y}")
    
    # 7. Validate the result.
    assert y is not None, f"Prediction for {dataset} returned None"
    assert not np.any(np.isnan(y)), f"Prediction for {dataset} contains NaN"
    assert np.size(y) > 0, f"Prediction for {dataset} is empty"