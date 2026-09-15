import sys
from pathlib import Path
import yaml
import pytest
import numpy as np
import os
from typing import Any, Dict

import pytest

from src.tasks import OLYMPUS_TASKS

# Base directory containing task configs.
CONFIG_BASE = Path(__file__).parent.parent.parent / "configs" / "tasks"

pytestmark = pytest.mark.env_o11

def find_config(dataset):
    """
    Find a dataset config below CONFIG_BASE and return its Path, or None.
    """
    if not CONFIG_BASE.exists():
        print(f"[WARN] Directory does not exist: {CONFIG_BASE}")
        return None

    for subdir in CONFIG_BASE.iterdir():
        if not subdir.is_dir():
            continue
        for ext in ('.yaml', '.yml'):
            candidate = subdir / f"{dataset}{ext}"
            if candidate.exists():
                return candidate
    return None


@pytest.mark.parametrize("dataset", OLYMPUS_TASKS)
def test_config_matches_emulator(dataset):
    """Check that emulator parameter ranges match the task configuration."""
    from olympus import Emulator


    config_path = find_config(dataset)
    if config_path is None:
        # Print available files to aid debugging.
        print(f"\n[DEBUG] No config found for {dataset}. Contents of {CONFIG_BASE}:")
        if CONFIG_BASE.exists():
            for subdir in CONFIG_BASE.iterdir():
                if subdir.is_dir():
                    files = [f.name for f in subdir.iterdir() if f.is_file()]
                    print(f"  {subdir.name}/: {files}")
        pytest.skip(f"No config for {dataset} found in {CONFIG_BASE}")

    print(f"\n[INFO] Found config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    em = Emulator(dataset, "BayesNeuralNet")
    assert em.is_trained, f"Emulator for {dataset} is not trained"

    config_params: dict[str, dict] = {p['name']: p for p in config['params']}
    em_params = em.param_space.parameters

    if 'less_is_better' in config:
        assert config['less_is_better'] == (em.dataset.goal == 'minimize'), \
            f"less_is_better mismatch for {dataset}; em.dataset.goal=={em.dataset.goal}"

    # Required parameter.
    if 'on_simplex' in config:
        assert config['on_simplex'] == (em.dataset.parameter_constriants == 'simplex'), \
            f"on_simplex mismatch for {dataset}"

    if config.get('on_simplex') or False:
        check_params_on_simplex(config_params, em_params)
    else:
        check_params_direct(config_params, em_params)


def check_params_direct(config_params: Dict[str, dict], em_params):
    assert len(config_params) == len(em_params), \
        f"Parameter count mismatch: config={len(config_params)}, emulator={len(em_params)}"

    for em_param in em_params:
        name = em_param.name
        assert name in config_params, f"Parameter {name} is missing from the config"
        cfg_p = config_params[name]

        if cfg_p['type'] == 'continuous':
            assert np.isclose(em_param.low, cfg_p['low']), \
                f"low mismatch for {name}: em={em_param.low}, cfg={cfg_p['low']}"
            assert np.isclose(em_param.high, cfg_p['high']), \
                f"high mismatch for {name}: em={em_param.high}, cfg={cfg_p['high']}"
        elif cfg_p['type'] == 'categorical':
            assert hasattr(em_param, 'categories'), \
                f"Parameter {name} must be categorical"
            assert set(em_param.categories) == set(cfg_p['categories']), \
                f"Category mismatch for {name}: em={em_param.categories}, cfg={cfg_p['categories']}"
        else:
            pytest.fail(f"Unknown parameter type {cfg_p['type']} for {name}")


def check_params_on_simplex(config_params: Dict[str, dict], em_params):
    """
    All parameters should be present with range [-2, 2]; the simplex transform then maps them into the emulator parameter space.
    """

    # A simplex has one fewer independent coordinate, but the config still lists all parameters.
    assert len(config_params) == len(em_params), \
        f"Parameter count mismatch: config={len(config_params)}, emulator={len(em_params)}"

    for em_param in em_params:
            name = f"{em_param.name}_aux"
            assert name in config_params, f"Parameter {name} is missing from the config"
            cfg_p = config_params[name]

            if cfg_p['type'] == 'continuous':
                assert np.isclose(-2.0, cfg_p['low']), \
                    f"low mismatch for {name}: em={-2.0}, cfg={cfg_p['low']}"
                assert np.isclose(2.0, cfg_p['high']), \
                    f"high mismatch for {name}: em={2.0}, cfg={cfg_p['high']}"
            elif cfg_p['type'] == 'categorical':
                pytest.fail(f"Does not expect canteogiral on simplex (parameter: {name})")
            else:
                pytest.fail(f"Unknown parameter type {cfg_p['type']} for {name}")
