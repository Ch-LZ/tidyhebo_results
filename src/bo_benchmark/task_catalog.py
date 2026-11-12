"""Validated Olympus task definitions, usable in the Python 3.10 oracle too."""
import hashlib
import json
import math
from pathlib import Path

CATALOG = Path(__file__).parent / 'assets/olympus_tasks.json'


def task_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_catalog():
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    if len(catalog) != 19:
        raise ValueError('Expected all 19 Olympus tasks')
    for name, config in catalog.items():
        best, worst = config['y_p_star'], config['y_p_w']
        if (type(config['less_is_better']) is not bool or type(config['on_simplex']) is not bool
                or not all(math.isfinite(x) for x in (best, worst)) or best == worst
                or ((best < worst) != config['less_is_better'])):
            raise ValueError('Invalid normalization/direction for '+name)
        params = config['params']
        if (not params or len({p['name'] for p in params}) != len(params)
                or any(p['type'] != 'continuous' or not math.isfinite(p['low'])
                       or not math.isfinite(p['high']) or p['low'] >= p['high'] for p in params)):
            raise ValueError('Invalid domain for '+name)
        if config['on_simplex'] and any((p['low'],p['high']) != (-2.,2.) for p in params):
            raise ValueError('Simplex mapping requires [-2, 2] auxiliary coordinates')
    return catalog


def task_reference(name):
    config = load_catalog()[name]
    return {'task_config': config, 'task_config_sha256': task_digest(config),
            'normalization': {'y_best':config['y_p_star'], 'y_worst':config['y_p_w']}}
