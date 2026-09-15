from pathlib import Path
import json

import numpy as np

from src.optimizers import RandomSearch
from src.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_random_search_bounds():
    opt = RandomSearch(5, 1)
    x = opt.suggest()
    assert x.shape == (1, 5)
    assert np.all((x >= 0) & (x <= 1))


def test_thermoelectric_display_transform():
    y = np.array([[-50.0, -70.0]])
    got = TASKS["thermoelectric"].display(y)
    assert np.allclose(got, [[1.0, 1.4]])


def test_reproducibility_notebook_is_valid_json():
    path = ROOT / "ZoMBI_reproducibility.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any(cell.get("cell_type") == "code" for cell in notebook["cells"])
