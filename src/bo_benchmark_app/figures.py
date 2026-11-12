"""Figure recipes consume the shared core metrics and frozen historical data."""
from pathlib import Path
import json
import os
from .configuration import ROOT


def execute_notebook(path):
    """Run the plain Python recipe in this interpreter, without another kernel."""
    namespace = {'__name__':'__main__'}
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        for index, cell in enumerate(document['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                exec(compile(source,f'{path}:cell{index}','exec'),namespace)
    finally:
        os.chdir(previous)
    return namespace


def render(name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if name == 'olympus':
        for notebook in ('03_olympus_sequential.ipynb','04_olympus_batch.ipynb'):
            execute_notebook(ROOT/'notebooks'/notebook)
        from .supplement_figure import render as supplement
        supplement(ROOT)
        result = ROOT/'figures/olympus'
    elif name == 'niah':
        from .niah_figure import render as plot
        result = plot(ROOT)
    elif name == 'bayesmark':
        from .bayesmark_figure import render as plot
        result = plot(ROOT)
    else:
        raise ValueError('Unknown paper figure group: '+name)
    plt.close('all')
    print('Figures:',result,flush=True)
    return result
