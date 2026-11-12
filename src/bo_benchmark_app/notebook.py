"""Open a notebook next to an immutable compact analysis snapshot."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def launch(report):
    source=Path(__file__).resolve().parents[2]/'notebooks/olympus_results.ipynb'
    target=report/'results.ipynb'
    shutil.copyfile(source,target)
    print('Notebook: '+str(target),flush=True)
    module=next((n for n in ('jupyterlab','notebook') if importlib.util.find_spec(n)),None)
    if module is None and shutil.which('jupyter'):
        return subprocess.call([shutil.which('jupyter'),'lab',str(target)])
    if module is None:
        print('Report saved. Jupyter is missing from this environment; open the notebook using your existing Jupyter installation.',flush=True)
        return 0
    return subprocess.call([sys.executable,'-m',module,str(target)])
