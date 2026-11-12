"""Small stdlib helpers for the user's existing mamba installation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_mamba():
    candidates = [os.environ.get('MAMBA_EXE'), shutil.which('mamba')]
    if os.environ.get('CONDA_EXE'):
        base = Path(os.environ['CONDA_EXE']).parent
        candidates += [base/'mamba.exe', base/'mamba', base.parent/'Library/bin/mamba.exe']
    candidates += [Path(sys.prefix)/'bin/mamba', Path(sys.prefix)/'Scripts/mamba.exe',
                   Path(sys.prefix)/'Library/bin/mamba.exe']
    if os.name == 'nt':
        homes = [Path.home(), Path(os.environ.get('LOCALAPPDATA', str(Path.home())))]
        for home in homes:
            for name in ('miniforge3', 'mambaforge', 'miniconda3', 'anaconda3'):
                base = home/name
                candidates += [base/'Library/bin/mamba.exe', base/'Scripts/mamba.exe', base/'condabin/mamba.bat']
    for value in candidates:
        if value and Path(value).is_file():
            path = Path(value).absolute()
            # Miniforge often exposes condabin/mamba.bat on PATH. Prefer its
            # native sibling when present, avoiding the shared CMD wrapper.
            if path.suffix.lower() in ('.bat', '.cmd'):
                for native in (path.with_suffix('.exe'),
                               path.parent.parent/'Library/bin/mamba.exe',
                               path.parent.parent/'Scripts/mamba.exe'):
                    if native.is_file():
                        return str(native)
            return str(path)
    raise RuntimeError('mamba not found. Start from your Mamba/Miniforge terminal or set MAMBA_EXE.')


def run_command(mamba, prefix, command, *, env=None):
    # Older mamba delegates to conda run; newer mamba streams by default.
    help_result = subprocess.run([str(mamba), 'run', '--help'], capture_output=True, text=True, check=True, env=env)
    streaming = ['--no-capture-output'] if '--no-capture-output' in help_result.stdout else []
    return [str(mamba), 'run', *streaming, '-p', str(prefix), *map(str, command)]
