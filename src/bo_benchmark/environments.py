"""Mamba environment helpers; no installs are triggered by experiment execution."""
from pathlib import Path
from .mamba_support import find_mamba, run_command


def mamba_run(prefix: Path, command):
    return run_command(find_mamba(), prefix, command)
