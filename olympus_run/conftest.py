# conftest.py
import os
import sys
from pathlib import Path

import pytest

# Add src to sys.path for all tests.
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    print(f"Inserting {src_path} as src_path to Path")
    sys.path.insert(0, str(src_path))

# Also expose the project root for configs and other shared resources.
project_root = Path(__file__).parent
# Fixture exposing the project root.
import pytest

@pytest.fixture(scope="session")
def project_root_path():
    return project_root

def pytest_configure(config):
    """Add environment information to pytest metadata."""
    env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    config._metadata = {'Environment': env}
    # Also record the Python version.
    config._metadata['Python'] = sys.version


@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(node, call, report):
    """Add the environment name to failure reports."""
    env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    report.sections.append((f"Environment: {env}", ""))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Show the environment name in the terminal summary."""
    env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    terminalreporter.write_sep("=", f"Environment: {env}")


@pytest.fixture(scope="session")
def env_name():
    """Return the name of the current Conda environment."""
    return os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
