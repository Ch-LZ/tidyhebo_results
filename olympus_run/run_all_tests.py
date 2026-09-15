#!/usr/bin/env python3
"""Run the Olympus test suite in all three Conda environments.

Common tests run in every environment; environment-specific tests run only in
the environment selected by their pytest marker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ENVS = ("o11", "hebo", "tidyhebo")
PROJECT_ROOT = Path(__file__).resolve().parent


def environment_runner() -> str:
    """Return the available Conda-compatible executable, preferring Mamba."""
    for executable in ("mamba", "conda"):
        if shutil.which(executable):
            return executable
    raise FileNotFoundError("Neither mamba nor conda is available in PATH")


def run_tests(runner: str, env: str, pytest_args: list[str]) -> None:
    """Run pytest in one environment with the appropriate marker expression."""
    marker_expr = f"env_common or env_{env}"
    cmd = [
        runner,
        "run",
        "-n",
        env,
        "python",
        "-m",
        "pytest",
        "-m",
        marker_expr,
        "tests/",
        "-rsv",
        *pytest_args,
    ]

    print(f"\n[RUN] Environment: {env}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if result.returncode == 5:
        print(f"[INFO] No matching tests in environment {env}.")
    elif result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tests in all benchmark environments; extra arguments are passed to pytest."
    )
    _, pytest_args = parser.parse_known_args()

    try:
        runner = environment_runner()
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for env in ENVS:
        run_tests(runner, env, pytest_args)


if __name__ == "__main__":
    main()
