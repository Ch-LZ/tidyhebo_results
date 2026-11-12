"""Run the legacy oracle in its configured interpreter."""

from bo_benchmark.mamba_support import find_mamba, run_command


def environment_command(definition, args):
    command = [definition['python'], *map(str, args)]
    prefix = definition.get('mamba_prefix')
    return run_command(find_mamba(), prefix, command) if prefix else command
