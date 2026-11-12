"""Read the tracked YAML recipes and local environment profiles."""

from pathlib import Path


def read_yaml(path):
    import yaml
    with Path(path).open(encoding='utf-8-sig') as stream:
        return yaml.safe_load(stream)


def write_yaml(path, data):
    import yaml
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def requirements(path):
    packages = list(read_yaml(path)['dependencies'])
    python = packages.pop(0)
    if not python.startswith('python='):
        raise ValueError(f'Expected python=VERSION in {path}')
    return python.split('=', 1)[1], packages
