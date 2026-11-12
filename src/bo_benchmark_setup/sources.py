"""Prepare pinned GitHub sources before creating the paper environments."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlopen

from .definitions import read_yaml

ROOT = Path(__file__).resolve().parents[2]
# The archived copy changed these files for installation and Windows path handling.
# Dataset and checkpoint bytes must match the archived hashes.
OLYMPUS_ADAPT = {
    'setup.py', 'src/olympus/_version.py',
    'src/olympus/databases/__init__.py', 'src/olympus/datasets/__init__.py',
    'src/olympus/datasets/dataset.py', 'src/olympus/emulators/emulator.py',
    'src/olympus/models/__init__.py', 'src/olympus/noises/__init__.py',
    'src/olympus/planners/__init__.py', 'src/olympus/scalarizers/__init__.py',
    'src/olympus/surfaces/__init__.py',
}


def _git(folder, *args):
    return subprocess.check_output(['git', '-C', str(folder), *args], text=True).strip()


def _install_tree(destination, populate):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.source-', dir=destination.parent) as temp:
        stage = Path(temp) / 'tree'
        stage.mkdir()
        populate(stage)
        backup = destination.with_name(destination.name + '.previous')
        if destination.exists():
            if backup.exists():
                raise FileExistsError(f'Move the old source backup before retrying: {backup}')
            destination.rename(backup)
        try:
            stage.rename(destination)
        except OSError:
            if not destination.exists() and backup.exists():
                backup.rename(destination)
            raise


def fetch_tidyhebo(destination, spec):
    commit, tag = spec['commit'], spec['tag']
    if (destination/'.git').is_dir():
        if _git(destination, 'rev-parse', 'HEAD') == commit and not _git(
                destination, 'status', '--porcelain', '--untracked-files=no'):
            return

    def populate(stage):
        _git(stage, 'init', '-q')
        try:
            _git(stage, 'fetch', '--quiet', '--depth=1', '--no-tags', spec['repository'], 'refs/tags/'+tag)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Could not fetch tidyHEBO {tag}. Publish the tag at {commit}.') from exc
        if _git(stage, 'rev-parse', 'FETCH_HEAD^{commit}') != commit:
            raise ValueError(f'tidyHEBO {tag} does not point at {commit}')
        _git(stage, '-c', 'core.autocrlf=false', 'checkout', '--quiet', '--detach', commit)

    _install_tree(destination, populate)


def _olympus_bytes(name, data, expected):
    if hashlib.sha256(data).hexdigest() == expected:
        return data
    lf = data.replace(b'\r\n', b'\n')
    for normalized in (lf, lf.replace(b'\n', b'\r\n')):
        if hashlib.sha256(normalized).hexdigest() == expected:
            return normalized
    if name not in OLYMPUS_ADAPT:
        raise ValueError(f'Olympus file checksum differs: {name}')
    if name in ('setup.py', 'src/olympus/_version.py'):
        return data  # Editable installation reads assets directly from the source tree.
    text = data.decode('utf-8')
    updated, count = re.subn(r"""(\w+)\.split\(["']/["']\)\[-1\]""", r'os.path.basename(\1)', text)
    if not count:
        raise ValueError(f'Expected a Windows path fix in Olympus file: {name}')
    if not re.search(r'^import os\b', updated, flags=re.M):
        import ast
        first = ast.parse(updated).body[0]
        line = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else 0
        parts = updated.splitlines(keepends=True)
        parts.insert(line, 'import os\n')
        updated = ''.join(parts)
    return updated.encode('utf-8')


def fetch_olympus(destination, spec, files):
    marker = destination/'.paper-commit'
    if marker.is_file() and marker.read_text().strip() == spec['commit']:
        if all((destination/name).is_file() and (
                name in OLYMPUS_ADAPT or hashlib.sha256((destination/name).read_bytes()).hexdigest() == expected)
               for name, expected in files.items()):
            return
        raise ValueError(f'Olympus source is incomplete or changed: {destination}')

    prefix = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/"

    def populate(stage):
        def fetch(item):
            name, expected = item
            target = stage/name
            for attempt in range(3):
                try:
                    with urlopen(prefix+quote(name), timeout=40) as stream:
                        data = stream.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_olympus_bytes(name, data, expected))
        with ThreadPoolExecutor(max_workers=12) as pool:
            for _ in pool.map(fetch, files.items()):
                pass
        (stage/'.paper-commit').write_text(spec['commit']+'\n', encoding='ascii')

    _install_tree(destination, populate)


def prepare_sources(*, olympus):
    env = ROOT/'environments'
    specs = read_yaml(env/'sources.yaml')
    print('Preparing pinned tidyHEBO source', flush=True)
    fetch_tidyhebo(ROOT/'vendor/tidyhebo', specs['tidyhebo'])
    if olympus:
        print('Preparing pinned Olympus source and emulator assets', flush=True)
        spec = specs['olympus']
        fetch_olympus(ROOT/'vendor/olympus', spec, read_yaml(env/spec['files']))
