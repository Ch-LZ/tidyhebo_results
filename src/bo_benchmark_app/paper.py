"""The three frozen campaigns reported in the paper."""
from pathlib import Path
from .configuration import ROOT

METHODS = ('random', 'optuna', 'hebo', 'vanilla_gp_qlogei', 'yj_qlogei', 'qmace_retained')
CAMPAIGNS = {
    'olympus': dict(config='OLY-002.json', repeats=30, q=(1,2,4,8,16), budget=100, n_init=4, tasks=19),
    'niah': dict(config='NIAH-005.json', repeats=12, q=(1,), budget=100, n_init=4, tasks=2),
    'bayesmark': dict(config='BAYESMARK-Q8-FULL108.json', repeats=15, q=(8,), budget=128, n_init=8, tasks=108),
}


def config_path(name):
    return ROOT / 'experiments' / CAMPAIGNS[name]['config']



PARQUET_FORMAT = 'tidyhebo-paper-parquet-v1'
NIAH_DISPLAY = {'poisson_ratio': (1.0, 1.0), 'thermoelectric': (-1.0, 50.0)}


def _parquet_rows(path, name):
    import json
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                x = row['initial_x_json']
                yield {
                    'status': 'completed',
                    'spec': dict(benchmark=name, task=row['task'], variant=row['variant'],
                                 q=row['q'], seed=row['seed'], repeat=row['repeat'],
                                 n_trials=row['budget'], n_init=row['n_init']),
                    'y': row['y'], 'batches': row['batches'], 'minimize': row['minimize'],
                    'initial_x': json.loads(x) if x is not None else None,
                }


def _validate_records(name, rows):
    import numpy as np
    from bo_benchmark.seeds import SeedManager

    definition = CAMPAIGNS[name]
    seeds = SeedManager(42,definition['repeats']).seeds
    records = {}
    initial = {}
    for r in rows:
        s = r['spec']
        key = s['task'],s['variant'],s['q'],s['seed']
        if key in records:
            raise ValueError('Duplicate historical run: '+str(key))
        if (r['status'] != 'completed' or s['variant'] not in METHODS
                or s['q'] not in definition['q'] or s['n_trials'] != definition['budget']
                or s['n_init'] != definition['n_init'] or not 0 <= s['repeat'] < len(seeds)
                or s['seed'] != seeds[s['repeat']] or len(r['y']) != definition['budget']
                or not np.isfinite(r['y']).all()
                or any(type(size) is not int or size < 1 for size in r['batches'])
                or sum(r['batches']) != len(r['y'])):
            raise ValueError('Invalid/incomplete paper run: '+str(key))
        records[key] = r
        points = r.get('initial_x')
        if points is None or len(points) != s['n_init'] or not np.isfinite(points).all():
            raise ValueError('Historical initial X missing or invalid: '+str(key))
        pair = s['task'],s['seed']
        if pair in initial and not np.array_equal(initial[pair],points):
            raise ValueError('Historical initial designs are not paired: '+str(key))
        initial[pair] = points
    tasks = sorted({k[0] for k in records})
    wanted = {(t,m,q,s) for t in tasks for m in METHODS for q in definition['q'] for s in seeds}
    if len(tasks) != definition['tasks'] or set(records) != wanted:
        raise ValueError(f'Incomplete {name} history: {len(tasks)} tasks, {len(records)} runs; '
                         f'expected {definition["tasks"]} tasks and '
                         f'{definition["tasks"]*len(METHODS)*len(definition["q"])*len(seeds)} runs')
    return records



def history_records(name, root=ROOT):
    """Read frozen Parquet or the original gzip JSONL format, with integrity checks."""
    import gzip
    import hashlib
    import json

    folder = Path(root) / 'data/historical'
    parquet = folder / (name + '.parquet')
    if parquet.exists() or (folder/'parquet_manifest.json').exists():
        manifest = json.loads((folder/'parquet_manifest.json').read_text(encoding='utf-8-sig'))
        if (manifest['format'] != PARQUET_FORMAT
                or manifest['initial_coordinates'] != 'per_evaluation'):
            raise ValueError('Unsupported Parquet history format')
        if name == 'olympus':
            from bo_benchmark.task_catalog import load_catalog
            if manifest['olympus']['task_catalog'] != load_catalog():
                raise ValueError('Olympus Parquet task catalog differs from the repository')
        expected = manifest['sha256']['data/historical/'+parquet.name]
        if hashlib.sha256(parquet.read_bytes()).hexdigest() != expected:
            raise ValueError('Historical data checksum mismatch: '+str(parquet))
        records = _validate_records(name, _parquet_rows(parquet, name))
        return records, manifest

    path = folder / (name + '.jsonl.gz')
    if not path.is_file():
        raise FileNotFoundError(f'Paper history missing: {parquet}. Run import-data with the paper archive.')
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'))
    expected = manifest['sha256']['data/historical/'+path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Historical data checksum mismatch: '+str(path))
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return _validate_records(name, (json.loads(line) for line in stream)), manifest


def _import_parquet(archive, root):
    import hashlib
    import json
    import os
    import tempfile

    names = [name + '.parquet' for name in CAMPAIGNS]
    members = archive.namelist()
    if sorted(members) != sorted([*names, 'manifest.json']):
        raise ValueError('Unexpected members in paper Parquet archive')
    manifest = json.loads(archive.read('manifest.json'))
    if (manifest['format'] != PARQUET_FORMAT
            or manifest['initial_coordinates'] != 'per_evaluation'):
        raise ValueError('Expected the compact paper history archive with per-evaluation initial coordinates')
    manifest['sha256'] = {}
    target = root/'data/historical'
    with tempfile.TemporaryDirectory(prefix='.paper-import-', dir=root) as temp:
        staging = Path(temp)/'data/historical'
        staging.mkdir(parents=True)
        for name in names:
            data = archive.read(name)
            (staging/name).write_bytes(data)
            manifest['sha256']['data/historical/'+name] = hashlib.sha256(data).hexdigest()
        manifest_name = 'parquet_manifest.json'
        (staging/manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        for name in CAMPAIGNS:
            records, _ = history_records(name, temp)
            print(f'Validated {name}: {len(records)} complete paper runs', flush=True)
        files = [*names, manifest_name]
        for name in files:
            final = target/name
            if final.exists() and final.read_bytes() != (staging/name).read_bytes():
                raise ValueError('Refusing to overwrite a different frozen file: '+str(final))
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            if not (target/name).exists():
                os.replace(staging/name, target/name)
    print('Paper history:', target, flush=True)


def import_archive(path, root=ROOT):
    """Import only paper histories and oracle files from the bounded exporter."""
    import hashlib
    import json
    import zipfile
    root = Path(root).resolve()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(item.file_size for item in members) > 300*1024*1024:
            raise ValueError('Unexpectedly large paper-data archive')
        if 'manifest.json' in archive.namelist():
            return _import_parquet(archive, root)
        pending = []
        for item in members:
            name = item.filename
            target = (root/name).resolve()
            if (not target.is_relative_to(root) or '\\' in name
                    or not name.startswith(('data/historical/','data/niah/NIAH005/prepared/',
                                             'vendor/olympus/','results/bayesmark_q8_full108/analysis/',
                                             'results/NIAH-005/analysis/'))):
                raise ValueError('Unexpected paper-data member: '+name)
        for item in members:
            name = item.filename
            if item.is_dir():
                continue
            if name.startswith('results/bayesmark_q8_full108/analysis/'):
                target = root/'data/historical/bayesmark_tables'/Path(name).name
            elif name.startswith('results/NIAH-005/analysis/'):
                target = root/'data/historical/sources/niah_analysis'/Path(name).name
            else:
                target = root/name
            data = archive.read(item)
            if target.exists() and target.read_bytes() != data:
                same = False
                if name.startswith('data/niah/') and target.suffix == '.npz':
                    # Regeneration can change ZIP timestamps while retaining the
                    # exact arrays and content-based asset identity.
                    import io
                    import numpy as np
                    with np.load(target,allow_pickle=False) as a, np.load(io.BytesIO(data),allow_pickle=False) as b:
                        same = set(a.files) == set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
                elif target.suffix == '.json':
                    same = json.loads(target.read_bytes()) == json.loads(data)
                elif target.suffix == '.py':
                    same = target.read_bytes().replace(b'\r\n',b'\n') == data.replace(b'\r\n',b'\n')
                if same:
                    continue
                raise ValueError('Refusing to overwrite a different frozen file: '+str(target))
            pending.append((target,data))
        for target,data in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    for name in CAMPAIGNS:
        records,_ = history_records(name,root)
        print(f'Imported {name}: {len(records)} complete paper runs',flush=True)
