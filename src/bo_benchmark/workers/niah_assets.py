"""Version-independent ZoMBI RF inference and a finite Wildfire lookup oracle."""
from pathlib import Path
import hashlib
import json
import numpy as np

COMMIT = '04a7f0fa1a1a91c2afe60f1b0c60138e21c0de99'
SOURCES = {
    'poisson_ratio': ('data/poisson/poisson_RF_trained.pkl', 'b98b9a3d2a07840364232f77c5a19b48f8c1c15e2686a42c66aa0fbc49f0b132'),
    'thermoelectric': ('data/thermoelectric/zt_RF_trained.pkl', 'f5cf78bb82e46df69aedb848e300e06379011f86406532a98f94db9cedfb0790'),
    'wildfire': ('data/wildfires/fire_risk.pkl', 'b38ca22d087c9a74d1142c6c25214adced5517693633ad218a4b3469641efb9f'),
}
TASKS = tuple(SOURCES)
FIXED_REFERENCES = {
    'poisson_ratio': (-7.150534639650474, 49.13633514513019),
    'thermoelectric': (-209.5662761781472, -1.3705570121631987),
}


def content_id(arrays, metadata):
    digest = hashlib.sha256(json.dumps({k:v for k,v in metadata.items() if k != 'asset_id'}, sort_keys=True).encode())
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        digest.update(key.encode() + a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
    return digest.hexdigest()


class ForestOracle:
    def __init__(self, arrays, metadata):
        self.a, self.metadata = arrays, metadata
        self.dim = metadata['dim']

    def __call__(self, x):
        # sklearn trees convert input to float32 before double-threshold tests.
        x = np.asarray(x, dtype=np.float32).reshape(-1, self.dim)
        if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
            raise ValueError('RF oracle expects finite normalized coordinates in [0,1]')
        a = self.a
        total = np.zeros(len(x))
        for root in a['roots']:
            nodes = np.full(len(x), root, dtype=np.int64)
            active = a['left'][nodes] >= 0
            while active.any():
                rows = np.flatnonzero(active)
                node = nodes[rows]
                goes_left = x[rows, a['feature'][node]] <= a['threshold'][node]
                nodes[rows] = np.where(goes_left, a['left'][node], a['right'][node])
                active = a['left'][nodes] >= 0
            total += a['value'][nodes]
        return total / len(a['roots'])


class TableOracle:
    def __init__(self, arrays, metadata):
        self.a, self.metadata = arrays, metadata
        self.features = arrays['x']
        self.dim = self.features.shape[1]

    def query(self, row_id):
        if not 0 <= row_id < len(self.features):
            raise IndexError('Candidate ID outside Wildfire table')
        return float(self.a['y'][row_id])

    def physical(self, row_id):
        return self.features[row_id] * self.a['span'] + self.a['low']


def load_asset(data_dir, task):
    folder = Path(data_dir)
    metadata = json.loads((folder/f'{task}.json').read_text(encoding='utf-8'))
    if metadata['protocol'] != 'niah-asset-v1' or metadata['task'] != task or metadata['source_commit'] != COMMIT:
        raise ValueError('Incompatible prepared NIAH data')
    with np.load(folder/f'{task}.npz', allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if content_id(arrays, metadata) != metadata['asset_id']:
        raise ValueError(f'Prepared data integrity check failed: {task}')
    return TableOracle(arrays, metadata) if metadata['kind']=='table' else ForestOracle(arrays, metadata)


def export_candidates(data_dir):
    """Publish a feature-only Wildfire asset; optimizer never opens oracle targets."""
    folder = Path(data_dir)
    oracle = load_asset(folder, 'wildfire')
    arrays = {key: oracle.a[key] for key in ('x', 'low', 'span', 'source_row', 'counts')}
    metadata = {'protocol': 'niah-candidates-v1', 'task': 'wildfire',
                'source_asset_id': oracle.metadata['asset_id'], 'dim': oracle.dim,
                'rows': len(oracle.features)}
    metadata['asset_id'] = content_id(arrays, metadata)
    np.savez_compressed(folder/'wildfire_candidates.npz', **arrays)
    (folder/'wildfire_candidates.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def load_candidates(data_dir, source_asset_id):
    folder = Path(data_dir)
    metadata = json.loads((folder/'wildfire_candidates.json').read_text())
    if metadata.get('protocol') != 'niah-candidates-v1' or metadata.get('source_asset_id') != source_asset_id:
        raise ValueError('Wildfire candidate source mismatch')
    with np.load(folder/'wildfire_candidates.npz', allow_pickle=False) as source:
        if set(source.files) != {'x', 'low', 'span', 'source_row', 'counts'}:
            raise ValueError('Candidate asset must contain features only, no targets')
        arrays = {key: source[key] for key in source.files}
    if content_id(arrays, metadata) != metadata['asset_id']:
        raise ValueError('Wildfire candidate integrity check failed')
    x = arrays['x']
    if (x.shape != (metadata['rows'], metadata['dim']) or not np.isfinite(x).all()
            or ((x < 0) | (x > 1)).any() or len(np.unique(x, axis=0)) != len(x)):
        raise ValueError('Invalid/duplicate finite candidate features')
    return arrays, metadata
