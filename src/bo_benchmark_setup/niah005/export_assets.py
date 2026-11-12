"""Re-export pinned ZoMBI forests without loading legacy sklearn estimators.

Optional regeneration utility, not needed for execution. Requires joblib 1.5.3,
NumPy and sklearn (native Tree is used only as an independent prediction check).
Old tree states contain no missing-value routing; inputs must be finite.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import joblib
import joblib.numpy_pickle as jp
from sklearn.tree import _tree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/bo_benchmark/workers'))
from niah_assets import SOURCES, COMMIT, FIXED_REFERENCES, content_id, ForestOracle


class LegacyState:
    """Passive container for verified pickle state, never a fitted estimator."""
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.__dict__.update(state)


class ForestReader(jp.NumpyUnpickler):
    def find_class(self, module, name):
        if module.startswith('sklearn.'):
            if (module, name) not in {
                ('sklearn.ensemble._forest', 'RandomForestRegressor'),
                ('sklearn.tree._classes', 'DecisionTreeRegressor'),
                ('sklearn.tree._tree', 'Tree'),
            }:
                raise ValueError(f'Unexpected estimator class: {module}.{name}')
            return LegacyState
        return super().find_class(module, name)


def export(task, raw, out):
    expected = SOURCES[task][1]
    if hashlib.sha256(raw.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned source checksum mismatch')
    if joblib.__version__ != '1.5.3':
        raise ValueError('Regeneration reader requires joblib==1.5.3')
    # Private Joblib decoding is intentionally confined to this versioned utility.
    with raw.open('rb') as f:
        with jp._validate_fileobject_and_memmap(f, str(raw), None) as (stream, _):
            forest = ForestReader(str(raw), stream, True).load()
    trees = [est.tree_ for est in forest.estimators_]
    offsets = np.cumsum([0] + [t.node_count for t in trees], dtype=np.int64)
    arrays = dict(roots=offsets[:-1],
        left=np.concatenate([np.where(t.nodes['left_child'] >= 0, t.nodes['left_child'] + start, -1)
                             for start, t in zip(offsets, trees)]),
        right=np.concatenate([np.where(t.nodes['right_child'] >= 0, t.nodes['right_child'] + start, -1)
                              for start, t in zip(offsets, trees)]),
        feature=np.concatenate([t.nodes['feature'] for t in trees]).astype(np.int16),
        threshold=np.concatenate([t.nodes['threshold'] for t in trees]),
        value=np.concatenate([t.values.reshape(-1) for t in trees]))
    best, worst = FIXED_REFERENCES[task]
    metadata = dict(protocol='niah-asset-v1', task=task, source_commit=COMMIT,
        source_file=SOURCES[task][0], source_sha256=expected, kind='forest',
        dim=int(forest.n_features_in_), trees=len(trees),
        feature_names=[f'x{i}' for i in range(forest.n_features_in_)],
        normalization=dict(y_best=best, y_worst=worst,
                           kind='fixed archived-observation references, not certified global extrema'),
        display_scale=50. if task=='thermoelectric' else 1.,
        display_sign=-1. if task=='thermoelectric' else 1.,
        needle_levels=[1.4] if task=='thermoelectric' else [-1.2, -1.7],
        conversion=dict(reader='passive-legacy-tree-v1',joblib=joblib.__version__,
                        source_sklearn=forest._sklearn_version))
    rng = np.random.RandomState(1729)
    x = rng.uniform(size=(1024, metadata['dim']))
    x = np.vstack([x, np.zeros((1, metadata['dim'])), np.ones((1, metadata['dim']))])
    # Exercise finite float32 points at and beside real split thresholds.
    threshold_points = []
    for t in trees[:50]:
        valid = np.flatnonzero((t.nodes['left_child'] >= 0) &
                              (t.nodes['threshold'] >= 0) & (t.nodes['threshold'] <= 1))
        if len(valid):
            node = int(valid[len(valid)//2]); feature = int(t.nodes['feature'][node])
            value = np.float32(t.nodes['threshold'][node])
            for split in [np.nextafter(value,np.float32(0)), value, np.nextafter(value,np.float32(1))]:
                point = rng.uniform(size=metadata['dim']); point[feature] = split
                threshold_points.append(point)
    x = np.vstack([x, threshold_points]); reference = np.zeros(len(x))
    for t in trees:
        native = _tree.Tree(*t.args)
        nodes = np.zeros(t.nodes.shape, dtype=_tree.NODE_DTYPE)
        for name in t.nodes.dtype.names:
            nodes[name] = t.nodes[name]
        native.__setstate__(dict(max_depth=t.max_depth, node_count=t.node_count,
                                 nodes=nodes, values=t.values))
        reference += native.predict(x.astype(np.float32)).reshape(-1)
    reference /= len(trees)
    portable = ForestOracle(arrays, metadata)(x)
    np.testing.assert_allclose(portable, reference, rtol=1e-12, atol=1e-12)
    metadata['validation'] = dict(points=len(x), reference='sklearn native Tree traversal',
        max_absolute_error=float(np.max(np.abs(portable-reference))))
    metadata['asset_id'] = content_id(arrays, metadata)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'{task}.npz', **arrays)
    (out/f'{task}.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(task, metadata['validation'], flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/niah/NIAH005/prepared')
    args=parser.parse_args()
    for task in ('poisson_ratio','thermoelectric'):
        export(task,args.raw_dir/(task+'.pkl'),args.output_dir)
