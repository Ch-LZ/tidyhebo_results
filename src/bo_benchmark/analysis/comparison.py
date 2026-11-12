"""Strict raw-trajectory validation shared by campaign decisions and reports."""
import json
from pathlib import Path
import numpy as np
from ..task_catalog import task_reference
from .metrics import trajectory_metrics


def adapter_metadata(record):
    return record.get('provenance',{}).get('benchmark',{}).get('identity',{}).get('adapter',{})


def validate_record(record):
    s = record['spec']
    y = np.asarray(record['y'], dtype=float)
    batches = record['batches']
    if (y.ndim != 1 or len(y)<2 or not np.isfinite(y).all()
            or any(type(b) is not int or b<1 for b in batches) or sum(batches)!=len(y)
            or len(y)!=s.get('n_trials',len(y))):
        raise ValueError('Invalid/incomplete completed trajectory: '+str(record.get('run_id',s)))
    initial = s['n_init']
    if type(initial) is not int or not 0 <= initial < len(y):
        raise ValueError('Invalid initialization count')
    reference = task_reference(s['task'])
    meta = adapter_metadata(record)
    if 'task_config_sha256' in meta and meta['task_config_sha256']!=reference['task_config_sha256']:
        raise ValueError('Task reference changed; historical results require explicit reanalysis')
    if 'normalization' in meta and meta['normalization']!=reference['normalization']:
        raise ValueError('Stored normalization differs from audited task reference')
    minimize = reference['task_config']['less_is_better']
    if 'minimize' in record and record['minimize']!=minimize:
        raise ValueError('Raw objective direction conflicts with task reference')
    return reference


def protocol_key(record):
    s=record['spec'];meta=adapter_metadata(record)
    return (s.get('n_trials',len(record['y'])),s['n_init'],s.get('startup','internal'),
            meta.get('task_assets_sha256'),meta.get('tensorflow'),meta.get('numpy'))


def paired_records(folders, names=None):
    groups={}
    for folder in folders:
        for path in sorted((Path(folder)/'results/trajectories').glob('*.json')):
            r=json.loads(path.read_text(encoding='utf-8'))
            if r['status']!='completed' or r['spec']['benchmark']!='olympus':continue
            s=r['spec']
            if names is not None and s['variant'] not in names:continue
            validate_record(r)
            key=(s['task'],s['q'],s['seed']);bucket=groups.setdefault(key,{})
            previous=bucket.get(s['variant'])
            if previous is not None:
                # Timing can vary; conflicting scientific observations cannot.
                if any(previous.get(k)!=r.get(k) for k in ('spec','y','batches','optimizer_x','x','minimize')):
                    raise ValueError('Conflicting duplicate completed trajectory: '+str(key))
                continue
            bucket[s['variant']]=r
    for key,bucket in groups.items():
        if len({protocol_key(r) for r in bucket.values()})>1:
            raise ValueError('Incompatible comparison protocols/assets: '+str(key))
    return groups


def scores(record):
    reference=validate_record(record)['normalization']
    values=trajectory_metrics(record['y'],reference['y_best'],reference['y_worst'],
        batch_sizes=[record['batches']],n_init=record['spec']['n_init'])
    return values['mean_regret_auc'],values['final_regret'],float(record.get('elapsed',0))
