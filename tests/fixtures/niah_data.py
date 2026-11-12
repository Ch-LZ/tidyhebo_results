"""Small labelled protocol fixtures, NOT downloaded ZoMBI scientific data."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/bo_benchmark/workers'))
from niah_assets import COMMIT,content_id,export_candidates


def prepared(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(11)
    table=rng.uniform(size=(24,2))
    assets={}
    for task in ('poisson_ratio','thermoelectric','wildfire'):
        meta={'protocol':'niah-asset-v1','task':task,'source_commit':COMMIT,'dim':2,
              'fixture':True,'feature_names':['feature_a','feature_b'],
              'normalization':{'y_best':-3.,'y_worst':1.,'kind':'test-fixture'},
              'display_sign':-1. if task=='thermoelectric' else 1.,
              'display_scale':50. if task=='thermoelectric' else 1.,
              'needle_levels':[1.4] if task=='thermoelectric' else [0.] if task=='wildfire' else [-1.2,-1.7]}
        if task=='wildfire':
            arrays={'x':table,'y':table.sum(axis=1)-1.,'low':np.array([10.,20.]),'span':np.array([2.,3.]),
                    'source_row':np.arange(len(table)),'counts':np.ones(len(table),dtype=np.int64)}
            meta.update(kind='table',rows=len(table))
        else:
            arrays={'roots':np.array([0]),'left':np.array([1,-1,-1]),'right':np.array([2,-1,-1]),
                    'feature':np.array([0,-2,-2],dtype=np.int16),'threshold':np.array([.5,-2,-2]),
                    'value':np.array([0.,-100.,-1.]) if task=='thermoelectric' else np.array([0.,-2.2,.4])}
            meta.update(kind='forest',trees=1)
        meta['asset_id']=content_id(arrays,meta)
        np.savez_compressed(folder/f'{task}.npz',**arrays)
        (folder/f'{task}.json').write_text(json.dumps(meta))
        assets[task]=(arrays,meta)
    export_candidates(folder)
    return assets
