"""Independent qmace_lab simplex loop: original initial frame, actual BO feedback."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor/tidyhebo'))
sys.path.insert(0,str(ROOT/'src/bo_benchmark/workers'))
from modern_adapters import core_config
from tidyhebo import tidyHEBO,RealP
from tidyhebo.design_space import MixedDesignSpace

legacy=ROOT/'old_version/tidyhebo_results/reference_optimizers'
sys.path.insert(0,str(legacy if legacy.exists() else ROOT/'src/bo_benchmark/workers'))
if legacy.exists():from simplex_transformer import SimplexTransformer
else:from simplex import SimplexTransformer
request=json.loads(Path(sys.argv[1]).read_text());options=request['options'];seed=request['seed']
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
params=[RealP(i,-2.,2.,name=name) for i,name in enumerate(request['columns'])]
space=MixedDesignSpace(params);mapper=SimplexTransformer(len(params)+1)
initial=torch.quasirandom.SobolEngine(len(params),scramble=True,seed=seed).draw(4,dtype=torch.double)
frame=space.from_GP_to_DS(initial,return_df=True)
def evaluate(physical):return (physical@np.arange(1,physical.shape[1]+1)).reshape(-1,1)
physical=mapper.transform(frame.to_numpy());y=evaluate(physical)
core=tidyHEBO(params,train_x=frame,train_y=-y,config=core_config(options,1,seed))
xs=[physical.tolist()];ys=y.ravel().tolist();sizes=[4]
while len(ys)<21:
    q=min(16,21-len(ys));core.config=core_config(options,q,seed)
    frame=core.suggest();physical=mapper.transform(frame.to_numpy());y=evaluate(physical)
    inverse=pd.DataFrame(mapper.inverse_transform(physical),columns=[p.name for p in params])
    core.observe(inverse,-y)
    xs.append(physical.tolist());ys.extend(y.ravel().tolist());sizes.append(q)
Path(sys.argv[2]).write_text(json.dumps({'x':xs,'y':ys,'batches':sizes}))
