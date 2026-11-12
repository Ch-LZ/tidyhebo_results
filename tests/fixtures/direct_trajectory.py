"""Independent direct-library trajectory; no new adapter/Controller in reference loop."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'vendor/tidyhebo'
sys.path.insert(0,str(source))
sys.path.insert(0,str(ROOT/'src/bo_benchmark/workers'))
from modern_adapters import core_config
from tidyhebo import tidyHEBO,RealP
from tidyhebo.design_space import MixedDesignSpace
from botorch.test_functions.synthetic import Branin

settings=json.loads(Path(sys.argv[1]).read_text())
seed=settings['seed'];q=settings['q'];budget=settings['budget'];options=settings['options']
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
task=Branin().double()
params=[RealP(i,float(a),float(b)) for i,(a,b) in enumerate(task.bounds.T)]
space=MixedDesignSpace(params)
legacy=options['startup']=='legacy_qmace'
core=None if legacy else tidyHEBO(params,n_init=options['n_init'],config=core_config(options,1,seed))
xs=[];ys=[];sizes=[]
while len(ys)<budget:
    n=min(q,budget-len(ys))
    if len(ys)<options['n_init']:
        n=min(options['n_init']-len(ys),budget-len(ys),options['n_init'] if legacy else q)
    if core is None:
        initial=torch.quasirandom.SobolEngine(len(params),scramble=True,seed=seed).draw(options['n_init'],dtype=torch.double)
        frame=space.from_GP_to_DS(initial,return_df=True)
    else:
        core.config=core_config(options,n,seed)
        frame=core.suggest()
    x=frame.to_numpy(dtype=float)
    y=task(torch.tensor(x,dtype=torch.double)).detach().numpy().reshape(-1,1)
    if core is None:core=tidyHEBO(params,train_x=frame,train_y=-y,config=core_config(options,1,seed))
    else:core.observe(frame,-y)
    xs.append(x.tolist());ys.extend(y.ravel().tolist());sizes.append(len(x))
Path(sys.argv[2]).write_text(json.dumps({'x':xs,'y':ys,'batches':sizes}))
