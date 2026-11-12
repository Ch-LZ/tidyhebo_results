"""Small native-state fixture for failed/pending/paused recovery checks."""
import json
import sys
from bo_benchmark.campaign import plan,campaign_identity
from bo_benchmark.domain import RunOutcome,CandidateBatch,EvaluationBatch
from bo_benchmark.storage import FileStore
from bo_benchmark.workspace import Workspace

def fixture(root,failed=()):
    c={'benchmark':'olympus','tasks':['colors_bob'],'q':[1,2,4],'repeats':2,
       'n_init':2,'n_trials':6,'workers':2,'output_dir':str(root/'workspace'),
       'environments':{'model':{'python':sys.executable},'benchmark':{'python':sys.executable}},
       'models':[{'name':n,'kind':'random','options':{}} for n in
           ['yj_qlogei','qmace_retained','point_mace_adaptive_v031']]}
    path=root/'campaign.json';path.write_text(json.dumps(c))
    planned=plan(path);ws=Workspace(planned['workspace']);ws.prepare()
    (ws.results/'trajectories').mkdir()
    statuses=[];store=FileStore(ws.results)
    for i,s in enumerate(planned['specs']):
        label=planned['labels'][s.run_id]
        q=label['q'];batches=[];n=0
        while n<6:
            size=min(q,6-n,2-n if n<2 else 6-n)
            batches.append(size);n+=size
        state='failed' if i in failed else 'completed'
        error='Interrupted in-flight batch; outcome unknown, automatic replay is disabled' if state=='failed' else None
        r={'run_id':s.run_id,'spec':label,'status':state,'error':error,'minimize':True,
            'y':[.9,.8,.7,.6,.5,.4],'batches':batches,'elapsed':7.+i,
            'x':[[[.1,.2]]], 'diagnostics':[]}
        (ws.results/'trajectories'/(s.run_id+'.json')).write_text(json.dumps(r))
        statuses.append({'run_id':s.run_id,'status':state,'error':error})
        batch=EvaluationBatch(CandidateBatch(s.run_id,tuple((.1,.2) for _ in range(6))),tuple(r['y']))
        store.save_outcome(RunOutcome(s,state,(batch,),error=error,direction='minimize'))
    (ws.results/'campaign.json').write_text(json.dumps(campaign_identity(planned)))
    (ws.results/'summary.json').write_text(json.dumps({'runs':statuses}))
    (ws.results/'native/optimizer').mkdir(parents=True)
    (ws.results/'native/optimizer/shared.pkl').write_bytes(b'preserve shared native')
    return path,planned,ws

