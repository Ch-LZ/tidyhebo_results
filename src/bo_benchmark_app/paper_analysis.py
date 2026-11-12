"""Thin I/O around the single shared file of metric definitions."""
from pathlib import Path
import json
from .configuration import ROOT
from .paper import METHODS, NIAH_DISPLAY, history_records


def analyze(name):
    if name == 'bayesmark':
        import os
        import subprocess
        import tempfile
        import numpy as np
        from .bayesmark_score import loss_array
        from bayesmark_batch.full_suite import oracle_env, SOURCE
        from bayesmark_batch.environment import environment_command
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT),str(SOURCE)])
        losses, tasks, seeds = loss_array(ROOT)
        # The older scoring environment does not need a Parquet dependency.
        with tempfile.TemporaryDirectory(prefix='tidyhebo-bayesmark-') as temp:
            input_path = Path(temp)/'losses.npz'
            np.savez(input_path, losses=losses, tasks=tasks, seeds=seeds)
            command = [ROOT/'src/bo_benchmark_app/bayesmark_score.py', input_path]
            subprocess.run(environment_command(oracle_env({'benchmark':'bayesmark_full'}), command),
                           cwd=ROOT,env=env,check=True)
        return ROOT/'analysis/bayesmark'
    import numpy as np
    import pandas as pd
    from bo_benchmark.analysis.metrics import (
        olympus_run_table,olympus_statistics,best_so_far,median_curve,
    )
    from bo_benchmark.seeds import SeedManager
    from bo_benchmark.task_catalog import load_catalog

    records,manifest = history_records(name)
    out = ROOT/'analysis'/name
    out.mkdir(parents=True,exist_ok=True)
    if name == 'olympus':
        runs = olympus_run_table(records,sorted(load_catalog()),METHODS,(1,2,4,8,16),SeedManager(42,30).seeds)
        task_metrics,summary = olympus_statistics(runs,METHODS,(1,2,4,8,16))
        runs.to_csv(out/'run_metrics.csv',index=False)
        task_metrics.to_csv(out/'task_metrics.csv',index=False)
        summary.to_csv(out/'summary.csv',index=False)
    else:
        rows = []
        for task in ('poisson_ratio','thermoelectric'):
            for method in METHODS:
                group = [records[task,method,1,s] for s in SeedManager(42,12).seeds]
                sign, scale = NIAH_DISPLAY[task]
                curves = np.stack([best_so_far(r['y'],minimize=r['minimize']) for r in group])
                curve = median_curve(sign*curves/scale)
                rows.extend(dict(task=task,method=method,evaluation=i+1,median_best=value)
                            for i,value in enumerate(curve))
        pd.DataFrame(rows).to_csv(out/'median_curves.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(history_sha256=manifest['sha256']),indent=2)+'\n')
    print(out)
    return out
