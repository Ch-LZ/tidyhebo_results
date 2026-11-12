"""Recompute scores from portable histories in the pinned Bayesmark environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def loss_array(root):
    import numpy as np
    from bo_benchmark_app.paper import history_records
    from bo_benchmark_app.bayesmark_figure import METHODS

    records,_ = history_records('bayesmark',root)
    tasks = sorted({key[0] for key in records})
    seeds = sorted({key[3] for key in records})
    methods = list(METHODS)
    losses = np.empty((16,8,len(tasks),len(methods),len(seeds)))
    for ti,task in enumerate(tasks):
        for mi,method in enumerate(methods):
            for si,seed in enumerate(seeds):
                losses[:,:,ti,mi,si] = np.asarray(records[task,method,8,seed]['y']).reshape(16,8)
    return losses, tasks, seeds


def main(input_path=None):
    import numpy as np
    import pandas as pd
    from bo_benchmark.analysis.metrics import bayesmark_scores, repeat_score_statistics
    from bo_benchmark_app.bayesmark_figure import METHODS

    if input_path is None:
        losses, tasks, seeds = loss_array(ROOT)
    else:
        with np.load(input_path, allow_pickle=False) as source:
            losses = source['losses']
            tasks, seeds = source['tasks'].tolist(), source['seeds'].tolist()
    methods = list(METHODS)
    curves,tasks_scores,repeats = bayesmark_scores(losses,tasks,methods)
    scores,intervals = repeat_score_statistics(repeats)
    reference = ROOT/'data/historical/bayesmark_tables/Bayesmark_q8_leaderboard_by_repeat.csv'
    if reference.is_file():
        expected = pd.read_csv(reference).set_index('seed').loc[seeds,methods].to_numpy()
        if not np.allclose(repeats,expected,rtol=1e-10,atol=1e-10):
            raise ValueError('Recomputed Bayesmark scores differ from the archived paper scores')
    out = ROOT/'analysis/bayesmark'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(repeats,columns=methods).assign(seed=seeds).to_csv(out/'Bayesmark_q8_leaderboard_by_repeat.csv',index=False)
    pd.DataFrame(tasks_scores,columns=methods).assign(task=tasks).to_csv(out/'Bayesmark_q8_leaderboard_by_task.csv',index=False)
    pd.DataFrame(curves,columns=methods).assign(evaluations=np.arange(8,129,8)).to_csv(out/'Bayesmark_q8_leaderboard_curve.csv',index=False)
    pd.DataFrame(dict(method=methods,leaderboard_score=scores,ci_low=intervals[0],ci_high=intervals[1])).to_csv(out/'Bayesmark_q8_leaderboard.csv',index=False)
    print(out)
    return 0


if __name__ == '__main__':
    sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
    raise SystemExit(main(*sys.argv[1:]))
