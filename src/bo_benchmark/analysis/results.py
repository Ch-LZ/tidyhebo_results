"""Plot neutral campaign exports, retaining seeds, feedback rounds and physical units."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import best_so_far, incumbent_utility, calculate_metrics, curve_quartiles, needle_statistics


def load_runs(results, benchmark=None, complete_only=True):
    paths = sorted((Path(results)/'trajectories').glob('*.json'))
    if not paths: raise FileNotFoundError('No trajectory exports in '+str(Path(results)/'trajectories'))
    records = [json.loads(path.read_text()) for path in paths]
    records = [r for r in records if benchmark is None or r['spec']['benchmark']==benchmark]
    expected={}
    for record in records:
        label=record['spec'];key=(label['benchmark'],label['task'],label['q'])
        expected.setdefault(key,set()).add(label['variant'])
    for record in records:
        label=record['spec'];key=(label['benchmark'],label['task'],label['q'])
        record['_expected_variants']=sorted(expected[key])
    if complete_only: records = [r for r in records if r['status']=='completed']
    seen = set()
    for record in records:
        label = record['spec']; y = np.asarray(record['y'],dtype=float)
        if y.ndim!=1 or not np.isfinite(y).all() or sum(record['batches'])!=len(y):
            raise ValueError('Invalid raw trajectory: '+record['run_id'])
        if record['status']=='completed' and len(y)!=label['n_trials']:
            raise ValueError('Incomplete run marked completed')
        key = (label['benchmark'],label['task'],label['q'],label['variant'],label['seed'])
        if key in seen: raise ValueError('Duplicate seed within one comparison group')
        seen.add(key)
    return records


def metadata(record):
    return record.get('provenance',{}).get('benchmark',{}).get('identity',{}).get('adapter',{})


def best_curve(record, respect_batches=True, historical_initial=True):
    best = best_so_far(record['y'], minimize=record['minimize'],
        batch_sizes=record['batches'] if respect_batches else None,
        n_init=record['spec']['n_init'] if historical_initial else 0,
        hide_initial=True)
    if record['spec']['benchmark']=='niah':
        meta = metadata(record)
        best = meta.get('display_sign',1.)*best/meta.get('display_scale',1.)
    return best


def group_runs(records):
    groups = {}
    for record in records:
        if record['status']!='completed': continue
        label = record['spec']; key=(label['benchmark'],label['task'],label['q'],label['variant'])
        groups.setdefault(key,[]).append(record)
    for runs in groups.values():
        runs.sort(key=lambda r:r['spec']['seed'])
        if len({len(r['y']) for r in runs})!=1: raise ValueError('Mixed budgets in one comparison')
    return groups


def paired_runs(records):
    """Compare variants only on the intersection of completed seeds per task/q."""
    groups = group_runs(records); output=[]
    for benchmark,task,q in sorted({key[:3] for key in groups}):
        variants = {key[3]:runs for key,runs in groups.items() if key[:3]==(benchmark,task,q)}
        expected=set(next(iter(variants.values()))[0].get('_expected_variants',variants))
        if set(variants)!=expected:continue
        common = set.intersection(*[{r['spec']['seed'] for r in runs} for runs in variants.values()])
        for runs in variants.values():output.extend(r for r in runs if r['spec']['seed'] in common)
    return output


def plot_best(records, output_dir=None, paired=True):
    import matplotlib.pyplot as plt
    if paired: records=paired_runs(records)
    groups=group_runs(records); figures=[]
    labels={'poisson_ratio':"Poisson's ratio",'thermoelectric':'Thermoelectric ZT','wildfire':'Wildfire risk'}
    for benchmark,task,q in sorted({key[:3] for key in groups}):
        fig,ax=plt.subplots(figsize=(6.1,4.3))
        subset={key[3]:runs for key,runs in groups.items() if key[:3]==(benchmark,task,q)}
        for variant,runs in sorted(subset.items()):
            curves=np.vstack([best_curve(r) for r in runs]);lower,median,upper=curve_quartiles(curves)
            x=np.arange(1,len(median)+1)
            line,=ax.plot(x,median,label=f'{variant} (n={len(runs)})')
            ax.fill_between(x,lower,upper,alpha=.2,color=line.get_color())
        if benchmark=='niah':
            for threshold in metadata(next(iter(subset.values()))[0]).get('needle_levels',[]):
                ax.axhline(threshold,color='black',linestyle='--',linewidth=1)
        ax.set(xlabel='Function evaluations',ylabel=labels.get(task,'Best raw objective'),title=f'{task}, q={q}')
        ax.legend();fig.tight_layout();figures.append(fig)
        if output_dir is not None:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            for extension in ('pdf','png'):fig.savefig(folder/f'{benchmark}-{task}-q{q}.{extension}',dpi=160)
    return figures


def robustness_table(records):
    rows=[]
    catalog=json.loads((Path(__file__).parent.parent/'assets/olympus_tasks.json').read_text())
    for (benchmark,task,q,variant),runs in group_runs(paired_runs(records)).items():
        meta=metadata(runs[0]);reference=meta.get('normalization')
        if benchmark=='olympus':
            from .comparison import validate_record
            for record in runs:validate_record(record)
            reference=validate_record(runs[0])['normalization']
        if reference is None:continue  # Synthetic needs an explicit worst reference; do not invent one.
        raw=np.vstack([r['y'] for r in runs])
        metrics=calculate_metrics(raw,reference['y_best'],reference['y_worst'],
             batch_sizes=[r['batches'] for r in runs],sequential_initial_evaluations=runs[0]['spec']['n_init'])
        rows.append({'benchmark':benchmark,'task':task,'q':q,'variant':variant,'n_runs':len(runs),**metrics})
    return pd.DataFrame(rows)


def needle_table(records):
    rows=[]
    for (_,task,q,variant),runs in group_runs(paired_runs(records)).items():
        if runs[0]['spec']['benchmark']!='niah':continue
        meta=metadata(runs[0]);curves=np.vstack([best_curve(r) for r in runs])
        for threshold in meta.get('needle_levels',[]):
            stats=needle_statistics(curves,threshold,higher=meta.get('display_sign',1)<0,
                                    inclusive=task!='wildfire')
            rows.append({'task':task,'q':q,'variant':variant,'threshold':threshold,'n_runs':len(runs),**stats})
    return pd.DataFrame(rows)


def coverage_table(records):
    return pd.DataFrame([{'task':r['spec']['task'],'q':r['spec']['q'],'variant':r['spec']['variant'],
                          'repeat':r['spec']['repeat'],'seed':r['spec']['seed'],'status':r['status'],
                          'evaluations':len(r['y'])} for r in records])
