"""Plot archived leaderboard scores; no learner or oracle environment is needed."""
from pathlib import Path
import math
import hashlib
import json
import numpy as np
import pandas as pd
from bo_benchmark.analysis.metrics import repeat_score_statistics
from bo_benchmark.seeds import SeedManager

METHODS = {
    'hebo': ('HEBO','#9467bd'),
    'qmace_retained': ('tidyHEBO\n(qMACE)','#d62728'),
    'yj_qlogei': ('tidyHEBO\nw/qLogEI','#8c564b'),
    'vanilla_gp_qlogei': ('qLogEI','#ff7f0e'),
    'optuna': ('Optuna','#2ca02c'),
    'random': ('random','#1f77b4'),
}


def render(root):
    import matplotlib.pyplot as plt
    from .paper import PARQUET_FORMAT, history_records

    _, manifest = history_records('bayesmark',root)
    folder = Path(root)/'data/historical/bayesmark_tables'
    table = folder/'Bayesmark_q8_leaderboard_by_repeat.csv'
    if manifest.get('format') == PARQUET_FORMAT:
        provenance = json.loads((folder/'parquet_source.json').read_text(encoding='utf-8'))
        table_bytes = table.read_bytes().replace(b'\r\n', b'\n')
        if (provenance['parquet_sha256'] != manifest['sha256']['data/historical/bayesmark.parquet']
                or provenance['repeat_table_sha256'] != hashlib.sha256(table_bytes).hexdigest()):
            raise ValueError('Bayesmark repeat table does not match the frozen Parquet history')
    frame = pd.read_csv(table)
    seeds = sorted(SeedManager(42,15).seeds)
    if len(frame) != 15 or frame.seed.duplicated().any() or set(frame.seed) != set(seeds):
        raise ValueError('Bayesmark figure requires all 15 matched repeats')
    values = frame.set_index('seed').loc[seeds,list(METHODS)].to_numpy()
    scores, intervals = repeat_score_statistics(values)
    selected = np.argsort(-scores,kind='stable')
    plot_methods = [list(METHODS)[i] for i in selected]
    plot_scores, plot_intervals = scores[selected], intervals[:,selected]
    positions = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(8.2,5.1))
    bars = ax.bar(positions,plot_scores,color=[METHODS[m][1] for m in plot_methods],width=.8)
    ax.vlines(positions,plot_intervals[0],plot_intervals[1],color='#444444',linewidth=2)
    for i,bar in enumerate(bars):
        top = max(plot_scores[i],plot_intervals[1,i])
        ax.annotate(f'{plot_scores[i]:.2f}',(bar.get_x()+bar.get_width()/2,top),
                    xytext=(0,7),textcoords='offset points',ha='center',fontsize=12)
    bottom = min(75,5*math.floor((np.nanmin([plot_scores,plot_intervals[0]])-3)/5))
    top = np.nanmax([plot_scores,plot_intervals[1]])
    ax.set_ylim(bottom,100 if top <= 97 else top+4)
    ax.set_xticks(positions,[METHODS[m][0] for m in plot_methods],fontsize=11)
    ax.set_ylabel('Bayesmark score',fontsize=15)
    ax.set_xlabel('Optimizer',fontsize=14)
    ax.tick_params(axis='y',labelsize=12)
    fig.tight_layout()
    out = Path(root)/'figures/bayesmark'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/('Figure_6_Bayesmark.'+ext),dpi=300)
    pd.DataFrame(dict(method=list(METHODS),score=scores,low=intervals[0],high=intervals[1])).to_csv(
        out/'figure_values.csv',index=False)
    plt.close(fig)
    return out
