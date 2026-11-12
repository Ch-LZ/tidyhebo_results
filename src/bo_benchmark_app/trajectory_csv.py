"""One shareable CSV per experiment/task/optimizer, including all q and repeats."""
from collections import defaultdict
import csv
import hashlib
from io import StringIO
import json
from bo_benchmark.storage import atomic_write


def export(records, workspace, changed=None):
    groups = defaultdict(list)
    selected = {(r['spec']['task'], r['spec']['variant']) for r in records
                if changed is None or r['run_id'] in changed}
    for record in records:
        key = record['spec']['task'], record['spec']['variant']
        if key in selected:
            groups[key].append(record)
    fields = ['experiment','task','optimizer','q','repeat','seed','status','run_id',
              'evaluation','batch','position_in_batch','objective','best_after_batch',
              'minimize','x','optimizer_x','config_sha256','task_assets_sha256']
    for (task, variant), runs in groups.items():
        stream = StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in sorted(runs, key=lambda r: (r['spec']['q'], r['spec']['repeat'])):
            s = r['spec']
            base = dict(experiment=workspace.root.name, task=task, optimizer=variant,
                **{k:s[k] for k in ('q','repeat','seed')}, status=r['status'], run_id=r['run_id'],
                minimize=r['minimize'], config_sha256=hashlib.sha256(json.dumps(
                    s['config'],sort_keys=True).encode()).hexdigest(), task_assets_sha256=r.get(
                    'provenance',{}).get('benchmark',{}).get('identity',{}).get('adapter',{}).get('task_assets_sha256',''))
            offset = 0
            incumbent = None
            for batch, size in enumerate(r['batches']):
                values = r['y'][offset:offset+size]
                choose = min if r['minimize'] else max
                best = choose(values)
                incumbent = best if incumbent is None else choose(incumbent,best)
                for position, value in enumerate(values):
                    writer.writerow(dict(base, evaluation=offset+position+1, batch=batch,
                        position_in_batch=position, objective=value, best_after_batch=incumbent,
                        x=json.dumps(r['x'][batch][position]),
                        optimizer_x=json.dumps(r['optimizer_x'][batch][position])))
                offset += size
        atomic_write(workspace.root/'trajectories'/f'{task}__{variant}.csv',
                     stream.getvalue().encode('utf-8-sig'))
