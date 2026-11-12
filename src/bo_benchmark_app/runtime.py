"""Controller export and subprocess hooks for the frozen bo_benchmark executor.

Kept intact to preserve tested native STOP/resume behavior. install() is called
once by the application, never by scientific workers. No second BO loop.
"""
from bo_benchmark.analysis.metrics import runtime_metric_rows
import asyncio
from collections import Counter,defaultdict
import csv
from io import StringIO
import json
import os
import subprocess
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bo_benchmark import cli,storage,environments,mamba_support
from bo_benchmark.transport import JsonLinesChannel

STATS=defaultdict(float)
PENDING=Counter()
_STARTED=time.perf_counter()



ACTIVATION_KEYS={'PATH','CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_SHLVL',
    'CONDA_DLL_SEARCH_MODIFICATION_ENABLE','CONDA_EXE','CONDA_PYTHON_EXE',
    'PYTHONNOUSERSITE','SSL_CERT_FILE','REQUESTS_CA_BUNDLE','PROJ_LIB','GDAL_DATA'}


def direct_environments(definitions):
    """Ask Mamba for activation once per prefix, then use its Python directly."""
    import copy
    result=copy.deepcopy(definitions);activated={}
    for definition in result.values():
        prefix=definition.get('mamba_prefix')
        if not prefix:continue
        key=(str(prefix),definition['python'])
        if key not in activated:
            code="import json,os; print('BO_ACTIVATION033='+json.dumps({k:v for k,v in os.environ.items() if k in "+repr(sorted(ACTIVATION_KEYS))+"}))"
            command=mamba_support.run_command(mamba_support.find_mamba(),Path(prefix),[definition['python'],'-c',code])
            failure=None
            for _ in range(3):
                STATS['activation_launches']+=1;started=time.perf_counter()
                completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
                STATS['activation_seconds']+=time.perf_counter()-started
                lines=[line for line in completed.stdout.splitlines() if line.startswith('BO_ACTIVATION033=')]
                if completed.returncode==0 and lines:
                    values=json.loads(lines[-1].split('=',1)[1])
                    activated[key]={k:v for k,v in values.items() if k in ACTIVATION_KEYS}
                    break
                failure=completed.stderr[-2000:]
            else:raise RuntimeError('Mamba activation failed before launching workers: '+str(failure))
        definition['env']={**activated[key],**definition.get('env',{})}
        definition.pop('mamba_prefix')
    return result


def build_record(spec,planned,store,workspace,runtime,previous):
    outcome=store.outcome(spec.run_id);checkpoint=store.checkpoint(spec.run_id)
    batches=checkpoint.batches if checkpoint else (outcome.batches if outcome else ())
    status=outcome.status if outcome else ('paused' if checkpoint else 'pending')
    direction=checkpoint.direction if checkpoint else (outcome.direction if outcome else None)
    provenance=(runtime.provenance(spec) if runtime else None) or previous.get('provenance',{})
    y=[value for batch in batches for value in batch.values]
    return {'schema_version':1,'run_id':spec.run_id,'spec':planned['labels'][spec.run_id],'status':status,
        'error':outcome.error if outcome else None,'minimize':direction=='minimize',
        'x':[b.physical_points or b.candidates.points for b in batches],
        'optimizer_x':[b.candidates.points for b in batches],'y':y,
        'batches':[len(b.values) for b in batches],'timings':[b.seconds for b in batches],
        'af_timings':[b.diagnostics.get('af_seconds') for b in batches],
        'warp_active':[b.diagnostics.get('warp_active') for b in batches],
        'native_repairs':[b.diagnostics.get('native_repair') for b in batches],
        'ga_budgets':[b.diagnostics.get('ga_budget') for b in batches],
        'diagnostics':[dict(b.diagnostics) for b in batches],
        'candidate_ids':[b.diagnostics.get('candidate_id') for b in batches],
        'elapsed':sum(b.seconds or 0 for b in batches),
        'elapsed_kind':'sum_batch_wall_seconds_including_native_save',
        'provenance':provenance,'in_flight':checkpoint.in_flight if checkpoint else False}


def summaries(records,planned):
    rows=runtime_metric_rows(records)
    stream=StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['task','variant','q','repeat','seed','best','mean_best','seconds','rank'])
    writer.writeheader();writer.writerows(rows)
    summary={'runs':[{k:r[k] for k in ('run_id','status','error')} for r in records],
             'completed':len(rows),'total':len(records),'comparisons':rows}
    return stream.getvalue().encode('utf-8'),summary


def fast_export(planned,store,workspace,runtime=None):
    started=time.perf_counter();STATS['export_calls']+=1
    try:
        callback=store.__dict__.pop('_export_outcome_phase033',False)
        records=store.__dict__.setdefault('_export_records033',{})
        first=not records
        # Refresh only the terminal event during a run. Final/explicit export
        # refreshes all checkpoint tails, including paused and pending records.
        changed=store.__dict__.pop('_export_dirty033',set())
        if first or not callback:changed={s.run_id for s in planned['specs']}
        modified=first
        for spec in planned['specs']:
            if spec.run_id not in changed:continue
            path=workspace.results/'trajectories'/f'{spec.run_id}.json'
            previous=records.get(spec.run_id)
            if previous is None:previous=json.loads(path.read_text()) if path.exists() else {}
            current=build_record(spec,planned,store,workspace,runtime,previous)
            if current!=previous or not path.exists():
                cli.write_json(path,current);STATS['trajectory_json_writes']+=1;modified=True
            records[spec.run_id]=current
        ordered=[records[s.run_id] for s in planned['specs']]
        # REPORT_BATCH_30S: trajectories stay current; batch aggregate reports.
        report_dirty = store.__dict__.setdefault('_report_dirty033', set())
        report_dirty.update(changed)
        now = time.perf_counter()
        if (callback and not first
                and now - store.__dict__.get('_report_last033', float('-inf')) < 30):
            store.__dict__['_report_pending033'] = True
            STATS['report_exports_deferred'] += 1
            return ordered
        modified = modified or store.__dict__.get('_report_pending033', False)
        if any(v.get('env',{}).get('BO_BASELINE') for v in planned['environments'].values()):
            from .trajectory_csv import export
            export(ordered,workspace,report_dirty if callback and not first else None)
        if modified or not (workspace.results/'summary.json').exists():
            csv_bytes,summary=summaries(ordered,planned)
            cli.atomic_write(workspace.results/'summary.csv',csv_bytes)
            cli.write_json(workspace.results/'summary.json',summary)
            STATS['summary_writes']+=1
        store.__dict__['_report_last033'] = time.perf_counter()
        store.__dict__['_report_pending033'] = False
        report_dirty.clear()
        return ordered
    finally:
        elapsed=time.perf_counter()-started
        STATS['export_seconds']+=elapsed;STATS['export_max_seconds']=max(STATS['export_max_seconds'],elapsed)


def install():
    if getattr(cli,'_runtime_installed',False):return
    cli._runtime_installed=True
    original_outcome=storage.FileStore.save_outcome
    def outcome(store,value):
        original_outcome(store,value)
        store.__dict__.setdefault('_export_dirty033',set()).add(value.spec.run_id)
        store._export_outcome_phase033=True
    storage.FileStore.save_outcome=outcome
    original_write=storage.FileStore._write
    def write(store,value):
        started=time.perf_counter()
        try:return original_write(store,value)
        finally:
            elapsed=time.perf_counter()-started;STATS['state_write_seconds']+=elapsed
            STATS['state_write_calls']+=1;STATS['state_write_max_seconds']=max(STATS['state_write_max_seconds'],elapsed)
    storage.FileStore._write=write
    cli.export=fast_export
    original_command=mamba_support.run_command
    commands={}
    def command(mamba,prefix,argv,*,env=None):
        STATS['mamba_command_calls']+=1
        if env is not None:return original_command(mamba,prefix,argv,env=env)
        key=(str(mamba),str(prefix),tuple(map(str,argv)))
        if key not in commands:
            started=time.perf_counter()
            commands[key]=tuple(original_command(mamba,prefix,argv))
            STATS['mamba_help_seconds']+=time.perf_counter()-started;STATS['mamba_help_calls']+=1
        return list(commands[key])
    mamba_support.run_command=command;environments.run_command=command
    original_request=JsonLinesChannel.request
    async def request(channel,command,payload=None):
        key=str(channel.ready.get('role','worker'))+'.'+command
        PENDING[key]+=1
        try:return await original_request(channel,command,payload)
        finally:PENDING[key]-=1
    JsonLinesChannel.request=request
    original_execute=cli.execute
    async def execute(args):
        if args.action not in ('run','resume'):return await original_execute(args)
        planned=cli.plan(args.config,args.workspace)
        live_config=os.environ.get('BO_EXPERIMENT_CONFIG')
        live_config=Path(live_config) if live_config else None
        live_enabled=bool(live_config and 'target_repeats' in json.loads(live_config.read_text(encoding='utf-8-sig')))
        last_workers=None
        if live_enabled:
            from .baselines import live_workers
            from bo_benchmark.workspace import Workspace
            with Workspace(planned['workspace']).controller_lease():
                last_workers=live_workers(live_config,planned['workspace'])
        target=planned['workspace']/'results/controller_performance.json'
        def save():
            elapsed=time.perf_counter()-_STARTED
            cli.write_json(target,{'elapsed_seconds':elapsed,**dict(STATS),
                'pending_rpc':{k:v for k,v in PENDING.items() if v},
                'controller_blocking_fraction':(STATS['export_seconds']+STATS['state_write_seconds'])/max(elapsed,1e-9),
                'note':'Measured controller overhead only; active JSON tails refreshed at final export. No claim of total BO speedup.'})
        async def monitor():
            nonlocal last_workers
            previous=time.perf_counter()
            while True:
                await asyncio.sleep(5)
                now=time.perf_counter();lag=max(0.,now-previous-5)
                if live_enabled:
                    try:
                        value=json.loads(live_config.read_text(encoding='utf-8-sig'))['workers']
                        if value!=last_workers:
                            last_workers=live_workers(live_config,planned['workspace'])
                            print('Experiment workers: '+str(last_workers),flush=True)
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        print('Ignoring incomplete workers edit: '+str(error),flush=True)
                STATS['event_loop_lag_max_seconds']=max(STATS['event_loop_lag_max_seconds'],lag)
                save();previous=time.perf_counter()
        watcher=asyncio.create_task(monitor())
        original_plan=cli.plan
        restore_live=None
        try:
            planned['environments']=await asyncio.to_thread(direct_environments,planned['environments'])
            # The scientific specs/labels are unchanged; only process activation
            # is performed once before the old execute opens its workers.
            cli.plan=lambda *a,**kw:planned
            if live_enabled:
                from .live_repeats import install as install_live
                restore_live=install_live(live_config,planned)
            return await original_execute(args)
        finally:
            if restore_live:restore_live()
            cli.plan=original_plan
            watcher.cancel();await asyncio.gather(watcher,return_exceptions=True);save()
    cli.execute=execute
