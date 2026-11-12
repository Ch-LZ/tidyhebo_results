"""Standalone campaign CLI: plan, run/resume, stop, limit, export, doctor."""
import argparse
import asyncio
import csv
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import sys
import time

from .campaign import plan, campaign_identity
from .controller import Controller
from .storage import FileStore, atomic_write
from .workspace import Workspace
from .scientific_runtime import ScientificRuntime


def write_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode('utf-8'))


def prepare(planned):
    workspace = Workspace(planned['workspace']);workspace.prepare()
    with workspace.controller_lease():
        manifest = workspace.results/'campaign.json'
        current = campaign_identity(planned)
        if manifest.exists():
            old = json.loads(manifest.read_text())
            n = min(len(old['seeds']),len(current['seeds']))
            if old['seeds'][:n] != current['seeds'][:n]: raise ValueError('Campaign seed schedule changed')
            for name, spec in current['run_specs'].items():
                if name in old['run_specs'] and spec != old['run_specs'][name]:
                    raise ValueError('Existing run has different scientific settings: '+name)
            old['run_specs'].update(current['run_specs'])
            if len(current['seeds'])>len(old['seeds']):old['seeds']=current['seeds']
            old['seed_manager']={**current['seed_manager'],'seeds':old['seeds']}
            current=old
        write_json(manifest,current)
        write_json(workspace.results/'seeds.json',current['seed_manager'])
        control=workspace.results/'control.json'
        if not control.exists():write_json(control,{'workers':planned['workers']})
    return workspace


class Progress:
    def __init__(self, planned, store, quiet):
        self.planned=planned;self.counts={};self.status={};self.step=None
        for spec in planned['specs']:
            old=store.outcome(spec.run_id);cp=store.checkpoint(spec.run_id)
            self.counts[spec.run_id]=max(old.evaluations if old else 0,sum(len(b.values) for b in cp.batches) if cp else 0)
            self.status[spec.run_id]=old.status if old else 'pending'
        self.bars=None
        self._portion_repeat=None
        if not quiet:
            from tqdm import tqdm
            self.bars=[tqdm(total=sum(s.protocol.budget for s in planned['specs']),desc='All',unit='eval',dynamic_ncols=True,initial=sum(self.counts.values())),
                       tqdm(total=1,desc='Repeat',unit='eval',dynamic_ncols=True)]
            self.refresh()

    def refresh(self):
        if not self.bars:return
        allbar, portion=self.bars
        count=sum(self.counts.values())
        if count>allbar.n:allbar.update(count-allbar.n)
        repeats=sorted(set(v['repeat'] for v in self.planned['labels'].values()))
        active=next((r for r in repeats if any(self.status[k]!='completed' for k,v in self.planned['labels'].items() if v['repeat']==r)),repeats[-1])
        names=[k for k,v in self.planned['labels'].items() if v['repeat']==active]
        total=sum(next(s.protocol.budget for s in self.planned['specs'] if s.run_id==k) for k in names)
        count=sum(self.counts[k] for k in names)
        if active!=self._portion_repeat:
            portion.reset(total=total)
            portion.initial=count;portion.n=count
            self._portion_repeat=active
        elif count>portion.n:portion.update(count-portion.n)
        portion.set_description(f'Repeat {active+1}')
        done=sum(s=='completed' for s in self.status.values())
        suffix=f'runs={done}/{len(self.status)}'
        if self.step is not None:suffix+=f', step={self.step:.2f}s'
        allbar.set_postfix_str(suffix,refresh=False);allbar.refresh();portion.refresh()

    async def monitor(self, finished):
        while not finished.is_set():
            self.refresh()
            try:await asyncio.wait_for(finished.wait(),2)
            except asyncio.TimeoutError:pass

    def close(self):
        self.refresh()
        for bar in self.bars or []:bar.close()


def export(planned, store, workspace, runtime=None):
    rows=[];records=[]
    for spec in planned['specs']:
        outcome=store.outcome(spec.run_id);checkpoint=store.checkpoint(spec.run_id)
        batches=checkpoint.batches if checkpoint else (outcome.batches if outcome else ())
        status=outcome.status if outcome else ('paused' if checkpoint else 'pending')
        label=planned['labels'][spec.run_id]
        direction=checkpoint.direction if checkpoint else (outcome.direction if outcome else None)
        path=workspace.results/'trajectories'/f'{spec.run_id}.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        provenance=(runtime.provenance(spec) if runtime else None) or previous.get('provenance',{})
        y=[value for batch in batches for value in batch.values]
        record={'schema_version':1,'run_id':spec.run_id,'spec':label,'status':status,
                'error':outcome.error if outcome else None,'minimize':direction=='minimize',
                'x':[b.physical_points or b.candidates.points for b in batches],
                'optimizer_x':[b.candidates.points for b in batches], 'y':y,
                'batches':[len(b.values) for b in batches],
                'timings':[b.seconds for b in batches],
                'af_timings':[b.diagnostics.get('af_seconds') for b in batches],
                'warp_active':[b.diagnostics.get('warp_active') for b in batches],
                'native_repairs':[b.diagnostics.get('native_repair') for b in batches],
                'ga_budgets':[b.diagnostics.get('ga_budget') for b in batches],
                'diagnostics':[dict(b.diagnostics) for b in batches],
                'candidate_ids':[b.diagnostics.get('candidate_id') for b in batches],
                'elapsed':sum(b.seconds or 0 for b in batches),
                'elapsed_kind':'sum_batch_wall_seconds_including_native_save',
                'provenance':provenance,'in_flight':checkpoint.in_flight if checkpoint else False}
        write_json(path,record);records.append(record)
        if status=='completed':
            best=[]
            for value in y:best.append(value if not best else (min(best[-1],value) if record['minimize'] else max(best[-1],value)))
            rows.append({**{k:label[k] for k in ('task','variant','q','repeat','seed')},
                         'best':best[-1],'mean_best':sum(best)/len(best),
                         'seconds':record['elapsed'],'rank':None})
    expected={}
    for label in planned['labels'].values():expected.setdefault((label['task'],label['q'],label['repeat']),set()).add(label['variant'])
    for key,variants in expected.items():
        group=[row for row in rows if (row['task'],row['q'],row['repeat'])==key]
        if {row['variant'] for row in group}!=variants:continue
        minimize=next(r['minimize'] for r in records if r['spec']['task']==key[0])
        values=sorted((row['best'] if minimize else -row['best']) for row in group)
        for row in group:
            value=row['best'] if minimize else -row['best']
            row['rank']=sum(i+1 for i,x in enumerate(values) if x==value)/values.count(value)
    stream=StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['task','variant','q','repeat','seed','best','mean_best','seconds','rank']);writer.writeheader();writer.writerows(rows)
    atomic_write(workspace.results/'summary.csv',stream.getvalue().encode('utf-8'))
    write_json(workspace.results/'summary.json',{'runs':[{k:r[k] for k in ('run_id','status','error')} for r in records],
                'completed':len(rows),'total':len(records),'comparisons':rows})
    return records


async def execute(args):
    planned=plan(args.config,args.workspace)
    if args.action=='plan':
        print(json.dumps({'workspace':str(planned['workspace']),'runs':[asdict(s) for s in planned['specs']],
                          'environments':planned['environments']},ensure_ascii=False,indent=2));return 0
    workspace=prepare(planned)
    store=FileStore(workspace.results)
    if args.action=='export':
        with workspace.controller_lease():export(planned,store,workspace)
        return 0
    runtime=ScientificRuntime(workspace,planned['environments'])
    if args.action=='doctor':
        with workspace.controller_lease():
            # Distinct combinations, no suggest/evaluate/observe and no run records.
            seen=set()
            for spec in planned['specs']:
                key=json.dumps([asdict(spec.optimizer),asdict(spec.task)],sort_keys=True)
                if key in seen:continue
                seen.add(key)
                benchmark=optimizer=None
                try:
                    benchmark=await runtime.open_benchmark(spec.task,Path('.'),spec.seed)
                    description=await benchmark.describe()
                    optimizer=await runtime.open_optimizer(spec.optimizer,Path('.'),description,spec.seed)
                    print(f'OK {spec.task.name}/{planned["labels"][spec.run_id]["variant"]}',flush=True)
                finally:
                    for client in (optimizer,benchmark):
                        if client is not None:await client.close()
        return 0
    if args.action=='resume':
        with workspace.controller_lease():workspace.stop_file.unlink(missing_ok=True)
    progress=Progress(planned,store,args.quiet)
    class CampaignStore(FileStore):
        new_batches=0
        def save_checkpoint(inner,cp):
            super().save_checkpoint(cp)
            progress.counts[cp.spec.run_id]=sum(len(b.values) for b in cp.batches)
            if cp.restore_mode=='native':
                inner.new_batches+=1
                if cp.batches:progress.step=cp.batches[-1].seconds
                if args.stop_after_batches and inner.new_batches>=args.stop_after_batches:workspace.stop_file.touch()
        def save_outcome(inner,outcome):
            super().save_outcome(outcome);progress.status[outcome.spec.run_id]=outcome.status
            export(planned,inner,workspace,runtime)
    store=CampaignStore(workspace.results)
    finished=asyncio.Event();monitor=asyncio.create_task(progress.monitor(finished))
    stop=asyncio.Event()
    async def soft_deadline():
        await asyncio.sleep(args.max_seconds)
        print('Time budget reached; saving at the next batch boundary.',flush=True)
        stop.set()
    timer=asyncio.create_task(soft_deadline()) if getattr(args,'max_seconds',None) else None
    try:
        outcomes=await Controller(runtime,store,workspace,planned['workers'],control_file=workspace.results/'control.json').run(planned['specs'],stop=stop)
    finally:
        if timer:
            timer.cancel()
            await asyncio.gather(timer,return_exceptions=True)
        finished.set();await monitor;progress.close()
    with workspace.controller_lease():records=export(planned,store,workspace,runtime)
    failed=[r for r in records if r['status']=='failed']
    for record in failed:print(record['run_id']+': '+str(record['error']),file=sys.stderr)
    print(f'{len(records)-len(failed)} nonfailed / {len(records)} runs; results: {workspace.results}',flush=True)
    return int(bool(failed))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('plan','run','resume','doctor','export'):
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True);p.add_argument('--workspace',type=Path)
        if name in ('run','resume'):
            p.add_argument('--quiet',action='store_true');p.add_argument('--stop-after-batches',type=int)
            p.add_argument('--max-seconds',type=float,help='Soft pause after a completed batch, never kill an evaluation')
    for name in ('stop','limit'):
        p=sub.add_parser(name);p.add_argument('--workspace',type=Path,required=True)
        if name=='limit':p.add_argument('--workers',type=int,required=True)
    args=parser.parse_args(argv)
    try:
        if args.action in ('stop','limit'):
            workspace=Workspace(args.workspace.resolve());workspace.prepare()
            if args.action=='stop':workspace.stop_file.touch()
            else:
                from .domain import positive_int
                positive_int(args.workers);write_json(workspace.results/'control.json',{'workers':args.workers})
            return 0
        if getattr(args,'stop_after_batches',None) is not None:
            from .domain import positive_int
            positive_int(args.stop_after_batches)
        if getattr(args,'max_seconds',None) is not None:
            import math
            if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
                raise ValueError('max-seconds must be finite and positive')
        return asyncio.run(execute(args))
    except KeyboardInterrupt:return 130
    except Exception as exc:
        parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
