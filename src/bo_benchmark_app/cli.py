"""Small application CLI; scheduling and scientific code stay in the executor."""
import argparse
import json
from pathlib import Path
from .configuration import DEFAULT, resolved


def parser():
    p=argparse.ArgumentParser(description='BO benchmark: run, resume, stop, metrics. Experiments live in experiments/.')
    p.add_argument('action',metavar='{run,resume,stop,metrics,status,retry,watch}',choices=['run','resume','stop','metrics','status','retry','watch','plan','check','doctor','export','workers','limit'])
    p.add_argument('experiment',type=Path,nargs='?',help='experiment JSON (default: OLY-002)')
    p.add_argument('--config',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--workspace',type=Path)
    p.add_argument('--workers',type=int)
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--stop-after-batches',type=int)
    p.add_argument('--max-seconds',type=float)
    p.add_argument('--expect-failed',type=int,default=15)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--apply',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--no-metrics',action='store_true',help='skip automatic Olympus report after execution')
    p.add_argument('--background',action='store_true',help='detach execution and write its console to the workspace')
    p.add_argument('--snapshot',action='store_true',help='analyze atomic published results while execution continues')
    p.add_argument('--notebook',action='store_true',help='open a Jupyter report after metrics')
    p.add_argument('--window',action='store_true',help='open a separate read-only progress console on Windows')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    from .baselines import install
    install()
    if a.config and a.experiment:
        raise ValueError('Choose positional experiment or --config, not both')
    config=a.config or a.experiment or DEFAULT
    with resolved(config,a.workspace) as (path,data):
        if data.get('benchmark') == 'niah':
            from bo_benchmark_setup.niah005.runtime import install as install_niah
            install_niah()
        if data.get('benchmark') == 'bayesmark_full' and a.action not in ('status','metrics','clean'):
            import sys
            from .configuration import ROOT
            sys.path.insert(0,str(ROOT))
            from bayesmark_batch.campaign import execute, plan
            from functools import partial
            args=[{'run':'resume','check':'doctor'}.get(a.action,a.action),'--config',str(config)]
            if a.quiet: args.append('--quiet')
            return execute(args, partial(plan, smoke=data.get('paper_smoke',False)))
        if a.action=='metrics':
            from .metrics import analyze
            out=analyze(path,snapshot=a.snapshot)
            if a.notebook:
                from .notebook import launch
                return launch(out)
            return 0
        if a.action=='status':
            summary=Path(data['output_dir'])/'results/summary.json'
            value=json.loads(summary.read_text()) if summary.exists() else {'status':'not started'}
            if 'runs' in value:
                from collections import Counter
                value={'total':len(value['runs']),'statuses':dict(Counter(r['status'] for r in value['runs']))}
            print(json.dumps(value,indent=2));return 0
        if a.action=='retry':
            from .recovery import archive_failed
            archive_failed(path,expected_failed=a.expect_failed,dry_run=a.dry_run)
            if a.dry_run:return 0
        from . import runtime
        runtime.install()
        action={'check':'doctor','workers':'limit','retry':'resume'}.get(a.action,a.action)
        args=([action,'--workspace',data['output_dir']] if action in ('stop','limit')
              else [action,'--config',str(path)])
        if a.workers is not None:
            if action=='limit':args+=['--workers',str(a.workers)]
            else:
                # Admission limit is runtime-only, never a scientific config change.
                from bo_benchmark.storage import atomic_write
                from bo_benchmark.workspace import Workspace
                if a.workers<1:raise ValueError('workers must be positive')
                ws=Workspace(Path(data['output_dir']))
                with ws.controller_lease():atomic_write(ws.results/'control.json',json.dumps({'workers':a.workers}).encode())
        if a.quiet:args+=['--quiet']
        for name in ('stop_after_batches','max_seconds'):
            value=getattr(a,name)
            if value is not None:args+=['--'+name.replace('_','-'),str(value)]
        code=runtime.cli.main(args)
        return code


if __name__=='__main__':
    raise SystemExit(main())
