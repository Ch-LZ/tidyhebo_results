"""File-backed mock campaign: live workers config, STOP and native resume.

python examples/persistent_mock_campaign.py --workspace /path/to/workspace --workers 2
Edit <workspace>/results/control.json (workers). Touch results/STOP to pause;
remove it and rerun the same command to continue. No production subprocess Runtime.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from bo_benchmark.controller import Controller
from bo_benchmark.domain import OptimizerSpec, ProtocolSpec, RunSpec, TaskSpec
from bo_benchmark.mocks import MockRuntime
from bo_benchmark.storage import FileStore, atomic_write
from bo_benchmark.workspace import Workspace


async def campaign(args):
    workspace = Workspace(args.workspace)
    workspace.prepare()
    control_file = workspace.results/'control.json'
    if not control_file.exists():
        atomic_write(control_file, json.dumps({'workers':args.workers}).encode())
    batches = 0

    def on_observe(X, y):
        nonlocal batches
        batches += 1
        if args.stop_after_batches is not None and batches >= args.stop_after_batches:
            workspace.stop_file.touch()

    runtime = MockRuntime(on_observe=on_observe)
    store = FileStore(workspace.results)
    specs = [RunSpec(f'mock_{i}', OptimizerSpec('random','mock_optimizer'),
                     TaskSpec('quadratic','mock_benchmark'), ProtocolSpec(args.budget,args.q), i)
             for i in range(args.runs)]
    outcomes = await Controller(runtime,store,workspace,args.workers,
                                control_file=control_file).run(specs)
    summary = dict(pid=os.getpid(), native_restores=sum(e.action=='restore' for e in runtime.events),
                   new_batches=sum(e.action=='evaluate_started' for e in runtime.events),
                   peak_mock_evaluations=runtime.peak_evaluations,
                   runs={key:dict(status=r.status,evaluations=r.evaluations,error=r.error,
                                  direction=r.direction,
                                  batches=[dict(request_id=b.candidates.request_id,
                                                X=b.candidates.points,y=b.values) for b in r.batches])
                         for key,r in outcomes.items()})
    atomic_write(workspace.results/'mock_summary.json',
                 (json.dumps(summary,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(summary,indent=2,allow_nan=False))
    return int(any(r.status=='failed' for r in outcomes.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=ROOT)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--budget',type=int,default=9)
    parser.add_argument('--q',type=int,default=4)
    parser.add_argument('--stop-after-batches',type=int)
    args = parser.parse_args()
    if min(args.workers,args.runs,args.budget,args.q)<1 or (args.stop_after_batches is not None and args.stop_after_batches<1):
        parser.error('Counts must be positive')
    return asyncio.run(campaign(args))


if __name__=='__main__':
    raise SystemExit(main())
