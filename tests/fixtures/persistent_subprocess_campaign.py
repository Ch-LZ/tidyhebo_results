"""Persistent toy campaign: native worker state, live limit, clean STOP/resume.

Run --help for separate optimizer/benchmark Python executables. No env install.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from bo_benchmark.controller import Controller
from bo_benchmark.domain import OptimizerSpec, TaskSpec, ProtocolSpec, RunSpec, positive_int
from bo_benchmark.storage import FileStore, atomic_write
from bo_benchmark.subprocess_runtime import ToySubprocessRuntime
from bo_benchmark.transport import WorkerLaunch
from bo_benchmark.workspace import Workspace

WORKER = Path(__file__).resolve().parents[2] / 'src/bo_benchmark/workers/toy_worker.py'


async def campaign(args):
    workspace = Workspace(args.workspace.resolve())
    workspace.prepare()
    control = workspace.results / 'control.json'
    if not control.exists():
        atomic_write(control, json.dumps({'workers': args.workers}).encode('utf-8'))

    class CampaignStore(FileStore):
        saved_batches = 0
        def save_checkpoint(self, checkpoint):
            super().save_checkpoint(checkpoint)
            if checkpoint.restore_mode == 'native':
                self.saved_batches += 1
                if args.stop_after_batches and self.saved_batches >= args.stop_after_batches:
                    workspace.stop_file.touch()

    store = CampaignStore(workspace.results)
    def launcher(role, environment, spec):
        python = args.optimizer_python if role == 'optimizer' else args.benchmark_python
        return WorkerLaunch((python, '-u', str(WORKER), role),
                            env={'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''})
    runtime = ToySubprocessRuntime(launcher, checkpoint_root=workspace.results / 'native')
    specs = [RunSpec(f'toy_{i}', OptimizerSpec('random', 'toy_optimizer'),
                     TaskSpec('quadratic', 'toy_benchmark', {'noise_std': args.noise_std}),
                     ProtocolSpec(args.budget, args.q), args.seed + i) for i in range(args.runs)]
    results = await Controller(runtime, store, workspace, args.workers,
                               control_file=control).run(specs)
    summary = {'controller_pid': os.getpid(),
               'workers': [channel.ready for channel in runtime.channels],
               'new_batches': store.saved_batches,
               'runs': {name: {'status': outcome.status, 'error': outcome.error,
                              'evaluations': outcome.evaluations, 'direction': outcome.direction,
                              'batches': [{'request_id': b.candidates.request_id,
                                           'X': b.candidates.points, 'y': b.values}
                                          for b in outcome.batches]}
                        for name, outcome in sorted(results.items())}}
    encoded = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
    atomic_write(workspace.results / 'subprocess_summary.json', encoded.encode('utf-8'))
    print(encoded)
    return int(any(outcome.status == 'failed' for outcome in results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, help='Active runs, up to 2 subprocesses each')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--budget', type=int, default=9)
    parser.add_argument('--q', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-std', type=float, default=0.)
    parser.add_argument('--stop-after-batches', type=int)
    parser.add_argument('--optimizer-python', default=sys.executable)
    parser.add_argument('--benchmark-python', default=sys.executable)
    args = parser.parse_args()
    for value in (args.workers, args.runs, args.budget, args.q):
        positive_int(value)
    if args.stop_after_batches is not None:
        positive_int(args.stop_after_batches)
    import math
    if not math.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error('--noise-std must be finite and nonnegative')
    return asyncio.run(campaign(args))


if __name__ == '__main__':
    raise SystemExit(main())
