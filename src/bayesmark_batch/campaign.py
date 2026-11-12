"""Plan Bayesmark runs for the audited controller without modifying the runner."""
from dataclasses import replace
from pathlib import Path
import os

from bo_benchmark.domain import OptimizerSpec, ProtocolSpec, RunSpec, TaskSpec, positive_int
from bo_benchmark.seeds import SeedManager
from bo_benchmark_app.configuration import load

from .full_suite import SMOKE as FULL_SMOKE, TASKS as FULL_TASKS, oracle_env

METHODS = ('qmace_retained', 'yj_qlogei', 'vanilla_gp_qlogei', 'hebo', 'optuna', 'random')
BASELINES = set(METHODS[3:])

def plan(path, workspace=None, *, smoke=False):
    data = load(path, workspace)
    if (data['benchmark'] != 'bayesmark_full'
            or (data['q'], data['n_init'], data['n_trials'], data['startup']) != (
                [8], 8, 128, 'legacy_qmace')):
        raise ValueError('Expected 16 batches of 8, with a shared Sobol first batch')
    if data['tasks'] != 'all':
        raise ValueError('Full Bayesmark requires all 108 public tasks')
    tasks = FULL_SMOKE if smoke else tuple(FULL_TASKS)
    if len(data['models']) != len(METHODS) or {m['name'] for m in data['models']} != set(METHODS):
        raise ValueError('Expected all six methods exactly once')
    envs = data['environments']
    optimizers = {}
    for model in data['models']:
        name = model['name']
        env = name if name in BASELINES else 'model'
        tag = name if name in BASELINES else None
        options = model['options']
        if (model.get('environment', 'model') != env or envs[env].get('env', {}).get('BO_BASELINE') != tag
                or model['kind'] != 'tidyhebo' or options.get('n_init') != 8
                or options.get('startup') != data['startup']):
            raise ValueError(f'Incorrect optimizer environment or startup: {name}')
        optimizers[name] = OptimizerSpec(model['kind'], env, options)
    for source in envs['model'].get('sources', []):
        if not Path(source).is_dir():
            raise FileNotFoundError(f'Missing tidyHEBO source: {source}')
    envs['benchmark'] = oracle_env(data)
    positive_int(data['workers'])
    manager = SeedManager(data['master_seed'], 1 if smoke else data['repeats'])
    protocol = ProtocolSpec(16 if smoke else 128, 8, 8, True)
    specs, labels = [], {}
    for repeat, seed in enumerate(manager.seeds):
        for task in tasks:
            benchmark = TaskSpec(task, 'benchmark', {'benchmark': data['benchmark']})
            for name in METHODS:
                optimizer = optimizers[name]
                run_id = f'{task}-q8-{name}-r{repeat:03d}'
                specs.append(RunSpec(run_id, optimizer, benchmark, protocol, seed))
                labels[run_id] = dict(benchmark=data['benchmark'], task=task, variant=name,
                    q=8, repeat=repeat, seed=seed, n_trials=protocol.budget, n_init=8,
                    startup=data['startup'], config=dict(optimizer.options))
    return dict(specs=specs, labels=labels, environments=envs, workspace=Path(data['output_dir']),
                workers=data['workers'], threads=data['threads'], seeds=manager.seeds,
                seed_manager=manager.state_dict(), legacy_config=False)

def execute(argv, planner):
    from bo_benchmark import cli
    from bo_benchmark.transport import JsonLinesChannel
    from bo_benchmark_app import baselines, runtime

    baselines.install()
    if os.name == 'nt':
        import asyncio
        from subprocess import ABOVE_NORMAL_PRIORITY_CLASS

        spawn = asyncio.create_subprocess_exec

        async def spawn_worker(*args, **kwargs):
            if args and args[-1] in ('optimizer', 'benchmark'):
                kwargs['creationflags'] = kwargs.get('creationflags', 0) | ABOVE_NORMAL_PRIORITY_CLASS
            return await spawn(*args, **kwargs)

        asyncio.create_subprocess_exec = spawn_worker

    original = JsonLinesChannel.open

    async def open_worker(cls, launch):
        suite = launch.env.get('BO_BAYESMARK_ORACLE')
        if launch.argv[-1] == 'benchmark' and suite == 'full':
            argv = list(launch.argv)
            index = next(i for i, arg in enumerate(argv) if Path(arg).name == 'scientific_worker.py')
            worker = Path(__file__).with_name('full_worker_recovery.py')
            argv[index] = str(worker)
            launch = replace(launch, argv=tuple(argv))
        return await original(launch)

    JsonLinesChannel.open = classmethod(open_worker)
    cli.plan = planner
    runtime.install()
    os.environ.pop('BO_EXPERIMENT_CONFIG', None)
    return cli.main(argv)
