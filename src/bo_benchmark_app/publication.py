"""One public command; execution continues through the existing controller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .configuration import ROOT
from .paper import CAMPAIGNS, config_path


def smoke_configs(name):
    """One feedback round per q; retain all methods and the real oracle."""
    source = json.loads(config_path(name).read_text(encoding='utf-8-sig'))
    folder = ROOT / 'results/smoke' / name
    folder.mkdir(parents=True,exist_ok=True)
    qs = CAMPAIGNS[name]['q']
    for q in qs:
        data = json.loads(json.dumps(source))
        data['target_repeats'] = 1
        data['q'] = [q]
        data['workers'] = 2
        data['output_dir'] = str(folder / f'q{q}')
        if name == 'bayesmark':
            data['paper_smoke'] = True
        else:
            data['tasks'] = ['hplc','colors_bob'] if name == 'olympus' else ['poisson_ratio','thermoelectric']
            data['n_trials'] = data['n_init'] + q
        # Move only paths, never scientific options or seeds, to the generated config location.
        data['environment_profiles'] = str(ROOT/'environments/local.yaml')
        for definition in data['environments'].values():
            definition['sources'] = [str((ROOT/'experiments'/p).resolve()) for p in definition.get('sources',[])]
        if name == 'niah':
            data['data_dir'] = str(ROOT/'data/niah/NIAH005/prepared')
        path = folder / f'config_q{q}.json'
        content = json.dumps(data,indent=2)+'\n'
        if path.exists() and path.read_text() != content:
            raise ValueError('Existing smoke config changed; preserve its results before a new smoke: '+str(path))
        path.write_text(content,encoding='utf-8')
        yield path, Path(data['output_dir'])


def verify_smoke(config, workspace):
    """Require all planned real trajectories, including matched initial X."""
    import numpy as np
    from .configuration import resolved
    with resolved(config) as (path, data):
        if data['benchmark'] == 'bayesmark_full':
            from bayesmark_batch.campaign import plan
            planned = plan(path, smoke=data.get('paper_smoke', False))
        else:
            from bo_benchmark.campaign import plan
            planned = plan(path)
    initial = {}
    for run_id, label in planned['labels'].items():
        p = workspace/'results/trajectories'/(run_id+'.json')
        r = json.loads(p.read_text())
        if (r['status'] != 'completed' or r['spec'] != label
                or len(r['y']) != label['n_trials'] or not np.isfinite(r['y']).all()):
            raise ValueError('Smoke did not complete: '+run_id)
        points = np.concatenate(r['optimizer_x'],axis=0)[:label['n_init']]
        key = label['task'],label['seed']
        if key in initial and not np.array_equal(initial[key],points):
            raise ValueError('Smoke initial designs differ: '+run_id)
        initial[key] = points
    print(f'Smoke passed: {len(planned["specs"])} runs; {workspace}',flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'import-data':
        parser = argparse.ArgumentParser(description='Import the frozen paper-data ZIP')
        parser.add_argument('action')
        parser.add_argument('archive',type=Path)
        args = parser.parse_args(argv)
        from .paper import import_archive
        import_archive(args.archive)
        return 0
    from .bootstrap import main as controller
    if len(argv) >= 2 and argv[1] not in (*CAMPAIGNS,'all') and argv[0] not in ('figures','smoke','setup'):
        return controller(argv)
    parser = argparse.ArgumentParser(description='Reproduce tidyHEBO paper experiments from the repository root.')
    parser.add_argument('action',choices=['run','resume','smoke','figures','check','setup','status','stop','watch','metrics','plan'])
    parser.add_argument('experiment',choices=[*CAMPAIGNS,'all'])
    args, extra = parser.parse_known_args(argv)
    names = list(CAMPAIGNS) if args.experiment == 'all' else [args.experiment]
    if args.action == 'setup':
        sys.path.insert(0,str(ROOT/'src/bo_benchmark'))
        from mamba_support import find_mamba, run_command
        mamba = find_mamba()
        commands = [[sys.executable,str(ROOT/'src/bo_benchmark_setup/setup_environments.py'),'--analysis']]
        if 'olympus' in names: commands[0].append('--olympus')
        if 'niah' in names: commands[0].append('--niah')
        commands.append([sys.executable,str(ROOT/'src/bo_benchmark_setup/setup_baselines.py')])
        if 'niah' in names: commands.append([sys.executable,str(ROOT/'src/bo_benchmark_setup/setup_niah.py')])
        for command in commands: subprocess.run(command,cwd=ROOT,check=True)
        if 'bayesmark' in names:
            from bayesmark_batch.full_suite import setup
            setup(mamba, run_command)
        return 0
    if args.action == 'figures':
        from .figures import render
        for name in names: render(name)
        return 0
    if args.action == 'metrics':
        from .paper_analysis import analyze
        for name in names: analyze(name)
        return 0
    if args.action == 'check':
        from .paper import history_records
        for name in names:
            records,_ = history_records(name)
            print(f'{name}: full historical coverage verified ({len(records)} runs)')
        return 0
    for name in names:
        if args.action == 'smoke':
            for config, workspace in smoke_configs(name):
                code = controller(['resume',str(config),'--no-metrics',*extra])
                if code: return code
                verify_smoke(config,workspace)
        else:
            code = controller([args.action,str(config_path(name)),*extra])
            if code: return code
    return 0
