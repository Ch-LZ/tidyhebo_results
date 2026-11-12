"""Campaign planning and neutral results; no BO framework imports in controller."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from .seeds import SeedManager
from .task_catalog import load_catalog, task_digest
from .domain import OptimizerSpec, TaskSpec, ProtocolSpec, RunSpec, positive_int, safe_name

ROOT = Path(__file__).resolve().parents[2]


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def expand_path(value, base):
    value = os.path.expandvars(os.path.expanduser(value))
    # Expand Windows-style env vars even when a config is planned on POSIX.
    import re
    value = re.sub(r'%([^%]+)%', lambda m: os.environ.get(m.group(1), m.group(0)), value)
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def plan(path, workspace=None):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    legacy = 'variants' in data
    allowed = {'benchmark','tasks','q','repeats','start_repeat','n_trials','n_init','master_seed',
               'workers','threads','output_dir','olympus_python','model','acquisition','search','variants',
               'models','environments','startup','data_dir','candidate_limit'}
    if set(data) - allowed: raise ValueError('Unknown campaign settings: '+str(sorted(set(data)-allowed)))
    data = merge({'benchmark':'synthetic', 'tasks':['ackley','hartmann'], 'q':[1], 'repeats':30,
                  'start_repeat':0, 'n_trials':100, 'n_init':4, 'master_seed':42,
                  'workers':2, 'threads':1, 'startup':'internal'}, data)
    for key in ('repeats','n_trials','n_init','workers','threads'): positive_int(data[key])
    if not 0 <= data['start_repeat'] < data['repeats'] or type(data['start_repeat']) is not int:
        raise ValueError('Invalid start_repeat')
    if not 0 <= data['master_seed'] < 2**32 or type(data['master_seed']) is not int:
        raise ValueError('Invalid master_seed')
    if data['n_init'] >= data['n_trials']: raise ValueError('n_init must be less than n_trials')
    if data['benchmark'] not in ('synthetic','olympus','niah'): raise ValueError('Unknown benchmark')
    catalog = load_catalog()
    niahtasks = ('poisson_ratio','thermoelectric','wildfire')
    supported = (set(catalog) if data['benchmark']=='olympus' else
                 set(niahtasks) if data['benchmark']=='niah' else {'ackley','hartmann','branin'})
    tasks = sorted(supported) if data['tasks']=='all' else data['tasks']
    data_dir = expand_path(data.get('data_dir','../data/niah/prepared'),path.parent)
    candidate_limit = data.get('candidate_limit',0)
    if type(candidate_limit) is not int or candidate_limit < 0: raise ValueError('candidate_limit must be nonnegative')
    if data['benchmark']=='niah' and 'wildfire' in tasks and data['q'] != [1]:
        raise ValueError('Wildfire preserves the finite legacy q=1 protocol')
    if not isinstance(tasks,list) or not tasks or len(set(tasks)) != len(tasks) or set(tasks)-supported:
        raise ValueError('Invalid/duplicate task names')
    if not isinstance(data['q'],list) or not data['q'] or len(set(data['q'])) != len(data['q']):
        raise ValueError('q must be a list of distinct counts')
    for q in data['q']: positive_int(q)
    models = data.get('models')
    if legacy:
        if models: raise ValueError('Choose models or legacy variants, not both')
        if not data['variants']: raise ValueError('Configure variants')
        models = []
        for variant in data['variants']:
            if set(variant)-{'name','search'}: raise ValueError('Legacy variants may override search only')
            models.append({'name':variant['name'], 'kind':'tidyhebo', 'options': {
                'model':data.get('model',{}), 'acquisition':data.get('acquisition',{}),
                'search':merge({'_target_':'layered'}, merge(data.get('search',{}),variant.get('search',{}))),
                'startup':'legacy_qmace', 'n_init':data['n_init']}})
    if not isinstance(models,list) or not models: raise ValueError('Configure at least one model')
    seen = set()
    for model in models:
        if set(model)-{'name','kind','options','environment'}: raise ValueError('Unknown model field')
        safe_name(model['name'])
        if model['name'] in seen: raise ValueError('Duplicate model name')
        seen.add(model['name'])
        model.setdefault('kind','tidyhebo')
        if model['kind'] not in ('tidyhebo','random'): raise ValueError('Unsupported optimizer kind')
        options = model.setdefault('options',{})
        if set(options)-{'n_init','startup','model','acquisition','search'}: raise ValueError('Unknown optimizer option')
        options.setdefault('n_init',data['n_init']);positive_int(options['n_init'])
        if options['n_init'] >= data['n_trials']: raise ValueError('Model n_init must be below n_trials')
        options.setdefault('startup',data['startup'])
        if data['benchmark']=='niah' and options['startup']!='internal':
            raise ValueError('NIAH uses its own initial-design policy, not legacy_qmace startup')
        if options['startup'] not in ('internal','legacy_qmace'): raise ValueError('Unknown startup policy')
        if model['kind']=='random' and options['startup']=='legacy_qmace':
            raise ValueError('legacy_qmace startup belongs to tidyhebo, not random')
        if model['kind']=='random' and set(options)-{'n_init','startup'}:
            raise ValueError('Random optimizer does not accept GP/search overrides')
    environments = data.get('environments',{})
    ready = os.environ.get('BO_BENCHMARK_READY')
    if not environments and ready:
        environments = json.loads(Path(ready).read_text())
    if not environments:
        environments = {'model':{'python':sys.executable,'sources':[str(ROOT/'vendor/tidyhebo')],
                                  'threads':data['threads']},
                        'benchmark':{'python':data.get('olympus_python') or sys.executable,
                                     'threads':data['threads']}}
        if data['benchmark']=='olympus' and not data.get('olympus_python'):
            raise ValueError('Olympus requires olympus_python or explicit benchmark environment')
    else:
        environments = {name:dict(value) for name,value in environments.items()}
        for definition in environments.values():
            definition['sources'] = [expand_path(p,path.parent) for p in definition.get('sources',[])]
    for model in models:
        name=model.get('environment','model')
        if name in environments and model['kind']=='tidyhebo' and 'sources' not in environments[name]:
            source=ROOT/'vendor/tidyhebo'
            environments[name]['sources']=[str(source)] if source.is_dir() else []
    for name, definition in environments.items():
        safe_name(name)
        if set(definition)-{'python','sources','threads','env','cpu_only','mamba_prefix','startup_timeout','startup_attempts'}:
            raise ValueError('Unknown environment option')
        definition.setdefault('python',sys.executable)
        raw = os.path.expandvars(definition['python'])
        if raw not in ('python','python3'): raw=expand_path(raw,path.parent)
        definition['python']=raw
        positive_int(definition.get('threads',1))
        selected_python = Path(raw)
        inferred_prefix = selected_python.parent if os.name == 'nt' else selected_python.parent.parent
        if not definition.get('mamba_prefix') and (inferred_prefix/'conda-meta').is_dir():
            definition['mamba_prefix'] = str(inferred_prefix)
        if definition.get('mamba_prefix'):
            definition['mamba_prefix']=expand_path(definition['mamba_prefix'],path.parent)
    output = Path(workspace).resolve() if workspace else Path(expand_path(data.get('output_dir','../results/campaign'),path.parent))
    manager = SeedManager(data['master_seed'], data['repeats'])
    seeds = manager.seeds
    specs, labels = [], {}
    for repeat in range(data['start_repeat'],data['repeats']):
        for task in tasks:
            for q in data['q']:
                for model in models:
                    environment = model.get('environment','model')
                    if environment not in environments or 'benchmark' not in environments:
                        raise ValueError('Missing model or benchmark environment')
                    run_id=f"{task}-q{q}-{model['name']}-r{repeat:03d}"
                    options = model['options']
                    benchmark_options = {'benchmark':data['benchmark']}
                    if data['benchmark']=='olympus':
                        benchmark_options['task_config_sha256'] = task_digest(catalog[task])
                    if data['benchmark']=='niah': benchmark_options.update(data_dir=data_dir,candidate_limit=candidate_limit,n_trials=data['n_trials'])
                    spec = RunSpec(run_id, OptimizerSpec(model['kind'],environment,options),
                        TaskSpec(task,'benchmark',benchmark_options),
                        ProtocolSpec(data['n_trials'],q,options['n_init'],options['startup']=='legacy_qmace'),seeds[repeat])
                    specs.append(spec)
                    labels[run_id]={'benchmark':data['benchmark'],'task':task,'variant':model['name'],
                                   'q':q,'repeat':repeat,'seed':seeds[repeat],'n_trials':data['n_trials'],
                                   'n_init':options['n_init'],'startup':options['startup'],'config':options}
    return {'specs':specs,'labels':labels,'environments':environments,'workspace':output,
            'workers':data['workers'],'threads':data['threads'],'seeds':seeds,
            'seed_manager':manager.state_dict(),'legacy_config':legacy}


def campaign_identity(planned):
    # Repeat range and parallelism may change; a run ID always binds full scientific spec.
    return {'schema_version':1, 'seeds':planned['seeds'], 'seed_manager':planned['seed_manager'],
            'run_specs':{s.run_id:asdict(s) for s in planned['specs']}}
