"""Resolve portable experiment configs and machine-local environment profiles."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'experiments/OLY-002.json'


def absolute(value,base):
    value=os.path.expandvars(os.path.expanduser(str(value)))
    # Existing Windows frozen files may use %NAME% expansions.
    import re
    value=re.sub(r'%([^%]+)%',lambda m:os.environ.get(m.group(1),m.group(0)),value)
    path=Path(value)
    return str(path if path.is_absolute() else (base/path).resolve())


def load(path=DEFAULT,workspace=None):
    path=Path(path).resolve()
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if 'target_repeats' in data:
        target=data.pop('target_repeats')
        if type(target) is not int or target<1:
            raise ValueError('target_repeats must be a positive integer')
        if 'repeats' in data and data['repeats']!=target:
            raise ValueError('Use target_repeats or an equal repeats value')
        data['repeats']=target
    profiles_path=data.pop('environment_profiles',None)
    profiles={};base=path.parent
    from bo_benchmark_setup.definitions import read_yaml
    if profiles_path:
        profile_file=Path(absolute(os.environ.get('BO_ENVIRONMENTS',profiles_path),base))
        if not profile_file.exists():
            profile_file=next((p for p in (ROOT/'environments/local.json', ROOT/'environments/ready.yaml', ROOT/'environments/ready_audited.json', ROOT/'environments/ready.json') if p.exists()),profile_file)
        if not profile_file.exists():
            raise FileNotFoundError('Environment profiles missing. Run: python run_experiments.py setup all')
        profiles=read_yaml(profile_file)
        if any(v.get('profile')=='baselines' for v in data.get('environments',{}).values()):
            extra=ROOT/'environments/baselines.ready.yaml'
            if not extra.exists(): extra=extra.with_suffix('.json')
            if not extra.exists():
                raise FileNotFoundError('Run first: python run_experiments.py setup all')
            profiles.update(read_yaml(extra))
        for definition in profiles.values():
            for key in ('python','mamba_prefix'):
                if key in definition and definition[key] not in ('python','python3'):
                    definition[key]=absolute(definition[key],profile_file.parent)
            definition['sources']=[absolute(p,profile_file.parent) for p in definition.get('sources',[])]
    environments={}
    for name,original in data.get('environments',{}).items():
        override=copy.deepcopy(original)
        profile=override.pop('profile',None)
        definition=copy.deepcopy(profiles[profile]) if profile else {}
        # Override paths belong to the experiment, not the local profile file.
        for key in ('python','mamba_prefix'):
            if key in override and override[key] not in ('python','python3'):
                override[key]=absolute(override[key],base)
        if 'sources' in override:
            override['sources']=[absolute(p,base) for p in override['sources']]
        definition.update(override)
        environments[name]=definition
    if 'environments' in data:
        data['environments']=environments
    data['output_dir']=absolute(workspace if workspace else data.get('output_dir','../results/campaign'),Path.cwd() if workspace else base)
    if 'data_dir' in data or data.get('benchmark')=='niah':
        data['data_dir']=absolute(data.get('data_dir','../data/niah/prepared'),base)
    if data.get('olympus_python'):
        data['olympus_python']=absolute(data['olympus_python'],base)
    if 'repeats' in data and 'target_repeats' in json.loads(path.read_text(encoding='utf-8-sig')):
        saved=Path(data['output_dir'])/'results/campaign.json'
        if saved.exists():
            data['repeats']=max(data['repeats'],len(json.loads(saved.read_text())['seeds']))
    return data


@contextmanager
def resolved(path=DEFAULT,workspace=None):
    data=load(path,workspace)
    with tempfile.TemporaryDirectory(prefix='bo-config-') as folder:
        target=Path(folder)/'resolved.json'
        target.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        yield target,data
