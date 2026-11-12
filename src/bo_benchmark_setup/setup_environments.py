"""Set up isolated ready-to-run environments using the user's existing Mamba."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src/bo_benchmark'))
from mamba_support import find_mamba,run_command
from bo_benchmark_setup.definitions import write_yaml, requirements
from bo_benchmark_setup.sources import prepare_sources


def install_packages(mamba, prefix, requirements, call):
    # Try one solve first; isolate unavailable packages only after a failure.
    specs = [('pytorch'+r[5:] if r.startswith('torch==') else r) for r in requirements]
    command = [mamba,'install','--yes','--prefix',prefix,
               '--override-channels','-c','conda-forge']
    try:
        call([*command,*specs])
    except subprocess.CalledProcessError:
        for requirement, spec in zip(requirements, specs):
            try:
                call([*command,spec])
            except subprocess.CalledProcessError:
                print('Mamba failed for '+requirement+'; using pip fallback.',flush=True)
                call([*command,'pip'])
                call(run_command(mamba,prefix,['python','-m','pip','install',requirement]))


def setup(olympus=False,dry_run=False,niah=False,analysis=False,torch_device="cpu"):
    mamba='mamba' if dry_run else find_mamba()
    if torch_device not in ('cpu','cuda','auto'):raise ValueError('Invalid torch device')
    cuda = torch_device == 'cuda' or (torch_device == 'auto' and shutil.which('nvidia-smi') is not None)
    if not dry_run:
        prepare_sources(olympus=olympus)
    modern=ROOT/'environments/optimizers/scientific'
    oracle=ROOT/'environments/benchmarks/olympus'
    def call(command):
        print(json.dumps(list(map(str,command))),flush=True)
        if not dry_run:subprocess.run(list(map(str,command)),check=True)
    def inside(prefix,command):
        argv=[mamba,'run','-p',str(prefix),*map(str,command)] if dry_run else run_command(mamba,prefix,command)
        call(argv)
    def install(prefix, requirements):
        return install_packages(mamba,prefix,requirements,call)
    def ensure(prefix,definition,legacy=False,cpu_torch=True):
        version, packages = requirements(definition)
        python=prefix/('python.exe' if sys.platform=='win32' else 'bin/python')
        if not python.exists():call([mamba,'create','--yes','--prefix',prefix,'--override-channels','-c','conda-forge','python='+version,'pip','setuptools','wheel'])
        marker=prefix/'bo-requirements.sha256'
        hash_state=hashlib.sha256(b'mamba-first-v1'+definition.read_bytes())
        if not legacy and cpu_torch:hash_state.update(b"cu126" if cuda else b"cpu")
        if legacy:
            paths=[]
            for parent,folders,files in os.walk(ROOT/'vendor/olympus'):
                folders[:]=[d for d in folders if d not in ('.scratch','__pycache__','.git')]
                paths.extend(Path(parent)/name for name in files)
            for path in sorted(paths):
                hash_state.update(path.relative_to(ROOT/'vendor/olympus').as_posix().encode())
                hash_state.update(path.read_bytes())
        digest=hash_state.hexdigest()
        if dry_run or not marker.exists() or marker.read_text()!=digest:
            if not legacy and cpu_torch:
                torch = next(r for r in packages if r.startswith('torch=='))
                version = torch.split('==',1)[1]
                variant = 'pytorch-gpu' if cuda else 'pytorch-cpu'
                try:
                    call([mamba,'install','--yes','--prefix',prefix,'--override-channels',
                          '-c','conda-forge','pytorch=='+version,variant+'=='+version])
                except subprocess.CalledProcessError:
                    print('Mamba Torch solve failed; using pip fallback.',flush=True)
                    inside(prefix,['python','-m','pip','install',torch,'--index-url',
                                   'https://download.pytorch.org/whl/'+('cu126' if cuda else 'cpu')])
                packages.remove(torch)
            install(prefix, packages)
            if legacy:inside(prefix,['python','-m','pip','install','--no-deps',
                                     '--no-build-isolation','-e',ROOT/'vendor/olympus'])
            if not dry_run:marker.write_text(digest)
        return str(python)
    model_python=ensure(modern,ROOT/'environments/qmace-lab.yaml')
    inside(modern,['python','-m','pip','install','--no-deps','--no-build-isolation','-e',ROOT])
    environments={'model':{'python':model_python,'mamba_prefix':str(modern),'threads':1,
                           'sources':[str(ROOT/'vendor/tidyhebo')]},
                  'benchmark':{'python':model_python,'mamba_prefix':str(modern),'threads':1}}
    if olympus:
        oracle_python=ensure(oracle,ROOT/'environments/olympus-local.yaml',True)
        environments['benchmark']={'python':oracle_python,'mamba_prefix':str(oracle),'threads':1}
    if niah:
        conversion=ROOT/'environments/benchmarks/niah_conversion'
        conversion_python=ensure(conversion,ROOT/'environments/niah-conversion.yaml',cpu_torch=False)
        environments['conversion']={'python':conversion_python,'mamba_prefix':str(conversion),'threads':1}
    if analysis:
        install(modern,requirements(ROOT/'environments/analysis.yaml')[1])
        inside(modern,['python','-m','ipykernel','install','--user','--name','bo-benchmark-analysis',
                       '--display-name','BO benchmark (analysis)'])
    pinned = [r for r in requirements(ROOT/'environments/qmace-lab.yaml')[1] if '==' in r]
    check = ('from importlib.metadata import version; '
             'pins='+repr(dict(r.split('==',1) for r in pinned))+'; '
             'actual={k:version(k).split("+")[0] for k in pins}; '
             'assert actual==pins, ("Scientific dependency drift",pins,actual); '
             'print("Scientific dependency pins verified")')
    inside(modern,['python','-c',check])
    if not dry_run:
        path=ROOT/'environments/ready.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        write_yaml(path,environments)
    return model_python,modern,environments


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--olympus',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--niah',action='store_true');parser.add_argument('--analysis',action='store_true')
    args=parser.parse_args();setup(args.olympus,args.dry_run,args.niah,args.analysis)
