"""Install HEBO/Optuna separately; never change running model/oracle environments."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'src/bo_benchmark'))
from mamba_support import find_mamba, run_command
from bo_benchmark_setup.setup_environments import install_packages
from bo_benchmark_setup.definitions import requirements, write_yaml


def main():
    prefix = ROOT/'environments/optimizers/baselines'
    python = prefix/('python.exe' if sys.platform=='win32' else 'bin/python')
    mamba = find_mamba()
    version, packages = requirements(ROOT/'environments/baselines.yaml')
    torch = next(p for p in packages if p.startswith('torch=='))
    torch_version = torch.split('==',1)[1]
    def call(command):
        subprocess.run(list(map(str,command)),check=True)
    if not python.exists():
        call([mamba,'create','--yes','--prefix',prefix,'--override-channels',
              '-c','conda-forge','python='+version,'pip'])
    def inside(*args):
        call(run_command(mamba,prefix,[str(python),*args]))
    try:
        call([mamba,'install','--yes','--prefix',prefix,'--override-channels',
              '-c','conda-forge','pytorch=='+torch_version,'pytorch-cpu=='+torch_version])
    except subprocess.CalledProcessError:
        print('Mamba CPU Torch solve failed; using the pinned CPU wheel.',flush=True)
        inside('-m','pip','install',torch,'--index-url','https://download.pytorch.org/whl/cpu')
    install_packages(mamba,prefix,[p for p in packages if p != torch],call)
    inside('-m','pip','check')
    inside('-c','from hebo.optimizers.hebo import HEBO; import optuna; print("HEBO/Optuna ready")')
    profile={'baselines':{'python':str(python),'mamba_prefix':str(prefix),
                          'threads':1,'cpu_only':True,'startup_attempts':3}}
    write_yaml(ROOT/'environments/baselines.ready.yaml',profile)
    print('Ready. Run: python run_experiments.py run olympus --background')


if __name__=='__main__':
    main()
