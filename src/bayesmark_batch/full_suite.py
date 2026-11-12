"""The public 108-task Bayesmark 0.0.8 suite and its isolated environment."""
import hashlib
from io import BytesIO
import os
from pathlib import Path
import subprocess
import tarfile
from urllib.request import urlopen

from bo_benchmark_setup.definitions import read_yaml

ROOT = Path(__file__).resolve().parents[2]

FULL_PREFIX = Path(os.environ.get('BO_BAYESMARK_PREFIX', ROOT/'environments/benchmarks/bayesmark'))
FULL_PYTHON = FULL_PREFIX / ('python.exe' if os.name == 'nt' else 'bin/python')
SOURCE = FULL_PREFIX / 'bayesmark-source-0.0.8'
URL = ('https://files.pythonhosted.org/packages/fe/35/'
       '5b3ad7f835676f53ed219cfe70e2f190cadbec21a9570e794489f721c00f/bayesmark-0.0.8.tar.gz')
SHA256 = '4a84c026d7aba41ac027a169cd965363fc1f39d4e3048a9e5c6414c28dc571a2'
MODELS = ('DT','MLP-adam','MLP-sgd','RF','SVM','ada','kNN','lasso','linear')
METRICS = {name: ('acc','nll') for name in ('breast','digits','iris','wine')}
METRICS.update({name: ('mae','mse') for name in ('boston','diabetes')})
TASKS = {f"{dataset}_{model.lower().replace('-', '_')}_{metric}": (model, dataset, metric)
         for dataset, metrics in METRICS.items() for model in MODELS for metric in metrics}
SMOKE = ('iris_dt_nll','diabetes_linear_mse')

def oracle_env(data):
    if data.get('benchmark') != 'bayesmark_full':
        raise ValueError('Expected the full paper Bayesmark suite')
    if not FULL_PYTHON.is_file() or not (SOURCE / 'bayesmark/sklearn_funcs.py').is_file():
        raise FileNotFoundError('Full Bayesmark environment missing. Run: python run_experiments.py setup bayesmark')
    return dict(python=str(FULL_PYTHON),
                **({'mamba_prefix': str(FULL_PREFIX)} if (FULL_PREFIX/'conda-meta').is_dir() else {}),
                sources=[str(SOURCE)], threads=1, cpu_only=True, startup_attempts=3,
                startup_timeout=120, env={'BO_BAYESMARK_ORACLE':'full'})

def setup(mamba, run_command):
    if not FULL_PYTHON.is_file():
        pins = read_yaml(ROOT/'environments/bayesmark.yaml')['dependencies']
        subprocess.run([mamba,'create','-y','--prefix',str(FULL_PREFIX),
                        '--override-channels','-c','conda-forge',*pins],check=True)
    marker = SOURCE / 'upstream.sha256'
    if not marker.is_file() or marker.read_text().strip() != SHA256:
        with urlopen(URL, timeout=60) as response:
            content = response.read(200_000)
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError('Bayesmark 0.0.8 archive SHA-256 mismatch')
        with tarfile.open(fileobj=BytesIO(content),mode='r:gz') as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if not member.isfile() or parts[:2] != ('bayesmark-0.0.8','bayesmark'):
                    continue
                relative = Path(*parts[1:])
                if '..' in relative.parts or relative.is_absolute():
                    raise ValueError('Invalid Bayesmark source path')
                path = SOURCE / relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
        marker.write_text(SHA256 + '\n',encoding='ascii')
    definition = oracle_env({'benchmark':'bayesmark_full'})
    check = Path(__file__).with_name('full_worker.py')
    env = dict(os.environ, PYTHONPATH=str(SOURCE), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [definition['python'], str(check), '--check']
    if definition.get('mamba_prefix'):
        command = run_command(mamba, definition['mamba_prefix'], command)
    subprocess.run(command,env=env,check=True)
    return 0
