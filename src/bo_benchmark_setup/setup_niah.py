"""Fetch checksum-pinned source forests and regenerate local portable oracles."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'src/bo_benchmark'))
from mamba_support import find_mamba, run_command
from bo_benchmark_setup.definitions import read_yaml


def main():
    prepared = ROOT/'data/niah/NIAH005/prepared'
    tasks = ('poisson_ratio','thermoelectric')
    metadata = {task:json.loads((prepared/(task+'.json')).read_text()) for task in tasks}
    expected_ids = {task:metadata[task]['asset_id'] for task in tasks}
    if all((prepared/(t+'.npz')).is_file() and (prepared/(t+'.json')).is_file() for t in tasks):
        # Check in the model Python: the bootstrap interpreter can be stdlib-only.
        pass
    else:
        raw = ROOT/'data/niah/raw'
        raw.mkdir(parents=True,exist_ok=True)
        for task in tasks:
            relative,expected = metadata[task]['source_file'],metadata[task]['source_sha256']
            commit = metadata[task]['source_commit']
            path = raw/(task+'.pkl')
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                continue
            temporary = path.with_suffix('.download')
            print('Downloading pinned NIAH source:',task,flush=True)
            digest = hashlib.sha256()
            with urlopen(f'https://raw.githubusercontent.com/PV-Lab/ZoMBI/{commit}/{relative}',timeout=120) as response, temporary.open('wb') as stream:
                total = 0
                for chunk in iter(lambda:response.read(4*1024*1024),b''):
                    stream.write(chunk);digest.update(chunk);total += len(chunk)
                    print(f'{task}: {total//(1024*1024)} MiB',flush=True)
            if digest.hexdigest() != expected:
                raise ValueError('Source checksum mismatch: '+task)
            temporary.replace(path)
        profiles = read_yaml(ROOT/'environments/ready.yaml')
        definition = profiles['conversion']
        command = [definition['python'],str(ROOT/'src/bo_benchmark_setup/niah005/export_assets.py'),str(raw),
                   '--output-dir',str(prepared)]
        if definition.get('mamba_prefix'):
            command = run_command(find_mamba(),definition['mamba_prefix'],command)
        subprocess.run(command,cwd=ROOT,check=True)
    profiles = read_yaml(ROOT/'environments/ready.yaml')
    definition = profiles['model']
    code = ('import sys; sys.path.insert(0,'+repr(str(ROOT/'src/bo_benchmark/workers'))+'); '
            'from niah_assets import load_asset; '
            'expected='+repr(expected_ids)+'; '
            'assert all(load_asset('+repr(str(prepared))+',t).metadata["asset_id"]==expected[t] for t in expected); '
            'print("NIAH portable assets verified")')
    command = [definition['python'],'-c',code]
    if definition.get('mamba_prefix'):
        command = run_command(find_mamba(),definition['mamba_prefix'],command)
    subprocess.run(command,cwd=ROOT,check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
