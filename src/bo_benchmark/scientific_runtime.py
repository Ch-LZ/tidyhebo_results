"""Production command workers with independent Python/source paths per environment."""
import os
from pathlib import Path
import sys
import tempfile
import uuid

from .subprocess_runtime import (NativeSubprocessOptimizerClient, NativeSubprocessBenchmarkClient,
                                task_payload)
from .transport import JsonLinesChannel, WorkerLaunch, TransportError


class ScientificRuntime:
    def __init__(self, workspace, environments):
        self.workspace = workspace
        self.environments = environments
        self.channels = []
        self.metadata = {}
        self.records = []
        self._temporary = []

    async def _open(self, role, spec, seed, task=None):
        definition = self.environments[spec.environment]
        python = definition.get('python', sys.executable)
        sources = [str(Path(p).resolve()) for p in definition.get('sources', [])]
        env = {**definition.get('env', {}), 'PYTHONHASHSEED': str(seed),
               'PYTHONUNBUFFERED': '1', 'TIDYHEBO_CPU_PROFILE': 'off'}
        threads = str(definition.get('threads', 1))
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
            env[name] = threads
        if definition.get('cpu_only', True): env['CUDA_VISIBLE_DEVICES'] = ''
        if sources: env['PYTHONPATH'] = os.pathsep.join(sources + [os.environ.get('PYTHONPATH', '')])
        temporary = tempfile.TemporaryDirectory(prefix='bo-worker-')
        self._temporary.append(temporary)
        for name in ('TEMP', 'TMP', 'TMPDIR'): env[name] = temporary.name
        worker = Path(__file__).with_name('workers') / 'scientific_worker.py'
        argv = [python, '-u', str(worker), role]
        if definition.get('mamba_prefix'):
            from .environments import mamba_run
            argv = mamba_run(Path(definition['mamba_prefix']), argv)
        log = self.workspace.results / 'logs' / (role + '-' + uuid.uuid4().hex + '.log')
        try:
            channel = await JsonLinesChannel.open(WorkerLaunch(tuple(argv), env=env,
                startup_timeout=definition.get('startup_timeout', 60),
                startup_attempts=definition.get('startup_attempts', 1), stderr_path=log))
        except BaseException:
            temporary.cleanup()
            raise
        # Keep temporary-directory lifetime tied to the worker, including failures.
        original_close = channel.close
        async def close():
            try: await original_close()
            finally:
                temporary.cleanup()
                channel.stderr_tail.clear()
                if channel in self.channels: self.channels.remove(channel)
        channel.close = close
        self.channels.append(channel)
        try:
            if channel.ready.get('role') != role: raise TransportError('Worker role mismatch')
            channel.initialized = await channel.request('initialize', {
                'name': spec.name, 'seed': seed, 'options': dict(spec.options),
                'task': task_payload(task) if task else None,
                'checkpoint_root': str((self.workspace.results / 'native').resolve())})
            identity = channel.initialized['identity']
            from .storage import atomic_write
            import json
            provenance_file = self.workspace.results / 'worker_versions.json'
            known = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
            key = role + ':' + spec.environment + ':' + spec.name
            version = {k:identity[k] for k in ('python','code','dependencies')}
            if role == 'benchmark': version['adapter'] = identity['adapter']
            if key in known and known[key] != version:
                raise ValueError('Worker versions/assets changed; use a new workspace: ' + key)
            known[key] = version
            atomic_write(provenance_file,json.dumps(known,indent=2).encode('utf-8'))
            self.metadata.setdefault(spec.environment, []).append(channel.initialized)
            self.records.append({'role':role,'name':spec.name,'seed':seed,'options':dict(spec.options),
                'task': task.metadata.get('task') if task else spec.name,
                'pid':channel.pid,'log':str(log.relative_to(self.workspace.results)), **channel.initialized})
            return channel
        except BaseException:
            await channel.close()
            raise

    def provenance(self, spec):
        result = {}
        for record in self.records:
            target = spec.optimizer if record['role'] == 'optimizer' else spec.task
            if (record['name'] == target.name and record['options'] == dict(target.options)
                    and record['seed'] == spec.seed and record['task'] == spec.task.name):
                result[record['role']] = record
        return result

    async def open_optimizer(self, spec, environment, task, seed):
        return NativeSubprocessOptimizerClient(await self._open('optimizer', spec, seed, task),
                                               [p.name for p in task.space.parameters])

    async def open_benchmark(self, spec, environment, seed):
        return NativeSubprocessBenchmarkClient(await self._open('benchmark', spec, seed))
