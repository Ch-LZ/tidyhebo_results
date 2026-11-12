"""Opt-in launch routing; the frozen executor and existing worker hashes stay intact."""
from dataclasses import replace
from pathlib import Path
import os


def headless_spawn(spawn):
    """Keep pipes/groups intact; workers must not inherit a Windows console."""
    async def create(*args, **kwargs):
        flags = kwargs.get('creationflags', 0)
        # CREATE_NO_WINDOW is ignored with DETACHED_PROCESS/CREATE_NEW_CONSOLE.
        kwargs['creationflags'] = (flags & ~0x00000018) | 0x08000000 | 0x00008000
        return await spawn(*args, **kwargs)
    return create


def install():
    from bo_benchmark.transport import JsonLinesChannel
    if getattr(JsonLinesChannel, '_baselines038', False):
        return
    JsonLinesChannel._baselines038 = True
    if os.name == 'nt':
        import asyncio
        asyncio.create_subprocess_exec = headless_spawn(asyncio.create_subprocess_exec)
    original = JsonLinesChannel.open

    async def open_worker(cls, launch):
        if launch.env.get('BO_BASELINE') and launch.argv[-1] == 'optimizer':
            argv = list(launch.argv)
            indices = [i for i, part in enumerate(argv) if Path(part).name == 'scientific_worker.py']
            if len(indices) != 1:
                raise ValueError('Cannot locate baseline worker entry point')
            argv[indices[0]] = str(Path(__file__).with_name('baseline_worker.py'))
            launch = replace(launch, argv=tuple(argv))
        return await original(launch)

    JsonLinesChannel.open = classmethod(open_worker)


def live_workers(config, workspace):
    """Apply edits to this experiment only; malformed edits keep the last limit."""
    import json
    from bo_benchmark.storage import atomic_write
    from bo_benchmark.domain import positive_int
    data = json.loads(Path(config).read_text(encoding='utf-8-sig'))
    value = data['workers']
    positive_int(value)
    atomic_write(workspace / 'results/control.json', json.dumps({'workers':value}).encode())
    return value
