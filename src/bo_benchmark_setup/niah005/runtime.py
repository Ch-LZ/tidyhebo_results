"""Local launch routing only; no scheduler or BO loop is reimplemented."""
from dataclasses import replace
from pathlib import Path
import os


def install():
    from bo_benchmark_app.baselines import install as baseline_install
    baseline_install()
    from bo_benchmark.transport import JsonLinesChannel
    if getattr(JsonLinesChannel, '_niah005', False):
        return
    JsonLinesChannel._niah005 = True
    original = JsonLinesChannel.open

    async def open_worker(cls, launch):
        if launch.env.get('BO_NIAH005') == '1' and launch.argv[-1] == 'optimizer':
            argv = list(launch.argv)
            if not launch.env.get('BO_BASELINE'):
                indices = [i for i, x in enumerate(argv) if Path(x).name == 'scientific_worker.py']
                if len(indices) != 1:
                    raise ValueError('Cannot locate scientific model worker')
                argv[indices[0]] = str(Path(__file__).with_name('model_worker.py'))
                launch = replace(launch, argv=tuple(argv))
        return await original(launch)

    JsonLinesChannel.open = classmethod(open_worker)
    if os.name == 'nt':
        import asyncio
        spawn = asyncio.create_subprocess_exec
        async def above_normal(*args, **kwargs):
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | 0x00008000
            return await spawn(*args, **kwargs)
        asyncio.create_subprocess_exec = above_normal
