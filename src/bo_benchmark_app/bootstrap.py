"""Enter the configured model environment once; keep output in the same console."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from .configuration import ROOT, DEFAULT, load


def above_normal():
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x8000):
            raise ctypes.WinError(ctypes.get_last_error())


def stream(command,env,log):
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        creationflags=0x08008000 if os.name=='nt' else 0)
    try:
        while True:
            data=process.stdout.read1(8192)
            if not data:break
            log.write(data);log.flush()
            output=getattr(sys.stdout,'buffer',None)
            if output is not None:output.write(data);output.flush()
            else:sys.stdout.write(data.decode('utf-8',errors='replace'));sys.stdout.flush()
        return process.wait()
    finally:
        process.stdout.close()


def argument_index(values, path):
    return next(i for i, value in enumerate(values) if Path(value) == path)


def main(argv=None):
    from .cli import parser
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv:argv=['--help']
    a=parser().parse_args(argv)
    path=(a.config or a.experiment or DEFAULT).resolve()
    log_path=ROOT/'results/cli_bootstrap.log'
    try:
        data=load(path,a.workspace)
        log_path=Path(data['output_dir'])/'results/cli_console.log'
        if a.action=='watch':
            from .monitor import watch,open_window
            if a.window:
                open_window(path,Path(data['output_dir']))
                return 0
            return watch(Path(data['output_dir']))
        if a.window and a.action not in ('run','resume','retry'):
            raise ValueError('--window is for run/resume/retry/watch')
        if a.window and not a.background:
            from .monitor import open_window
            open_window(path,Path(data['output_dir']))
        if a.background:
            if a.action not in ('run','resume','retry'):
                raise ValueError('--background is for run/resume/retry only')
            log_path.parent.mkdir(parents=True,exist_ok=True)
            args=[v for v in argv if v not in ('--background','--window')]
            if a.experiment:args[argument_index(args,a.experiment)]=str(path)
            if a.config:args[argument_index(args,a.config)]=str(path)
            if a.workspace:args[argument_index(args,a.workspace)]=str(a.workspace.resolve())
            command=[sys.executable,str(ROOT/'run_experiments.py'),*args]
            with (log_path.parent/'background_launch.log').open('ab') as output:
                options={'creationflags':0x08000000|0x00000200|0x00008000} if os.name=='nt' else {'start_new_session':True}
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT,**options)
            print('Background PID: '+str(process.pid)+'; log: '+str(log_path),flush=True)
            if a.window:
                from .monitor import open_window
                try:open_window(path,Path(data['output_dir']))
                except OSError as error:print('Progress window failed to open: '+str(error),flush=True)
            return 0
        definition=data.get('environments',{}).get('model',{})
        python=definition.get('python',sys.executable)
        if '--config' in argv:argv[argv.index('--config')+1]=str(path)
        elif a.experiment:argv[argument_index(argv,a.experiment)]=str(path)
        else:argv.insert(1,str(path))
        if '--workspace' in argv:argv[argv.index('--workspace')+1]=str(a.workspace.resolve())
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'src'),os.environ.get('PYTHONPATH','')]),
            PYTHONUNBUFFERED='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        env['BO_EXPERIMENT_CONFIG']=str(path)
        command=[python,'-u','-m','bo_benchmark_app.cli',*argv]
        if definition.get('mamba_prefix'):
            # Stdlib-only import; controller/scientific packages enter only the child.
            sys.path.insert(0,str(ROOT/'src/bo_benchmark'))
            from mamba_support import find_mamba,run_command
            command=run_command(find_mamba(),Path(definition['mamba_prefix']),command)
        try:above_normal()
        except OSError as error:print('Priority warning: '+str(error),flush=True)
        if a.action=='plan':
            # Planning is read-only and retains machine-readable stdout.
            return subprocess.call(command,cwd=ROOT,env=env)
        log_path.parent.mkdir(parents=True,exist_ok=True)
        provenance=log_path.parent/'invocations'
        provenance.mkdir(exist_ok=True)
        invocation={'argv':argv,'experiment':str(path),'experiment_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'resolved_config':data,'application_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob('*.py'))}}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        (provenance/(stamp+'.json')).write_text(json.dumps(invocation,indent=2)+'\n',encoding='utf-8')
        with log_path.open('ab',buffering=0) as log:
            log.write(('\nSTART '+datetime.now(timezone.utc).isoformat()+'\n').encode())
            code=stream(command,env,log)
            log.write(('\nEXIT_CODE='+str(code)+'\n').encode())
        print('Console log: '+str(log_path),flush=True)
        return code
    except Exception:
        detail=traceback.format_exc()
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('ab') as log:log.write(detail.encode('utf-8'))
        print(detail,file=sys.stderr);print('Console log: '+str(log_path),flush=True)
        return 1
