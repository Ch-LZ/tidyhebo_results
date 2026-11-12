"""Read-only console viewer: closing it never signals the benchmark processes."""
import os
import codecs
from pathlib import Path
import subprocess
import sys
import time


def open_window(config, workspace):
    if os.name != 'nt':
        raise RuntimeError('A separate console window is supported on Windows; use watch in your terminal elsewhere')
    entry = Path(__file__).resolve().parents[2] / 'run_experiments.py'
    return subprocess.Popen([sys.executable, '-u', str(entry), 'watch', str(config),
        '--workspace', str(workspace)], creationflags=0x00000010)


def console(title):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.SetConsoleTitleW.argtypes = [wintypes.LPCWSTR]
        kernel.SetConsoleTitleW(title)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.GetStdHandle(-11 & 0xffffffff)
        mode = wintypes.DWORD()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, mode.value | 0x0004)


def tail(path, output, stop=lambda: False):
    """Replay a short tail, then forward CR/ANSI bytes so tqdm bars redraw."""
    position = None
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    while not stop():
        try:
            size = path.stat().st_size
            if position is None:
                position = max(0, size - 8192)
            if size < position:
                position = 0
                decoder.reset()
            with path.open('rb') as stream:
                stream.seek(position)
                data = stream.read()
                position = stream.tell()
            if data:
                output.write(decoder.decode(data).encode('utf-8'))
                output.flush()
        except FileNotFoundError:
            pass
        time.sleep(.25)


def watch(workspace):
    console('Olympus benchmark — ' + workspace.name)
    path = workspace / 'results/cli_console.log'
    print('Read-only progress window. Close it or press Ctrl+C; computation continues.', flush=True)
    print('Log: ' + str(path), flush=True)
    print('Waiting for updates; the last displayed lines may be historical.', flush=True)
    try:
        tail(path, sys.stdout.buffer)
    except KeyboardInterrupt:
        return 0
    return 0
