"""Windows console detachment, path spelling and mixed-state recovery."""
import asyncio
import io
import json
from pathlib import Path, PureWindowsPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bo_benchmark_app import bootstrap, recovery
from bo_benchmark_app.baselines import headless_spawn
sys.path.insert(0,str(ROOT/'tests/fixtures'))
from recovery_campaign import fixture
from bo_benchmark.domain import RunOutcome
from bo_benchmark.storage import FileStore


class WindowsTests(unittest.TestCase):
    def test_windows_path_spelling(self):
        with patch.object(bootstrap,'Path',PureWindowsPath):
            for text in ['experiments/OLY-002.json',r'.\experiments\OLY-002.json',r'experiments\OLY-002.json']:
                self.assertEqual(bootstrap.argument_index(['status',text],PureWindowsPath('experiments/OLY-002.json')),1)
            self.assertEqual(bootstrap.argument_index(['--workspace','D:/path with spaces/results'],PureWindowsPath(r'D:\path with spaces\results')),1)

    def test_workers_keep_pipes_group_priority_and_remove_conflicting_console_flags(self):
        async def exercise():
            for flags in [0,0x200,0x200|0x8000,0x18|0x200|0x8000]:
                original=AsyncMock(return_value='process')
                result=await headless_spawn(original)('python','worker.py',creationflags=flags,
                    stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,env={'x':'y'})
                self.assertEqual(result,'process')
                kwargs=original.call_args.kwargs
                self.assertEqual(kwargs['creationflags'],(flags & ~0x18)|0x08008000)
                self.assertEqual(kwargs['stdin'],asyncio.subprocess.PIPE)
                self.assertEqual(kwargs['env'],{'x':'y'})
        asyncio.run(exercise())

    def test_controller_mamba_stream_has_no_console(self):
        process=Mock();process.stdout.read1.return_value=b'';process.wait.return_value=0
        with patch.object(bootstrap,'os',SimpleNamespace(name='nt')),patch.object(bootstrap.subprocess,'Popen',return_value=process) as popen:
            self.assertEqual(bootstrap.stream(['mamba','run'],{},io.BytesIO()),0)
            self.assertEqual(popen.call_args.kwargs['creationflags'],0x08008000)

    def test_retry_two_failures_keeps_pending_paused_and_completed_native_states(self):
        with tempfile.TemporaryDirectory() as folder:
            config,planned,ws=fixture(Path(folder),failed=(0,4))
            store=FileStore(ws.results)
            pending,paused=planned['specs'][1],planned['specs'][2]
            pending_path=ws.results/'runs'/pending.run_id/'state.pkl';pending_path.unlink()
            old=store.outcome(paused.run_id)
            (ws.results/'runs'/paused.run_id/'state.pkl').unlink()
            store.save_outcome(RunOutcome(paused,'paused',old.batches,direction='minimize'))
            summary=json.loads((ws.results/'summary.json').read_text())
            for r in summary['runs']:
                if r['run_id']==pending.run_id:r['status']='pending'
                if r['run_id']==paused.run_id:r['status']='paused'
            (ws.results/'summary.json').write_text(json.dumps(summary))
            failed={planned['specs'][i].run_id for i in (0,4)}
            kept={p:p.read_bytes() for p in (ws.results/'runs').glob('*/state.pkl') if p.parent.name not in failed}
            backup=recovery.archive_failed(config,expected_failed=2)
            self.assertEqual(len(list((backup/'runs').glob('*/state.pkl'))),2)
            self.assertFalse(pending_path.exists())
            self.assertTrue(all(p.read_bytes()==data for p,data in kept.items()))


if __name__=='__main__':unittest.main()
