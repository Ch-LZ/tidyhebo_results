"""Reversible restart of failed runs; completed state checks remain mandatory."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from bo_benchmark.storage import FileStore, atomic_write
from bo_benchmark.workspace import Workspace
from bo_benchmark.campaign import plan
from .metrics import sha

def archive_failed(config, workspace=None, expected_failed=15, dry_run=False):
    """Archive failed state records; new runs begin at seed/budget origin.

    Completed runs are not moved. Shared native UUID files are not modified.
    A clean process then resumes the full campaign, skipping completed outcomes.
    """
    planned=plan(config,workspace)
    ws=Workspace(planned['workspace'])
    moved=[]
    with ws.controller_lease():
        summary_path=ws.results/'summary.json'
        summary=json.loads(summary_path.read_text(encoding='utf-8-sig'))
        runs={r['run_id']:r for r in summary['runs']}
        if len(runs)!=len(summary['runs']) or set(runs)!=set(planned['labels']):
            raise ValueError('Summary run IDs differ from the frozen campaign')
        if any(r['status'] not in ('completed','failed','pending','paused') for r in runs.values()):
            raise ValueError('Unknown campaign status')
        failed=[s for s in planned['specs'] if runs[s.run_id]['status']=='failed']
        if not failed:
            print('No failed runs. Nothing to reset.',flush=True)
            return None
        if expected_failed is not None and len(failed)!=expected_failed:
            raise ValueError(f'Expected {expected_failed} failed runs, found {len(failed)}; no state was moved')
        campaign=json.loads((ws.results/'campaign.json').read_text(encoding='utf-8-sig'))
        store=FileStore(ws.results)
        for spec in failed:
            if campaign['run_specs'].get(spec.run_id)!=asdict(spec):
                raise ValueError('Saved campaign differs from config: '+spec.run_id)
            outcome=store.outcome(spec.run_id)
            if outcome is None or outcome.spec!=spec or outcome.status!='failed':
                raise ValueError('Native run status differs from summary: '+spec.run_id)
        # Summary may be stale: verify native completed outcomes before any move.
        # This deliberately reads one state at a time, never loads shared GP snapshots.
        for index,spec in enumerate(planned['specs']):
            if index % 25 == 0:
                print(f'Verifying completed states: {index}/{len(planned["specs"])}',flush=True)
            if runs[spec.run_id]['status']=='completed':
                outcome=store.outcome(spec.run_id)
                if outcome is None or outcome.spec!=spec or outcome.status!='completed':
                    raise ValueError('Completed native state differs from summary: '+spec.run_id)
        for spec in failed:
            print('RESTART FROM SEED: '+spec.run_id+' seed='+str(spec.seed),flush=True)
        if dry_run:
            print('DRY RUN: no files moved.',flush=True)
            return None
        parent=ws.results/'retry_backups'
        parent.mkdir(exist_ok=True)
        backup=Path(tempfile.mkdtemp(prefix='retry_',dir=parent))
        atomic_write(backup/'summary_before.json',summary_path.read_bytes())
        atomic_write(backup/'RETRY.json',json.dumps({'config_sha256':sha(Path(config).read_bytes()),
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'restart_specs':[asdict(s) for s in failed], 'policy':'Restart failed runs from original seed; preserve completed and all native files'},indent=2).encode())
        try:
            for spec in failed:
                for source in (ws.results/'runs'/spec.run_id,ws.results/'trajectories'/(spec.run_id+'.json')):
                    if source.exists():
                        target=backup/source.relative_to(ws.results)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        source.rename(target)
                        moved.append((source,target))
        except BaseException:
            for source,target in reversed(moved):
                target.rename(source)
            raise
        print('Backup: '+str(backup),flush=True)
    return backup

