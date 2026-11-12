"""Grow the existing controller's pending queue; never replace its BO loop."""
import asyncio
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time
from .configuration import resolved
from bo_benchmark import cli, controller
from bo_benchmark.campaign import plan, campaign_identity


def scientific(data):
    return {k:v for k,v in data.items() if k not in ('target_repeats','repeats','workers')}


class Extension:
    def __init__(self, config, planned):
        self.config, self.planned = Path(config), planned
        self.frozen = scientific(json.loads(self.config.read_text(encoding='utf-8-sig')))
        self.last_check = 0.
        self.progress = None
        self.owner = None
        self.added = []
        self.error = None

    def grow(self, queue, force=False):
        if self.owner.workspace.stop_file.exists():
            return
        if not force and time.monotonic()-self.last_check < 5:
            return
        self.last_check = time.monotonic()
        try:
            raw = json.loads(self.config.read_text(encoding='utf-8-sig'))
            if scientific(raw) != self.frozen:
                raise ValueError('Only workers and target_repeats can change live')
            target = raw.get('target_repeats',raw.get('repeats'))
            if type(target) is not int or target < 1:
                raise ValueError('target_repeats must be a positive integer')
            accepted = len(self.planned['seeds'])
            if target <= accepted:
                return
            with resolved(self.config,self.planned['workspace']) as (path,_):
                expanded = plan(path)
            old = {s.run_id:asdict(s) for s in self.planned['specs']}
            new = {s.run_id:asdict(s) for s in expanded['specs']}
            if expanded['seeds'][:accepted] != self.planned['seeds'] or any(new.get(k)!=v for k,v in old.items()):
                raise ValueError('Existing scientific specs or seeds changed')
            specs = [s for s in expanded['specs'] if s.run_id not in old]
        except (OSError,ValueError,KeyError,TypeError) as error:
            detail=str(error)
            if detail!=self.error:print('Live target ignored: '+detail,flush=True)
            self.error=detail
            return
        # Controller.run holds the workspace lease here. Persist accepted work first.
        identity=campaign_identity(expanded)
        manifest=self.owner.workspace.results/'campaign.json'
        saved=json.loads(manifest.read_text())
        for name,spec in identity['run_specs'].items():
            if name in saved['run_specs'] and saved['run_specs'][name]!=spec:
                raise ValueError('Saved run spec changed: '+name)
        saved['run_specs'].update(identity['run_specs'])
        saved.update(seeds=identity['seeds'],seed_manager=identity['seed_manager'])
        cli.write_json(manifest,saved)
        cli.write_json(self.owner.workspace.results/'seeds.json',identity['seed_manager'])
        self.planned['specs'].extend(specs)
        for key in ('labels','seeds','seed_manager'):
            self.planned[key]=expanded[key]
        # Progress and export closures refer to this same mutable planned object.
        records=self.owner.store.__dict__.get('_export_records033')
        from .runtime import build_record
        for spec in specs:
            state=controller._RunState(spec)
            self.added.append(state)
            queue.append(state)
            if records is not None:
                records[spec.run_id]=build_record(spec,self.planned,self.owner.store,self.owner.workspace,None,{})
            if self.progress is not None:
                self.progress.counts[spec.run_id]=0
                self.progress.status[spec.run_id]='pending'
        self.owner.store.__dict__.setdefault('_export_dirty033',set()).update(s.run_id for s in specs)
        if self.progress is not None and self.progress.bars:
            self.progress.bars[0].total=sum(s.protocol.budget for s in self.planned['specs'])
            self.progress.refresh()
        self.error=None
        print(f'Live target accepted: {accepted} -> {target}; added {len(specs)} runs',flush=True)


def install(config, planned):
    """Return a cleanup function; original scientific files/fingerprints stay intact."""
    extension=Extension(config,planned)
    original_controller,original_progress=cli.Controller,cli.Progress

    class Progress(original_progress):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            extension.progress=self

    class Pending(deque):
        def __bool__(self):
            extension.grow(self,force=len(self)==0)
            return len(self)>0

    class GrowingController(original_controller):
        async def run(self,*args,**kwargs):
            extension.owner=self
            original_queue=controller.deque
            controller.deque=lambda items=None: Pending(items) if items is not None else deque()
            try:
                return await super().run(*args,**kwargs)
            finally:
                controller.deque=original_queue
                # Frozen core tracks initial states in its cleanup dictionary;
                # close additional states too, including paused native workers.
                results=await asyncio.gather(*(self._close(s) for s in extension.added),return_exceptions=True)
                errors=[e for e in results if isinstance(e,BaseException)]
                if errors:raise RuntimeError('Extended worker cleanup failed: '+str(errors))

    cli.Controller,cli.Progress=GrowingController,Progress
    def restore():
        cli.Controller,cli.Progress=original_controller,original_progress
    return restore
