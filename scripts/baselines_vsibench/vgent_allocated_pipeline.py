"""Run accepted stages inside one Slurm allocation, independent of chat lifetime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    project=Path(os.environ['SANDMEM_ROOT'])
    root=project/'work/baselines_candidates_20261105'
    prepared=root/'vgent_formal_prepared_01'
    job=os.environ['SLURM_JOB_ID']
    state=dict(status='vgent_wiring_running',wire_job=job,formal_jobs=[],
               orchestration='inside Slurm allocation; no external background watcher')
    def save():
        (root/'reports/serial_completion.json').write_text(json.dumps(state,indent=2))
        (root/'reports'/('allocated_pipeline_'+job+'.json')).write_text(json.dumps(state,indent=2))
    save()
    try:
        for folder in (prepared,root/'vgent_wire_frozen_02'):
            manifest=json.loads((folder/'manifest.json').read_text())
            for path,expected in manifest['files'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Frozen dependency changed: '+path)
        pilot=root/'vgent_wire_01'
        subprocess.run(['timeout','--signal=TERM','--kill-after=30','3600',sys.executable,'-u',
            str(root/'vgent_wire_frozen_02/runtime/vgent_wire.py'),
            '--inputs',str(root/'vgent_wire_frozen_02/inputs.jsonl'),
            '--output',str(pilot),'--data-root',str(project/'work/baselines_vsibench_20261105'),
            '--repo',str(root/'repos/vgent')],check=True)
        acceptance=json.loads((pilot/'completion.json').read_text())
        if acceptance['status']!='wiring_passed_not_scored':
            raise RuntimeError('Wiring failed; full evaluation not started')
        run=root/'vgent_formal_01'
        subprocess.run([sys.executable,str(prepared/'runtime/freeze_vgent.py'),
            '--pilot',str(pilot),'--pilot-freeze',str(root/'vgent_wire_frozen_02'),
            '--inputs',str(root/'simple_formal_01/inputs.jsonl'),
            '--repo',str(root/'repos/vgent'),'--run',str(run)],check=True)
        if float(os.environ.get('SLURM_JOB_END_TIME',time.time()+43200))-time.time()<42300:
            state['status']='wiring_passed_formal_frozen_needs_fresh_allocation'
            save()
            return
        state.update(status='vgent_formal_running',formal_jobs=[job]);save()
        subprocess.run([sys.executable,'-u',str(run/'runtime/vgent_full.py'),
            '--data-root',str(project/'work/baselines_vsibench_20261105'),
            '--run',str(run)],check=True)
        completion=json.loads((run/'completion.json').read_text())
        state['status']=completion['status'];save()
    except Exception:
        state.update(status='stopped_requires_review',failure=traceback.format_exc());save()
        raise
    finally:
        sys.path.insert(0,str(prepared/'runtime'))
        from finish_candidates import report
        report(root)


if __name__=='__main__':
    main()
