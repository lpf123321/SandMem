"""Submit sealed A100 runs serially; resume only unattempted video checkpoints.

No score-dependent choices, retries of failed inputs, or parameter changes.
This controller itself runs on the login CPU, never requests another GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.rename(path)


def wait_job(job_id):
    while True:
        result = subprocess.run(['squeue', '-h', '-j', str(job_id), '-o', '%T'],
                                capture_output=True, text=True)
        if result.returncode == 0 and not result.stdout.strip():
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--after-job', type=int)
    parser.add_argument('--after-run', type=Path)
    args = parser.parse_args()
    state_path = args.run / 'serial_controller.json'
    if state_path.exists():
        raise ValueError('controller already started; inspect existing state')
    state = dict(method=args.method, run=str(args.run), jobs=[], status='waiting_for_slot',
                 after_job=args.after_job,
                 after_run=str(args.after_run) if args.after_run else None,
                 controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(state_path, state)
    if args.after_job:
        wait_job(args.after_job)
    if args.after_run:
        while True:
            previous = args.after_run / 'serial_controller.json'
            if previous.exists():
                previous_status = json.loads(previous.read_text())['status']
                if previous_status not in ('waiting_for_slot', 'job_active'):
                    break
            time.sleep(30)
    launcher = args.run / 'full.sbatch'
    launcher.write_bytes((Path(__file__).parent / 'full.sbatch').read_bytes())
    state['launcher_sha256'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    for _ in range(8):
        command = ['sbatch', '--parsable',
            '--output=' + str(args.run / 'slurm_%j.out'),
            '--error=' + str(args.run / 'slurm_%j.err'), str(launcher), str(args.run), args.method]
        submitted = subprocess.run(command, capture_output=True, text=True)
        state.setdefault('submissions', []).append(dict(command=command,
            returncode=submitted.returncode, stdout=submitted.stdout, stderr=submitted.stderr))
        if submitted.returncode:
            state['status'] = 'submission_failed'
            write(state_path, state)
            raise RuntimeError(submitted.stderr)
        job_id = int(submitted.stdout.strip().split(';')[0])
        state['jobs'].append(job_id)
        state['status'] = 'job_active'
        write(state_path, state)
        print('submitted', job_id, flush=True)
        wait_job(job_id)
        receipt = args.run / ('job_' + str(job_id) + '.json')
        if not receipt.exists():
            state['status'] = 'job_failed_without_clean_video_boundary'
            write(state_path, state)
            return
        result = json.loads(receipt.read_text())
        if result['status'] == 'all_videos_attempted':
            state['status'] = 'completed' if (args.run / 'scores.json').exists() else 'scoring_failed'
            write(state_path, state)
            return
        if result['status'] != 'time_checkpoint':
            state['status'] = result['status']
            write(state_path, state)
            return
        print('clean video checkpoint; continue unattempted videos', flush=True)
    state['status'] = 'controller_job_limit_reached'
    write(state_path, state)


if __name__ == '__main__':
    main()
