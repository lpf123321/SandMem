"""Video-boundary experiment checkpoints, never semantic memory paging."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class VideoCheckpoints:
    def __init__(self, output, manifest, *, maximum_session_seconds=8*3600, clock=time.monotonic):
        self.output, self.manifest, self.clock = Path(output), manifest, clock
        if not 0 < maximum_session_seconds <= 10*3600:
            raise ValueError('safe video-boundary session must be <=10h')
        self.started, self.maximum_seconds = clock(), maximum_session_seconds
        self.stop_requested = False
        self.current_index = None
        self.lock = (self.output/'execution.lock').open('a+')
        try:fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise RuntimeError('another full-run worker already holds the experiment lock')
        self.manifest_digest = json_sha(manifest)
        path = self.output/'progress.json'
        self.progress = json.loads(path.read_text()) if path.exists() else dict(
            manifest_digest=self.manifest_digest, records={}, interrupted_attempts=[], sessions=[])
        if self.progress['manifest_digest'] != self.manifest_digest:
            self.close()
            raise ValueError('checkpoint belongs to different inputs/code/configuration')
        # A crash between durable video marker and global index update loses no completed result.
        for i, record in enumerate(manifest['records']):
            marker = self.output/f'sample_{i:02d}'/'record_done.json'
            if marker.exists():
                row = json.loads(marker.read_text())
                if row['record_digest'] != json_sha(record) or row['manifest_digest'] != self.manifest_digest:
                    self.close()
                    raise ValueError('completed record marker provenance mismatch')
                self.progress['records'][str(i)] = row
        self.group_indices = {}
        counts = {}
        for i, r in enumerate(manifest['records']):
            key=(r['task'],r['duration'])
            self.group_indices[i]=counts.get(key,0)
            counts[key]=counts.get(key,0)+1
        self.session_id=uuid.uuid4().hex
        self.progress.setdefault('active_sessions',{})[self.session_id]=dict(
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),started_unix=time.time(),pid=os.getpid())
        self._save_index()

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    @property
    def all_done(self):return len(self.progress['records']) == len(self.manifest['records'])

    def should_stop(self):
        return self.stop_requested or self.clock()-self.started >= self.maximum_seconds

    def is_done(self, index):return str(index) in self.progress['records']

    def begin_record(self, index):
        if self.is_done(index):raise ValueError('completed or failed records must not be repeated')
        folder=self.output/f'sample_{index:02d}'
        if folder.exists():
            archived=self.output/'interrupted_attempts'/f'sample_{index:02d}_{uuid.uuid4().hex[:10]}'
            archived.parent.mkdir(exist_ok=True)
            folder.rename(archived)
            self.progress['interrupted_attempts'].append(dict(record_index=index,
                archive=str(archived.relative_to(self.output)), reason='unfinished_previous_worker',
                reported_cost_path=str((archived/'record_report.json').relative_to(self.output))))
        self.current_index, self.record_started=index,self.clock()
        self._save_index()

    def save_record(self, report):
        i=self.current_index
        if i is None:return
        folder=self.output/f'sample_{i:02d}'
        snapshot=dict(report, record_index=i, record_elapsed_seconds=self.clock()-self.record_started)
        atomic_json(folder/'record_report.json', snapshot)
        record=self.manifest['records'][i]
        rows=sorted((q for q in report['queries'] if q['sample_id']==record['sample_id']),key=lambda q:q['query_frame_id'])
        values=[q['prediction'] for q in rows]
        complete=len(rows)==len(record['queries']) and all(v is not None for v in values)
        video=report['videos'][0] if report['videos'] else {}
        prediction=json.dumps(values) if record['streaming'] else (values[0] if values else None)
        atomic_json(self.output/record['task']/record['duration']/f'prediction_{self.group_indices[i]:04d}.json',
            dict(sample_id=record['sample_id'],prediction=prediction,
                status='ok' if complete and not video.get('perception_failures',0) else 'incomplete',
                perception_failures=video.get('perception_failures',0),fresh_perception=report.get('fresh_perception',True),
                result_scope=self.manifest['purpose'],official_task=record['task'],partition=record['partition'],
                qa_statuses={s:sum(q['status']==s for q in rows)for s in {q['status']for q in rows}}))

    def finish_record(self, report, *, failed=False):
        self.save_record(report)
        i=self.current_index
        record=self.manifest['records'][i]
        video=report['videos'][0] if report['videos'] else {}
        row=dict(record_index=i,sample_id=record['sample_id'],partition=record['partition'],
            record_digest=json_sha(record),manifest_digest=self.manifest_digest,
            status='record_failed' if failed else 'completed',
            expected_queries=len(record['queries']),queries=len(report['queries']),
            valid_answers=sum(q['status']=='ok' for q in report['queries']),
            perception_failures=video.get('perception_failures',0),
            record_elapsed_seconds=self.clock()-self.record_started,
            peak_structure_bytes=video.get('peak_structure_bytes',0),
            error=report.get('record_error') if failed else None)
        atomic_json(self.output/f'sample_{i:02d}'/'record_done.json',row)
        self.progress['records'][str(i)]=row
        self.current_index=None
        self._save_index()

    def finish_session(self, report):
        row={k:report[k]for k in ('slurm_job_id','status','wall_seconds','model_load_seconds',
            'gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','process_peak_rss_bytes')if k in report}
        self.progress['sessions'].append(row)
        self.progress['active_sessions'].pop(self.session_id,None)
        self._save_index()
        atomic_json(self.output/'sessions'/f'{row.get("slurm_job_id","local")}_{uuid.uuid4().hex[:8]}.json',row)

    def _save_index(self):
        self.progress.update(status='completed' if self.all_done else 'in_progress',
            records_expected=len(self.manifest['records']),records_finished=len(self.progress['records']),
            record_failures=sum(r['status']=='record_failed'for r in self.progress['records'].values()),
            queries_expected=sum(len(r['queries'])for r in self.manifest['records']),
            valid_answers=sum(r['valid_answers']for r in self.progress['records'].values()),
            session_elapsed_seconds=self.clock()-self.started,
            purpose=self.manifest['purpose'],checkpoint_unit='completed_video_including_failures')
        atomic_json(self.output/'progress.json',self.progress)
