"""Call pinned original scoring functions, separately from inference inputs."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import math

OFFICIAL_COMMIT = '058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b'


def official_module(repo, task):
    if task not in {'recall', 'count', 'count_streaming'}:
        raise ValueError('unsupported official task')
    repo = Path(repo)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != OFFICIAL_COMMIT:
        raise ValueError('official scorer revision changed')
    directory = {'recall': 'cambrians_vsr', 'count': 'cambrians_vsc', 'count_streaming': 'cambrians_vsc_streaming'}[task]
    relative = f'lmms-eval/lmms_eval/tasks/{directory}/utils.py'
    path = repo / relative
    tracked = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{relative}'])
    if path.read_bytes() != tracked:
        raise ValueError('official scorer file modified')
    spec = importlib.util.spec_from_file_location('pinned_official_' + directory, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dict(repo_commit=revision, path=relative, sha256=hashlib.sha256(tracked).hexdigest())


def score_predictions(records, *, task, official_repo):
    module, provenance = official_module(official_repo, task)
    scored, failures, invalid_predictions = [], [], []
    for record in records:
        target_doc = deepcopy(record['target_doc'])
        prediction = record.get('prediction', '')
        try:
            if task == 'recall' and (not isinstance(prediction, str) or not prediction.strip()):
                # D046: an unanswered MC question remains in the denominator.
                # The pinned official matcher scores the empty string as wrong.
                # Never recover a letter from truncated reasoning or a target.
                invalid_predictions.append(dict(sample_id=record['sample_id'],
                    reason='missing_or_nontext_prediction' if not isinstance(prediction, str) else 'empty_prediction',
                    original_type=type(prediction).__name__, counted_incorrect=True))
                prediction = ''
            if task == 'count_streaming':
                import json
                values = json.loads(prediction)
                if not isinstance(values, list) or len(values) != len(target_doc['answers']):
                    failures.append(dict(sample_id=record['sample_id'], reason='streaming_prediction_length_mismatch'))
                    continue  # Prevent upstream zip from ignoring missing answers.
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError('nonfinite_or_nonnumeric_streaming_prediction')
            elif task == 'count' and not math.isfinite(float(prediction)):
                raise ValueError('nonfinite_count_prediction')
            processed = module.process_results(target_doc, [prediction])
            scored.append(processed['score'])
        except Exception as exc:
            failures.append(dict(sample_id=record['sample_id'], reason=type(exc).__name__))
    # Unexpected scorer errors still fail closed; handled unanswered Recall
    # questions have already been passed through the official matcher as wrong.
    result = module.aggregate_results(scored) if scored and not failures else None
    output = dict(status='ok' if not failures else 'scoring_failed', official=provenance,
                n_records=len(records), n_scored=len(scored), failures=failures, metrics=result)
    if task == 'recall':
        output.update(scoring_policy='D046_unanswered_recall_counted_incorrect_full_denominator',
            invalid_predictions=invalid_predictions, n_answer_failures=len(invalid_predictions),
            n_correct=sum(int(row['accuracy']) for row in scored),
            n_valid_predictions=len(records)-len(invalid_predictions))
    return output
