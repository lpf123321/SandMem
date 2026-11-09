"""Fixtures for official aggregation and complete-denominator enforcement."""
import json
from pathlib import Path
import tempfile

from score import official_scorer, score


root = Path('work/baselines_vsibench_20261105')
functions, _ = official_scorer(root / 'repos/thinking-in-space')
rows = [json.loads(line) for line in (root / 'data/test.jsonl').read_text().splitlines()]
predictions = [dict(question_id=row['id'], raw_output=row['ground_truth']) for row in rows]
result = score(root, predictions)
assert result['scores']['overall'] == 100.0
for prediction in predictions:
    prediction['raw_output'] = ''
result = score(root, predictions)
assert result['scores']['overall'] == 0.0
try:
    score(root, predictions[:-1])
except ValueError:
    pass
else:
    raise AssertionError('missing predictions were accepted')
try:
    score(root, predictions + predictions[:1])
except ValueError:
    pass
else:
    raise AssertionError('duplicate predictions were accepted')
assert functions['fuzzy_matching']('B. explanation') == 'B'
assert functions['fuzzy_matching']('3 meters') == '3'
print('PASS: perfect/empty full denominator, duplicate/missing IDs, official first-token parsing')
