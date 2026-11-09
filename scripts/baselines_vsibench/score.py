"""Use the pinned VSI-Bench functions; enforce the complete original denominator."""
import argparse
from collections import OrderedDict
from functools import partial
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from official_components import source_definitions


def official_scorer(repo):
    source = Path(repo) / 'lmms_eval/tasks/vsibench/utils.py'
    namespace = dict(np=np, pd=pd, OrderedDict=OrderedDict, partial=partial,
                     eval_logger=logging.getLogger('VSI-Bench'))
    # Constants are loaded from their literal upstream assignments as well.
    import ast
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    names = ['fuzzy_matching', 'exact_match', 'abs_dist_norm', 'mean_relative_accuracy',
             'to_float', 'vsibench_process_results', 'vsibench_aggregate_results']
    definitions, receipt = source_definitions(source, names, namespace)
    # Some pandas releases return Python float from Series.mean(), whereas
    # upstream assumes a NumPy scalar and calls .item(). Only normalize this
    # scalar conversion; keep every metric, threshold and aggregate unchanged.
    class ScalarItemCompatibility(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'item'
                    and not node.args and not node.keywords):
                return ast.copy_location(ast.Call(func=ast.Name(id='float', ctx=ast.Load()),
                    args=[node.func.value], keywords=[]), node)
            return node
    aggregate = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'vsibench_aggregate_results')
    compatible = ScalarItemCompatibility().visit(aggregate)
    module = ast.fix_missing_locations(ast.Module(body=[compatible], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), namespace)
    definitions['vsibench_aggregate_results'] = namespace['vsibench_aggregate_results']
    receipt.update(ast_modified=True, compatibility='NumPy-scalar .item() to float(); arithmetic unchanged')
    return definitions, receipt


def score(root, predictions):
    root = Path(root)
    references = [json.loads(line) for line in (root / 'data/test.jsonl').read_text().splitlines()]
    expected = {row['id'] for row in references}
    by_id = {row['question_id']: row for row in predictions}
    if len(predictions) != len(by_id):
        raise ValueError('duplicate prediction IDs')
    if set(by_id) != expected:
        raise ValueError(f'full scoring requires exactly {len(expected)} predictions; '
                         f'missing={len(expected - set(by_id))}, unexpected={len(set(by_id) - expected)}')
    functions, receipt = official_scorer(root / 'repos/thinking-in-space')
    scored = [functions['vsibench_process_results'](dict(row), [by_id[row['id']]['raw_output']])[
        'vsibench_score'] for row in references]
    results = functions['vsibench_aggregate_results'](scored)
    return dict(scores=results, question_count=len(references), video_count=288,
                scorer=receipt, reference_sha256=hashlib.sha256((root / 'data/test.jsonl').read_bytes()).hexdigest(),
                status_counts=pd.Series([row.get('status', 'generated') for row in predictions]).value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    args.output.write_text(json.dumps(score(args.root, predictions), indent=2))


if __name__ == '__main__':
    main()
