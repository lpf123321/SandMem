"""Finish discovery JSON at complete records; never repair partial values."""
from collections import Counter
import json
import re
import time


def complete_prefix(text, frame_id):
    """Find the last complete array entry and validate syntax-only closure."""
    fence = text.startswith('```json\n')
    body = text[len('```json\n'):] if fence else text
    match = re.search(r'"objects"\s*:\s*\[', body)
    if match is None:
        return None
    position = last_end = match.end()
    records = []
    decoder = json.JSONDecoder()
    while position < len(body):
        while position < len(body) and body[position] in ' \t\r\n,':
            position += 1
        start = position
        try:
            record, end = decoder.raw_decode(body, position)
        except ValueError:
            break
        if not isinstance(record, dict):
            break
        records.append((record, body[start:end]))
        position = last_end = end
    if not records:
        return None
    try:
        closed = json.loads(body[:last_end] + ']}')
    except ValueError:
        return None
    # Never append a second frame_id to overwrite an observed wrong one.
    tails = [']}'] if 'frame_id' in closed else ['],"frame_id":' + str(frame_id) + '}']
    for tail in tails:
        try:
            from sandbox_vlm.perception.discovery import parse_discovery
            parse_discovery(body[:last_end] + tail, frame_id, (1000, 1000))
        except ValueError:
            continue
        return dict(records=records, boundary=last_end + (len('```json\n') if fence else 0),
                    tail=tail + ('\n```' if fence else ''))
    return None


class RecordBoundary:
    def __init__(self, tokenizer, input_width, cap, frame_id, eos_ids):
        self.tokenizer, self.width, self.cap, self.fid = tokenizer, input_width, cap, frame_id
        self.eos_ids = tuple(eos_ids)
        self.pending = []
        self.events = []
        self.check_seconds = 0.
        self.costs = {}

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def token_cost(self, text):
        if text not in self.costs:
            self.costs[text] = len(self.tokenizer.encode(text, add_special_tokens=False))
        return self.costs[text]

    def __call__(self, input_ids, scores):
        import torch
        started = time.perf_counter()
        try:
            if input_ids.shape[0] != 1:
                raise ValueError('record boundary requires one chronological discovery frame')
            ids = input_ids[0, self.width:].tolist()
            if not self.pending:
                text = self.decode(ids)
                from sandbox_vlm.perception.discovery import parse_discovery
                try:
                    parse_discovery(text, self.fid, (1000, 1000))
                    self.pending = [self.eos_ids[0]]
                except ValueError:
                    pass
                if not self.pending:
                    top = int(scores[0].argmax())
                    candidate = self.decode([*ids, top])
                    if not candidate.startswith(text):
                        return scores
                    complete = complete_prefix(candidate, self.fid)
                    if complete is None or complete['boundary'] < len(text):
                        return scores
                    keys = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r, _ in complete['records']]
                    repeated = max(Counter(keys).values()) >= 2
                    longest = max(self.token_cost(s) for _, s in complete['records'])
                    remaining = self.cap - len(ids)
                    reason = ('complete_exact_record_repeated' if repeated else
                              'reserve_complete_record_and_JSON_closure' if
                              remaining <= longest + self.token_cost(complete['tail']) + 3 else None)
                    if reason is None:
                        return scores
                    # A greedy token may include a closing brace AND a comma.
                    # Preserve its chosen field characters through the brace;
                    # replace only the following structural continuation.
                    suffix = candidate[len(text):complete['boundary']] + complete['tail']
                    closing = self.tokenizer.encode(suffix, add_special_tokens=False)
                    if self.decode(closing) != suffix:
                        raise ValueError('tokenizer cannot preserve exact closing characters')
                    if len(closing) + 1 > remaining:
                        raise ValueError('complete-record closing syntax exceeds original output cap')
                    self.pending = closing + [self.eos_ids[0]]
                    self.events.append(dict(reason=reason, output_tokens=len(ids),
                        complete_records=len(keys), closing_suffix=suffix,
                        field_values_changed=False))
            token = self.pending.pop(0)
            masked = torch.full_like(scores, -float('inf'))
            masked[0, token] = 0
            return masked
        finally:
            self.check_seconds += time.perf_counter() - started

    def report(self):
        return dict(policy='complete_record_boundary_v1', events=self.events,
                    check_seconds=self.check_seconds, partial_record_recovery=False,
                    field_values_changed=False, retries=0)
