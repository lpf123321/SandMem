"""Authorized failure-only guard; never repairs or returns partial detections."""
from collections import Counter
import json
import time

import torch
from transformers import StoppingCriteria


def repeated_complete_record(text, repeats=3):
    """Parse complete top-level object entries, respecting strings/escapes.

    Incomplete or malformed text is not evidence of repetition. Objects must
    contain all required discovery fields. Equality includes description/box.
    """
    import re
    start = re.search(r'"objects"\s*:\s*\[', text)
    if start is None: return None
    position, counts, decoder = start.end(), Counter(), json.JSONDecoder()
    while position < len(text):
        while position < len(text) and text[position] in ' \t\n\r,': position += 1
        try: value, end = decoder.raw_decode(text, position)
        except ValueError: return None
        if not isinstance(value, dict): return None
        if (isinstance(value.get('name'), str) and isinstance(value.get('description'), str)
                and isinstance(value.get('bbox_2d'), list) and len(value['bbox_2d']) == 4):
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            counts[key] += 1
            if counts[key] >= repeats:
                return dict(record=value, occurrences=counts[key], character_end=end)
        position = end
    return None


class RepeatedDiscoveryStop(StoppingCriteria):
    def __init__(self, tokenizer, input_width, batch_size, *, interval=16):
        self.tokenizer, self.input_width = tokenizer, input_width
        self.batch_size, self.interval = batch_size, interval
        self.stops = [None] * batch_size
        self.check_seconds = 0.0

    def __call__(self, input_ids, scores, **kwargs):
        length = input_ids.shape[1] - self.input_width
        if length % self.interval == 0:
            started = time.perf_counter()
            tokens = input_ids[:, self.input_width:].tolist()
            for index, row in enumerate(tokens):
                if self.stops[index] is None:
                    text = self.tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    repeat = repeated_complete_record(text)
                    if repeat is not None:
                        self.stops[index] = dict(output_tokens=length, **repeat)
            self.check_seconds += time.perf_counter() - started
        return torch.tensor([s is not None for s in self.stops], dtype=torch.bool, device=input_ids.device)
