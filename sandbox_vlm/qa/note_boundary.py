"""Enforce the existing note/final-answer boundary in one greedy generation.

Only the literal FINAL_ANSWER prefix can be forced. Answer content remains
unconstrained. This is not post-hoc extraction, a retry, or an answer solver.
"""
import re


class NoteBoundary:
    PREFIX = '\nFINAL_ANSWER: '

    def __init__(self, tokenizer, input_tokens, max_output_tokens, eos_ids, *,
                 max_note_words=100, reserved_final_tokens=64):
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens
        self.max_note_words = max_note_words
        self.note_token_limit = max_output_tokens - reserved_final_tokens
        self.reserved_final_tokens = reserved_final_tokens
        self.prefix_ids = tokenizer.encode(self.PREFIX, add_special_tokens=False)
        if self.note_token_limit <= 0 or reserved_final_tokens <= len(self.prefix_ids):
            raise ValueError('insufficient reserved answer budget')
        self.eos_ids = set(eos_ids)
        self.prefix_start = None
        self.finished = False
        self.events = []
        self.forced_tokens = []

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('note boundary requires single-input greedy decoding')
        ids = input_ids[0, self.input_tokens:].tolist()
        if self.finished:
            return scores
        text = self.decode(ids)
        if self.prefix_start is None:
            # Respect a naturally generated final field, even before the budget boundary.
            if re.search(r'(?:^|\n)FINAL_ANSWER:', text):
                self.finished = True
                return scores
            top = int(scores[0].argmax())
            words = len(re.findall(r'\S+', text))
            candidate_words = len(re.findall(r'\S+', self.decode([*ids, top])))
            reason = ('note_token_reserve' if len(ids) >= self.note_token_limit else
                      'note_word_limit' if words >= self.max_note_words or candidate_words > self.max_note_words else
                      'eos_before_final' if top in self.eos_ids else None)
            if reason is None:
                return scores
            # If the model already began this exact token prefix, finish it once.
            progress = max((k for k in range(1, min(len(ids), len(self.prefix_ids)-1)+1)
                            if ids[-k:] == self.prefix_ids[:k]), default=0)
            self.prefix_start = len(ids)-progress
            self.events.append(dict(reason=reason, trigger_output_tokens=len(ids),
                                    note_words=words, prefix_start=self.prefix_start))
        offset = len(ids)-self.prefix_start
        if offset >= len(self.prefix_ids):
            self.finished = True
            return scores
        token = self.prefix_ids[offset]
        self.forced_tokens.append(dict(output_index=len(ids), token_id=token))
        masked = scores.clone()
        masked.fill_(-float('inf'))
        masked[0, token] = 0
        return masked

    def report(self):
        return dict(max_note_words=self.max_note_words,
                    reserved_final_tokens=self.reserved_final_tokens,
                    note_token_limit=self.note_token_limit,
                    forced_prefix=self.PREFIX, prefix_token_ids=self.prefix_ids,
                    events=self.events, forced_tokens=self.forced_tokens,
                    answer_content_constrained=False, generation_calls=1)
