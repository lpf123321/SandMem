"""D114: keep D112 inputs; enable the existing note boundary uniformly.

An early, natural final answer makes this logits processor an exact no-op.
Equivalence checks use syntax/length only, never task names or answer scores.
"""
import re


POLICY = dict(max_note_words=100, reserved_final_tokens=64)


def inert_boundary_proof(generation, max_output_tokens):
    raw = generation.get('raw_output', '')
    words = len(re.findall(r'\S+', raw))
    tokens = generation.get('output_tokens', max_output_tokens)
    limit = max_output_tokens - POLICY['reserved_final_tokens']
    natural = bool(re.search(r'(?:^|\n)FINAL_ANSWER:', raw))
    # ASCII byte-BPE prefixes cannot have more whitespace words than the full
    # text. Both thresholds remain inactive up to the naturally completed final
    # field; thereafter the processor returns the original logits unchanged.
    inert = (generation.get('status') == 'generated'
        and not generation.get('reached_output_limit', True)
        and raw.isascii() and natural
        and 0 < tokens < limit and words < POLICY['max_note_words'])
    return dict(equivalent=bool(inert), raw_ascii=raw.isascii(), complete_output_words=words,
                original_output_tokens=tokens, note_token_limit=limit,
                natural_final_field_present=natural, policy=dict(POLICY),
                proof_uses_answer_correctness=False)
