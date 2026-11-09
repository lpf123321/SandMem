"""Explicitly approved generation-time syntax constraints for Spatial.

No repairs, new model calls, answer keys or inferred identities. All masks and
grammar settings are disclosed. Existing output caps/failure checks still apply.
"""
from contextlib import contextmanager
import json
import re
import time

from sandbox_vlm.backbones.spatial_transport import SpatialDiscovery, SpatialAnswerer
from sandbox_vlm.backbones.spatial_qa_format import adapt_messages


def discovery_schema(fid):
    return dict(type='object', properties=dict(
        # Installed LMFE has a numeric-const bug; a one-element numeric enum
        # represents exactly the same schema without changing dependencies.
        frame_id=dict(type='integer', enum=[fid]),
        objects=dict(type='array', items=dict(type='object', properties=dict(
            name=dict(type='string', minLength=1),
            description=dict(type='string'),
            bbox_2d=dict(type='array', items=dict(type='number'), minItems=4, maxItems=4)),
            required=['name', 'description', 'bbox_2d'], additionalProperties=False))),
        required=['frame_id', 'objects'], additionalProperties=False)


class SyntaxTokens:
    def __init__(self, tokenizer):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        started = time.perf_counter()
        self.data = build_token_enforcer_tokenizer_data(tokenizer)
        self.alphabet = ''.join(sorted(set(''.join(t[1] for t in self.data.regular_tokens))))
        self.initialization_seconds = time.perf_counter() - started

    def parser_config(self, cap):
        from lmformatenforcer import CharacterLevelParserConfig
        # Avoid the library's implicit 20-object array cap. A 512-token output
        # cannot contain more than 512 object entries; this adds no tighter cap.
        return CharacterLevelParserConfig(alphabet=self.alphabet,
            max_json_array_length=cap, force_json_field_order=False,
            max_consecutive_whitespaces=cap)

    def prefix(self, parser):
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        approved_config = parser.config
        function = build_transformers_prefix_allowed_tokens_fn(self.data, parser)
        # TokenEnforcer replaces all parser configuration with its defaults.
        # Restore the disclosed settings (in particular no hidden 20-item cap).
        parser.config = approved_config
        return function


class DiscoverySyntaxModel:
    def __init__(self, original, grammar, frame_ids, cap):
        self.original, self.grammar, self.frame_ids, self.cap = original, grammar, frame_ids, cap

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate(self, **kwargs):
        from lmformatenforcer import JsonSchemaParser
        if 'prefix_allowed_tokens_fn' in kwargs:
            raise ValueError('unexpected existing discovery grammar')
        functions = [self.grammar.prefix(JsonSchemaParser(discovery_schema(fid),
            config=self.grammar.parser_config(self.cap))) for fid in self.frame_ids]
        width = kwargs['input_ids'].shape[1]
        def allowed(batch_id, sent):
            started = time.perf_counter()
            trace = getattr(self.grammar, 'trace_path', None)
            if trace is not None:
                with trace.open('a') as stream:
                    stream.write(json.dumps(dict(event='start',frame_id=self.frame_ids[batch_id],
                        output_ids=sent[width:].tolist()))+'\n')
            values = functions[batch_id](0, sent)
            if trace is not None:
                with trace.open('a') as stream:
                    stream.write(json.dumps(dict(event='end',frame_id=self.frame_ids[batch_id],
                        output_tokens=len(sent)-width,allowed_tokens=len(values),seconds=time.perf_counter()-started))+'\n')
            return values
        kwargs['prefix_allowed_tokens_fn'] = allowed
        return self.original.generate(**kwargs)


class SpatialSyntaxDiscovery(SpatialDiscovery):
    def __init__(self, config, runtime, grammar):
        super().__init__(config, runtime)
        self.grammar = grammar

    @contextmanager
    def constrained(self, ids):
        original = self.model
        self.model = DiscoverySyntaxModel(original, self.grammar, ids, self.config['max_new_tokens'])
        try:
            yield
        finally:
            self.model = original

    def discover_batch(self, paths, frame_ids, **kwargs):
        with self.constrained(frame_ids):
            results = super().discover_batch(paths, frame_ids, **kwargs)
        for result in results:
            result['spatial_syntax'] = dict(policy='original_discovery_json_v1',
                schema=discovery_schema(result['frame_id']), object_cap=None,
                description_word_limit='original prompt asks at most eight; syntax does not mask semantic text',
                coordinate_bounds_and_box_order='original strict parser, not enforced by grammar',
                name_length_cap=None, output_cap=self.config['max_new_tokens'],
                tokenizer_initialization_seconds=self.grammar.initialization_seconds,
                answer_or_object_content_selected=False, repair_or_retry=False)
        return results

    def discover(self, path, frame_id, **kwargs):
        with self.constrained([frame_id]):
            result = super().discover(path, frame_id, **kwargs)
        result['spatial_syntax'] = dict(policy='original_discovery_json_v1', schema=discovery_schema(frame_id))
        return result


def visible_answer_pattern(messages):
    # Only the unchanged original question block is used. No task labels or
    # benchmark answers enter this API. Visible choices define output syntax.
    blocks = [c['text'] for m in messages for c in m['content']
        if c.get('type') == 'text' and c.get('text', '').startswith('Original question (verbatim):\n')]
    if len(blocks) != 1:
        raise ValueError('one original question block required')
    choices = re.findall(r'^([A-Z])\. ', blocks[0], flags=re.M)
    if choices:
        return '(?:' + '|'.join(re.escape(c) for c in dict.fromkeys(choices)) + ')', choices
    return r'[^\r\n]+', []


def sparse_parser(pattern, config):
    from dataclasses import replace
    from lmformatenforcer import RegexParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParser
    config = replace(config, alphabet=''.join(dict.fromkeys(config.alphabet + '\n\r\t')))
    # TokenEnforcer intersects this alphabet at EVERY tokenizer-trie node.
    # A cached set preserves the exact character language, avoiding repeatedly
    # rebuilding a 17k-character set for every node of the vocabulary trie.
    note_alphabet = frozenset(config.alphabet)

    class NotesThenFinal(CharacterLevelParser):
        marker = 'FINAL_ANSWER:'

        def __init__(self, line_prefix='', field=None):
            super().__init__(config)
            self.line_prefix, self.field = line_prefix, field

        def add_character(self, character):
            if self.field is not None:
                return NotesThenFinal(None, self.field.add_character(character))
            if character == '\n':
                prefix = ''
            elif self.line_prefix is not None and self.marker.startswith(self.line_prefix + character):
                prefix = self.line_prefix + character
                if prefix == self.marker:
                    return NotesThenFinal(None, RegexParser('[ ]?' + pattern, config=config))
            else:
                prefix = None
            return NotesThenFinal(prefix)

        def get_allowed_characters(self):
            return self.field.get_allowed_characters() if self.field is not None else note_alphabet

        def can_end(self):
            return self.field is not None and self.field.can_end()

        def cache_key(self):
            if self.field is None:
                return ('spatial_notes_before_final', pattern, self.line_prefix)
            key = self.field.cache_key()
            return ('spatial_final_field', pattern, key) if key is not None else None

    return NotesThenFinal()


class FinalFieldSyntax:
    def __init__(self, tokenizer, grammar, input_width, pattern, cap):
        self.tokenizer, self.input_width = tokenizer, input_width
        self.function = grammar.prefix(sparse_parser(pattern, grammar.parser_config(cap)))
        self.masked_steps = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('sparse final-field syntax requires a single input')
        allowed = self.function(0, input_ids[0])
        if not allowed:
            raise ValueError('no legal final-field token')
        result = scores.new_full(scores.shape, -float('inf'))
        result[0, allowed] = scores[0, allowed]
        self.masked_steps += 1
        return result


class QASyntaxModel:
    def __init__(self, original, tokenizer, grammar, pattern, cap):
        self.original, self.tokenizer, self.grammar = original, tokenizer, grammar
        self.pattern, self.cap = pattern, cap
        self.boundary = self.field = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate(self, **kwargs):
        from transformers import LogitsProcessorList
        from sandbox_vlm.qa.note_boundary import NoteBoundary
        from sandbox_vlm.qa.bounded_notes import POLICY
        processors = list(kwargs.pop('logits_processor', []))
        existing = [p for p in processors if isinstance(p, NoteBoundary)]
        if not existing:
            eos = self.original.generation_config.eos_token_id
            self.boundary = NoteBoundary(self.tokenizer, kwargs['input_ids'].shape[1], self.cap,
                eos if isinstance(eos, list) else [eos], **POLICY)
            processors.append(self.boundary)
        self.field = FinalFieldSyntax(self.tokenizer, self.grammar,
            kwargs['input_ids'].shape[1], self.pattern, self.cap)
        processors.append(self.field)
        kwargs['logits_processor'] = LogitsProcessorList(processors)
        return self.original.generate(**kwargs)


class SpatialSyntaxAnswerer(SpatialAnswerer):
    def __init__(self, config, runtime, grammar):
        super().__init__(config, runtime)
        self.grammar = grammar

    def answer(self, messages, images, limits, **kwargs):
        _, sparse = adapt_messages(messages)
        if not sparse:
            return super().answer(messages, images, limits, **kwargs)
        pattern, choices = visible_answer_pattern(messages)
        original = self.model
        model = QASyntaxModel(original, self.processor.tokenizer, self.grammar, pattern, limits.max_output_tokens)
        self.model = model
        try:
            result = super().answer(messages, images, limits, **kwargs)
        finally:
            self.model = original
        result['spatial_syntax'] = dict(policy='original_sparse_final_field_v1', pattern=pattern,
            visible_choice_letters=choices, correct_choice_supplied=False, answer_chosen_by='model logits among syntactically valid tokens',
            final_field_mask_steps=model.field.masked_steps if model.field else 0,
            added_candidate_note_boundary=model.boundary.report() if model.boundary else None,
            EOS_before_final='disallowed by syntax; original note word/token reserve still applies',
            original_parser=True, output_cap=limits.max_output_tokens, repair_or_retry=False)
        return result
