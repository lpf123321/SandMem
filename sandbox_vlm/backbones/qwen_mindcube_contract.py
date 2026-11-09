"""Isolated Qwen output-contract repair. No weights, evidence or identities change.

Shared parser utilities are model-independent; importing them does not construct
Spatial or load its processor. Names remain unrestricted. Descriptions retain
the original eight-word instruction. Raw outputs still face the original parser.
"""
import re
from contextlib import contextmanager
from types import MethodType

from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.characterlevelparser import UnionParser
from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
from sandbox_vlm.backbones.qwen_reference_bundle import QwenReferenceBundle, AuditedQwenAnswerer
from sandbox_vlm.perception.discovery import QwenDiscovery
from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens, DiscoverySyntaxModel, QASyntaxModel, discovery_schema
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedJSON, BoundedUnion
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar, BudgetContext, BudgetEnforcer
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON

POLICY = 'qwen_mindcube_output_contract_v1'


class OriginalNameJSON(BoundedJSON):
    def transition(self, c):
        # Only descriptions had a word limit in the original Qwen prompt.
        if self.role == 'name':
            saved = self.words
            self.words = 0
            try:
                role, words, inside, escape = super().transition(c)
            finally:
                self.words = saved
            return role, 0, inside, escape
        return super().transition(c)

    def shortcut_key(self):
        if self.role == 'name' and not self.escape:
            return self.raw.shortcut_key()
        return super().shortcut_key()

    def add_character(self, c):
        role, words, inside, escape = self.transition(c)
        raw = self.raw.add_character(c)
        if isinstance(raw, UnionParser):
            return BoundedUnion([type(self)(p, self.text+c, role, words, inside, escape, self.alphabets)
                                 for p in raw.parsers], self.config)
        return type(self)(raw, self.text+c, role, words, inside, escape, self.alphabets)


def discovery_prefix(self, parser):
    fid = parser.context.model_class.properties['frame_id'].enum[0]
    context = BudgetContext(self.tokenizer, self.cap, fid)
    self.contexts.append(context)
    root = FastBudgetJSON(OriginalNameJSON(parser), context)
    function = TransformersPrefixAllowedTokensFn(BudgetEnforcer(self.original.data, root, self.index, context))
    width = [None]
    def allowed(batch_id, sent):
        if width[0] is None: width[0] = len(sent)
        context.generated_tokens = len(sent)-width[0]
        return function(batch_id, sent)
    return allowed


def visible_choices(messages):
    blocks = [b['text'] for m in messages for b in m['content']
              if b.get('type') == 'text' and b.get('text', '').startswith('Original question (verbatim):\n')]
    if len(blocks) != 1: raise ValueError('exactly one original question required')
    choices = list(dict.fromkeys(re.findall(r'\b([A-Z])\.\s+', blocks[0])))
    if len(choices) < 2 or choices != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]):
        raise ValueError('MindCube visible choices must be an ordered A..N list')
    return choices


class RegexModel:
    def __init__(self, original, grammar, pattern, cap):
        self.original, self.grammar, self.pattern, self.cap = original, grammar, pattern, cap
    def __getattr__(self, key): return getattr(self.original, key)
    def generate(self, **kwargs):
        if 'prefix_allowed_tokens_fn' in kwargs: raise ValueError('duplicate answer constraint')
        kwargs['prefix_allowed_tokens_fn'] = self.grammar.prefix(RegexParser(self.pattern, config=self.grammar.parser_config(self.cap)))
        return self.original.generate(**kwargs)


class ContractDiscovery(QwenDiscovery):
    def discover_batch(self, paths, frame_ids, **kwargs):
        self.grammar.contexts = []
        original = self.model
        self.model = DiscoverySyntaxModel(original, self.grammar, frame_ids, self.config['max_new_tokens'])
        try:
            results = super().discover_batch(paths, frame_ids, **kwargs)
        finally:
            self.model = original
        for row, context in zip(results, self.grammar.contexts):
            row['output_contract'] = dict(policy=POLICY, schema=discovery_schema(row['frame_id']),
                source_frame_id='fixed by the single-image caller in generation, never corrected after generation',
                name_word_limit=None, description_word_limit=8, output_cap=context.cap,
                exact_duplicate_boxes='forbidden within a single output; no physical identity inference',
                coordinates='canonical nonnegative decimals, original bounds and positive dimensions',
                closure_events=[v for text,v in context.events.items() if row['raw_output'].startswith(text)],
                closure='only syntax after complete field values; no partial record recovery',
                parser='original strict parse_discovery', retries=0)
        return results


class ContractAnswerer(AuditedQwenAnswerer):
    def answer(self, messages, images, limits, **kwargs):
        choices = visible_choices(messages)
        pattern = '(?:'+'|'.join(choices)+')'
        first = messages[0]['content'][0].get('text', '')
        sparse = 'at most 100 words' in first and 'FINAL_ANSWER:' in first
        original = self.model
        wrapper = QASyntaxModel(original, self.processor.tokenizer, self.grammar, pattern, limits.max_output_tokens) if sparse else RegexModel(original, self.grammar, pattern, limits.max_output_tokens)
        self.model = wrapper
        try:
            result = super().answer(messages, images, limits, **kwargs)
        finally:
            self.model = original
        result['output_contract'] = dict(policy=POLICY, sparse=sparse, visible_choice_letters=choices,
            answer_selected_by='model logits among all visible option letters; no answer key',
            output_cap=limits.max_output_tokens, input_messages_changed=False, retries=0,
            added_candidate_note_boundary=wrapper.boundary.report() if sparse and wrapper.boundary else None,
            final_field_mask_steps=wrapper.field.masked_steps if sparse and wrapper.field else None)
        return result


class QwenContractBundle(QwenReferenceBundle):
    def __init__(self, config, *, device='cuda:0'):
        super().__init__(config, device=device)
        syntax = SyntaxTokens(self.runtime.processor.tokenizer)
        grammar = BudgetGrammar(syntax, config['perception']['mllm']['max_new_tokens'], self.runtime.processor.tokenizer)
        grammar.prefix = MethodType(discovery_prefix, grammar)
        # Identical object layout: reuse the already loaded original Qwen objects.
        self.discovery.__class__ = ContractDiscovery
        self.discovery.grammar = grammar
        self.answerer.__class__ = ContractAnswerer
        self.answerer.grammar = syntax
