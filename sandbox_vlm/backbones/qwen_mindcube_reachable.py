"""Single-call discovery with a constructive remaining-token invariant.

Witnesses are CPU feasibility checks only. They are never appended to output,
sent to the model, or used to populate records. The model chooses every token
among continuations that can still finish inside the original output budget.
"""
from types import MethodType
import time
import re

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.tokenenforcer import TokenEnforcer
from lmformatenforcer.tokenlist import TokenList
from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn

from .qwen_mindcube_contract_total import SafeBudgetJSON, QwenSafeContractBundle, SafeDiscovery
from .qwen_mindcube_contract_fast import CachedClosedJSON, advance_plain_closed
from .spatial_budgeted_discovery import BudgetContext, BudgetEnforcer

_FIELD_START = re.compile(r'[,{]\s*"(?:name|description)"\s*:\s*"')
_COMPLETE_FIELD = re.compile(r'([,{]\s*"(?:name|description)"\s*:\s*)"(?:[^"\\]|\\.)*"')


class SemanticJSON(SafeBudgetJSON):
    """Original valid field language without the old longest-record heuristic."""
    def closure(self):
        return None


def completion_witness(parser, limit=2048):
    """Find a valid ASCII suffix, preferring closing existing values/containers.

    Its character count is a conservative token bound, since every ASCII
    character used here is checked to have a singleton token. Numeric choices
    obey the existing joint bounds/uniqueness parser. No rounding/precision cap.
    """
    if parser.can_end():
        return ''
    # Key names have finite schema alphabets; unrestricted string values can
    # close immediately (or use one character for an empty required name).
    preferred = '\"]}:,[{a9876543210.abcdefghijklmnopqrstuvwxyz_\\\\uABCDEF'
    def search(p, depth):
        if p.can_end():
            return ''
        if depth >= limit:
            return None
        choices = p.get_allowed_characters()
        ordering = preferred
        if getattr(p.base, 'role', None) and not p.base.escape and '"' not in choices:
            ordering = 'a'+ordering
        for c in dict.fromkeys(ordering):
            if c not in choices:
                continue
            suffix = search(p.add_character(c), depth+1)
            if suffix is not None:
                return c+suffix
        return None
    return search(parser, 0)


class ReachableJSON(SemanticJSON):
    def semantic(self):
        return SemanticJSON(self.base, self.context)

    def signature(self):
        if not hasattr(self, '_signature'):
            # Field text has no effect on schema completion or box uniqueness.
            # Reuse completion *proofs*, never model outputs, across equivalent
            # trie states. Keep escape states exact to retain their word rules.
            text = self.text
            if getattr(self.base, 'role', None) and not self.base.escape:
                matches = list(_FIELD_START.finditer(text))
                start = matches[-1].end()
                nonempty = bool(text[start:])
                text = text[:start]+('a' if nonempty else '')
            self._signature = _COMPLETE_FIELD.sub(lambda m: m[1]+'"a"', text)
        return self._signature

    def witness(self):
        if not hasattr(self, '_witness'):
            key = self.signature()
            cache = getattr(self.context, 'completion_cache', None)
            if cache is None:
                cache = self.context.completion_cache = {}
            if key not in cache:
                cache[key] = completion_witness(self.semantic())
            self._witness = cache[key]
        return self._witness

    def child_fits(self, c):
        child = self.add_character(c)
        suffix = child.witness()
        # Charge the token being selected AND the final EOS. Every remaining
        # suffix character has an admissible single-character token.
        return suffix is not None and len(suffix)+2 <= self.context.cap-self.context.generated_tokens

    def get_allowed_characters(self):
        key = self.context.generated_tokens
        if getattr(self, '_reachable_at', None) == key:
            return self._reachable_allowed
        if self.can_end():
            result = frozenset()  # EOS only; no padding after complete JSON.
        else:
            cache = getattr(self.context, 'threshold_cache', None)
            if cache is None:
                cache = self.context.threshold_cache = {}
            state_key = (self.signature(), getattr(self.base, 'role', None),
                         getattr(self.base, 'words', None), getattr(self.base, 'in_word', None),
                         getattr(self.base, 'escape', None))
            if state_key not in cache:
                allowed = self.semantic().get_allowed_characters()
                groups = {}
                if getattr(self.base, 'role', None) and not self.base.escape:
                    closing = self.string_closing()
                    if closing is not None:
                        groups[len(closing)+2] = allowed-{'"', '\\'}
                        if '"' in allowed:
                            groups[len(closing)+1] = frozenset('"')
                        if '\\' in allowed:
                            groups[len(closing)+3] = frozenset('\\')
                else:
                    for c in allowed:
                        suffix = self.add_character(c).witness()
                        if suffix is not None:
                            groups.setdefault(len(suffix)+2, set()).add(c)
                cache[state_key] = tuple((cost, frozenset(chars)) for cost, chars in groups.items())
            room = self.context.cap-key
            if getattr(self.context, 'profiling', False):
                self.context.profile_needed = max(self.context.profile_needed,
                    max((cost for cost, _ in cache[state_key]), default=0))
                if self.text.endswith('}') and not getattr(self.base, 'role', None):
                    self.context.profile_record_boundary = True
            result = frozenset().union(*(chars for cost, chars in cache[state_key] if cost <= room))
            # This earlier rule depends on actual completed-record token costs,
            # so it stays outside the budget-independent feasibility cache.
            forced = SafeBudgetJSON(self.base, self.context).closure()
            if forced is not None:
                result = result.intersection(forced[:1])
        self._reachable_at, self._reachable_allowed = key, result
        return result

    def shortcut_key(self):
        if SafeBudgetJSON(self.base, self.context).closure() is not None:
            return None
        key = self.base.shortcut_key()
        if isinstance(key, tuple) and key[0] == 'closed_string':
            allowed = self.get_allowed_characters()
            if allowed - {'"', '\\'}:
                return ('reachable_string',)+key[1:]
        return None

    def add_character(self, c):
        child = type(self)(self.base.add_character(c), self.context)
        if getattr(self.base, 'role', None) and not self.base.escape:
            closing = self.string_closing()
            if closing is not None:
                child._witness = closing[1:] if c == '"' else ('"'+closing if c == '\\' else closing)
        elif hasattr(self, '_witness') and self._witness and self._witness.startswith(c):
            child._witness = self._witness[1:]
        return child

    def string_closing(self):
        witness = self.witness()
        if witness is None:
            return None
        # The sole required-name minimum is one character. Witness search uses
        # 'a' there; after any actual character is selected that minimum is met.
        if witness.startswith('a"'):
            return witness[1:]
        return witness if witness.startswith('"') else None


class ReachableEnforcer(BudgetEnforcer):
    def __init__(self, data, parser, index, context, heads):
        super().__init__(data, parser, index, context)
        self.heads = heads
        self.profiles = {}

    def _compute_allowed_tokens(self, tokens, state):
        # LMFE's default broad exception handler silently substitutes EOS even
        # when JSON is incomplete. Fail visibly on an invariant violation.
        parser = state.parser
        profile_key = (parser.signature(), getattr(parser.base, 'role', None),
                       getattr(parser.base, 'words', None), getattr(parser.base, 'in_word', None),
                       getattr(parser.base, 'escape', None))
        previous = self.profiles.get(profile_key)
        room = self.context.cap-self.context.generated_tokens
        if previous is not None and room >= previous[0]:
            state.allowed_tokens = previous[1]
            return
        self.context.profiling = True
        self.context.profile_needed = 0
        self.context.profile_record_boundary = False
        allowed = TokenList(self.use_bitmask, self.vocab_size)
        try:
            self._collect_allowed_tokens(parser, self.tokenizer_tree.root, allowed, parser.shortcut_key())
        finally:
            self.context.profiling = False
        if state.parser.can_end():
            allowed.extend(self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id])
        if not allowed:
            raise ValueError('reachable discovery invariant violated: no legal next token')
        state.allowed_tokens = allowed
        # Reuse only a complete, unpruned trie profile with no completed-record
        # boundary (whose old heuristic depends on actual field token lengths).
        # The recorded maximum covers every visited character threshold, so
        # smaller budgets reuse it only while no candidate could be removed.
        if (room >= self.context.profile_needed and not self.context.profile_record_boundary
                and getattr(parser.base, 'role', None) and not parser.base.escape):
            self.profiles[profile_key] = (self.context.profile_needed, allowed)

    def _apply_new_characters(self, state, token_sequence):
        current = self.context.generated_tokens
        self.context.generated_tokens = max(0, current-1)
        try:
            new = TokenEnforcer.OutputTensorState(parser=state.parser)
            token = token_sequence[-1]
            if token in self.tokenizer_tree.new_word_tokens:
                new.current_word_tokens = [token]
                text = self.tokenizer_tree.tokens_to_strs[token]
            else:
                new.current_word_tokens = state.current_word_tokens+[token]
                previous = self.decoder(state.current_word_tokens)
                decoded = self.decoder(new.current_word_tokens)
                if not decoded.startswith(previous):
                    raise ValueError('non-prefix token decoding; invariant cannot be assumed')
                text = decoded[len(previous):]
            for c in text:
                if c not in new.parser.get_allowed_characters():
                    raise ValueError('selected token violates reachable grammar during replay')
                new.parser = new.parser.add_character(c)
            return new
        finally:
            self.context.generated_tokens = current

    def _collect_allowed_tokens(self, parser, node, allowed, shortcut):
        if isinstance(shortcut, tuple) and shortcut[0] == 'reachable_string':
            allowed.extend(self.index.safe[shortcut[1:]])
            for head, c, child in self.heads.heads:
                advanced = advance_plain_closed(parser.semantic(), head)
                if advanced is not None:
                    advanced = ReachableJSON(advanced.base, parser.context)
                    advanced._witness = parser.string_closing() if head else parser.witness()
                    if c in advanced.get_allowed_characters():
                        self.collect_special(advanced.add_character(c), child, allowed)
            return
        # No legacy string shortcut may bypass the remaining-budget mask.
        return TokenEnforcer._collect_allowed_tokens(self, parser, node, allowed, None)


def reachable_prefix(self, parser):
    fid = parser.context.model_class.properties['frame_id'].enum[0]
    context = BudgetContext(self.tokenizer, self.cap, fid)
    self.contexts.append(context)
    # Constructive guarantee requires real tokenizer support, not an assumed
    # character/token ratio. All witnesses use only this ASCII alphabet.
    singleton = {text for _, text, _ in self.original.data.regular_tokens if len(text) == 1}
    required = set('{}[],:"\\abcdefghijklmnopqrstuvwxyz_0123456789.ABCDEF')
    if not required <= singleton:
        raise ValueError('tokenizer lacks singleton tokens required by completion proof')
    root = ReachableJSON(CachedClosedJSON(parser), context)
    witness = root.witness()
    if witness is None or len(witness)+1 > self.cap:
        raise ValueError('original discovery budget cannot hold a valid schema output')
    enforcer = ReachableEnforcer(self.original.data, root, self.closed_index, context, self.closed_heads)
    fn = TransformersPrefixAllowedTokensFn(enforcer)
    width = [None]
    def allowed(batch_id, sent):
        if width[0] is None:
            width[0] = len(sent)
        context.generated_tokens = len(sent)-width[0]
        started = time.perf_counter()
        values = fn(batch_id, sent)
        context.mask_seconds = getattr(context, 'mask_seconds', 0.)+time.perf_counter()-started
        if not values:
            raise ValueError('reachable discovery invariant violated: empty token set')
        return values
    return allowed


class ReachableDiscovery(SafeDiscovery):
    def discover_batch(self, *args, **kwargs):
        original = self.model
        class Capture:
            def __getattr__(self, key):
                return getattr(original, key)
            def generate(self, **options):
                ids = original.generate(**options)
                self.outputs = ids[:, options['input_ids'].shape[1]:].detach().cpu().tolist()
                return ids
        capture = Capture()
        self.model = capture
        try:
            rows = super().discover_batch(*args, **kwargs)
        finally:
            self.model = original
        for row, ids, context in zip(rows, capture.outputs, self.grammar.contexts):
            row['output_token_ids'] = ids[:row['output_tokens']]
            row['constraint_seconds'] = context.mask_seconds
        for row in rows:
            row['output_contract'].update(
                policy='qwen_mindcube_reachable_json_v1',
                closure='each candidate prefix must retain an ASCII completion within remaining tokens including EOS',
                previous_longest_record_heuristic=True,
                witness_usage='CPU feasibility only; never emitted or used to fill object fields',
                name_word_limit=None, numeric_precision_limit=None,
                per_call_retry=False, original_output_cap=512)
        return rows


class QwenReachableBundle(QwenSafeContractBundle):
    def __init__(self, config, *, device='cuda:0'):
        super().__init__(config, device=device)
        self.discovery.__class__ = ReachableDiscovery
        self.discovery.grammar.prefix = MethodType(reachable_prefix, self.discovery.grammar)
