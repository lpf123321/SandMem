"""Second exact-language acceleration of the frozen discovery grammar.

Only parser traversal changes. Model inputs, logits, output limits, word/box
constraints and failure handling remain with the original implementation.
"""
from types import MethodType

from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
from lmformatenforcer.jsonschemaparser import JsonSchemaParser, StringParsingState

from .spatial_bounded_discovery import BoundedJSON as OriginalBoundedJSON, BoundedUnion
from lmformatenforcer.characterlevelparser import UnionParser
from .spatial_budgeted_discovery import BudgetContext, BudgetJSON, BudgetEnforcer


class BoundedJSON(OriginalBoundedJSON):
    """Cache immutable stack lookups; transitions still use the original parser."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._string_top = super().string_top()
        self._current_key = super().current_key()

    def string_top(self):
        return self._string_top

    def current_key(self):
        return self._current_key

    def add_character(self, c):
        role, words, inside, escape = self.transition(c)
        raw = self.raw.add_character(c)
        if isinstance(raw, UnionParser):
            return BoundedUnion([type(self)(p, self.text+c, role, words, inside,
                escape, self.alphabets) for p in raw.parsers], self.config)
        return type(self)(raw, self.text+c, role, words, inside, escape, self.alphabets)


class FastBudgetJSON(BudgetJSON):
    def closure(self):
        # The same immutable character state is queried several times during
        # one trie step. Budget is mutable between generated tokens, so it is
        # explicitly part of this per-state cache key.
        key = self.context.generated_tokens
        if getattr(self, '_closure_at', None) != key:
            # With this much remaining budget the reference loop skips every
            # candidate closing suffix; avoid rebuilding/sorting that list.
            longest_tail = len(']}],' + '"frame_id":' + str(self.context.frame_id) + '}')
            if (self.tail is None and self.context.cap - key > longest_tail + 2
                    and not self.base.can_end()):
                self._closure_value = self.context.close_tail(self.text)
            else:
                self._closure_value = super().closure()
            self._closure_at = key
        return self._closure_value

    def get_allowed_characters(self):
        key = self.context.generated_tokens
        if getattr(self, '_allowed_at', None) != key:
            self._allowed_value = super().get_allowed_characters()
            self._allowed_at = key
        return self._allowed_value

    def add_character(self, c):
        tail = self.closure()
        if tail is not None:
            if not tail or c != tail[0]:
                raise ValueError('only grammar closure fits current budget')
            tail = tail[1:]
        return type(self)(self.base.add_character(c), self.context, tail)


class PlainHeadIndex:
    def __init__(self, data, word_index):
        self.heads = []
        stack = [(data.tokenizer_tree.root, '')]
        while stack:
            node, head = stack.pop()
            for c, child in node.children.items():
                if id(child) not in word_index.special_nodes:
                    continue
                # This index is entered only inside an unescaped literal
                # field. Raw controls are absent from BoundedJSON.alphabet;
                # no ordinary head can make them legal. Escape subtrees after
                # a backslash still pass through the unchanged parser.
                if ord(c) < 32:
                    continue
                if c in ('"', '\\'):
                    self.heads.append((head, c, child))
                else:
                    stack.append((child, head + c))


def advance_plain(parser, text):
    """Bulk transition wholly inside an unrestricted quoted field.

Returns None for an illegal head. None of these characters can close a JSON
value or trigger budget closure. All syntax after the first special character
continues through the unchanged character parser and vocabulary trie.
"""
    if not text:
        return parser
    base = parser.base
    raw = base.raw
    top = base.string_top()
    words, inside = base.words, base.in_word
    for c in text:
        if c not in base.alphabet:
            return None
        if words >= 8 and not inside and c not in (' ', '\t', '\n', '\r'):
            return None
        if c.isspace():
            inside = False
        else:
            words += int(not inside)
            inside = True
            if words > 8:
                return None
    # Mirror JsonSchemaParser.add_character / StringParsingState.add_character
    # for plain characters, including its leading-whitespace behavior.
    clone = top._clone()
    clone.parsed_string += text.lstrip(' \t\n\r') if not top.parsed_string else text
    whitespaces = len(text) - len(text.rstrip(' \t\n\r'))
    if whitespaces == len(text):
        whitespaces += raw.num_consecutive_whitespaces
        last = raw.last_non_whitespace_character
    else:
        last = text.rstrip(' \t\n\r')[-1]
    stack = raw.object_stack[:-1] + [clone]
    updated = JsonSchemaParser(raw.context, raw.config, stack, whitespaces)
    updated.last_parsed_string = raw.last_parsed_string
    updated.last_non_whitespace_character = last
    updated.context.active_parser = updated
    bounded = BoundedJSON(updated, base.text + text, base.role, words, inside, '', base.alphabets)
    return FastBudgetJSON(bounded, parser.context)


class FastBudgetEnforcer(BudgetEnforcer):
    def __init__(self, data, parser, index, context, heads):
        super().__init__(data, parser, index, context)
        self.heads = heads

    def _collect_allowed_tokens(self, parser, node, allowed, shortcut):
        if (isinstance(shortcut, tuple) and shortcut[0] == 'bounded_words'
                and isinstance(parser, FastBudgetJSON) and isinstance(parser.base, BoundedJSON)
                and isinstance(parser.base.raw, JsonSchemaParser)):
            top = parser.base.string_top()
            if (isinstance(top, StringParsingState) and not top.pattern and not top.regex_parser
                    and not top.allowed_strings and top.max_length is None):
                allowed.extend(self.index.safe[shortcut[1], shortcut[2]])
                for head, c, child in self.heads.heads:
                    advanced = advance_plain(parser, head)
                    if advanced is not None and c in advanced.get_allowed_characters():
                        self.collect_special(advanced.add_character(c), child, allowed)
                return
        return super()._collect_allowed_tokens(parser, node, allowed, shortcut)


def fast_prefix(self, parser):
    fid = parser.context.model_class.properties['frame_id'].enum[0]
    context = BudgetContext(self.tokenizer, self.cap, fid)
    self.contexts.append(context)
    root = FastBudgetJSON(BoundedJSON(parser), context)
    function = TransformersPrefixAllowedTokensFn(FastBudgetEnforcer(
        self.original.data, root, self.index, context, self.fast_heads))
    first_width = [None]

    def allowed(batch_id, sent):
        if first_width[0] is None:
            first_width[0] = len(sent)
        context.generated_tokens = len(sent) - first_width[0]
        return function(batch_id, sent)
    return allowed


def install_fast_discovery_grammar(grammar):
    grammar.fast_heads = PlainHeadIndex(grammar.original.data, grammar.index)
    grammar.prefix = MethodType(fast_prefix, grammar)
    return grammar
