"""Approved eight-word fields and exact-box uniqueness during generation.

This only constrains output records. It never merges physical identities or
repairs/truncates generated text. Existing parser, caps and repeat guard remain.
"""
import json
import re
import time
from dataclasses import replace

from lmformatenforcer.characterlevelparser import CharacterLevelParser, UnionParser
from lmformatenforcer.jsonschemaparser import StringParsingState, ObjectParsingState
from lmformatenforcer.tokenenforcer import TokenEnforcer
from sandbox_vlm.backbones.spatial_syntax import SpatialSyntaxDiscovery


class BoundedUnion(CharacterLevelParser):
    def __init__(self, parsers, config):
        super().__init__(config);self.parsers=parsers
    def get_allowed_characters(self):
        return frozenset().union(*(p.get_allowed_characters() for p in self.parsers))
    def add_character(self,c):
        parsers=[]
        for p in self.parsers:
            if c in p.get_allowed_characters():
                child=p.add_character(c)
                parsers.extend(child.parsers if isinstance(child,BoundedUnion) else [child])
        if not parsers:raise ValueError('illegal JSON branch')
        return parsers[0] if len(parsers)==1 else BoundedUnion(parsers,self.config)
    def can_end(self):return any(p.can_end() for p in self.parsers)


class BoundedJSON(CharacterLevelParser):
    def __init__(self, raw, text='', role=None, words=0, in_word=False, escape='', alphabet=None):
        super().__init__(raw.config)
        self.raw, self.text, self.role = raw, text, role
        self.words, self.in_word, self.escape = words, in_word, escape
        if alphabet is None:
            plain = frozenset(raw.config.alphabet + '\\') - frozenset(chr(i) for i in range(32))
            alphabet = (plain,plain-{'"'})
        self.alphabets = alphabet
        self.alphabet = alphabet[0]

    def string_top(self):
        if not getattr(self.raw, 'object_stack', None):
            return None
        top = self.raw.object_stack[-1]
        return top if isinstance(top, StringParsingState) else None

    def current_key(self):
        for state in reversed(getattr(self.raw, 'object_stack', [])):
            if isinstance(state, ObjectParsingState):
                return state.current_key, state.current_stage.value
        return None, None

    def transition(self, c):
        role, words, inside, escape = self.role, self.words, self.in_word, self.escape
        top = self.string_top()
        key, stage = self.current_key()
        if role is None:
            if c == '"' and top is not None and not top.seen_opening_quote and stage == 'ParsingValue' and key in ('name','description'):
                role, words, inside, escape = key, 0, False, ''
            return role, words, inside, escape
        decoded = None
        if escape == '\\':
            if c == 'u':
                escape = 'u'
            else:
                decoded = dict(n='\n',r='\r',t='\t',b='\b',f='\f').get(c,c);escape=''
        elif escape.startswith('u'):
            escape += c
            if len(escape) == 5:
                decoded = chr(int(escape[1:],16));escape=''
        elif c == '\\':
            escape = '\\'
        elif c == '"':
            role, words, inside = None, 0, False
        else:
            decoded = c
        if decoded is not None:
            if decoded.isspace():
                inside = False
            else:
                words += int(not inside);inside=True
                if words > 8:
                    raise ValueError('approved eight-word field limit')
        return role, words, inside, escape

    def duplicate_box_close(self):
        key, _ = self.current_key()
        if key != 'bbox_2d':
            return False
        text = self.text + ']'
        matches = list(re.finditer(r'"bbox_2d"\s*:\s*(\[[^\[\]]*\])', text))
        if not matches or matches[-1].end() != len(text):
            return False
        values = []
        for match in matches:
            try:
                box = json.loads(match.group(1))
            except ValueError:
                continue
            if len(box) == 4 and all(type(v) in (int,float) for v in box):
                values.append(tuple(box))
        return len(values)>1 and values[-1] in values[:-1]

    def get_allowed_characters(self):
        top = self.string_top()
        if (top is not None and top.seen_opening_quote and not top.seen_closing_quote
                and not top.allowed_strings and not top.pattern):
            allowed = self.alphabet
            if top.min_length and len(top.parsed_string)<top.min_length:
                allowed = self.alphabets[1]
        else:
            allowed = frozenset(self.raw.get_allowed_characters())
        if self.role and (self.escape or (self.words >= 8 and not self.in_word)):
            if self.words >= 8 and not self.in_word and not self.escape:
                allowed = allowed.intersection({'"','\\',' ','\t','\n','\r'})
            checked = set()
            for c in allowed:
                try:self.transition(c);checked.add(c)
                except ValueError:pass
            allowed = frozenset(checked)
        if ']' in allowed and self.duplicate_box_close():
            allowed = allowed - {']'}
        return allowed

    def add_character(self, c):
        role, words, inside, escape = self.transition(c)
        raw=self.raw.add_character(c)
        if isinstance(raw,UnionParser):
            return BoundedUnion([BoundedJSON(p,self.text+c,role,words,inside,escape,self.alphabets)
                for p in raw.parsers],self.config)
        return BoundedJSON(raw, self.text+c, role, words, inside, escape, self.alphabets)

    def can_end(self):return self.raw.can_end()
    def cache_key(self):return None
    def shortcut_key(self):
        top = self.string_top()
        if self.role and top is not None and top.seen_opening_quote and not top.seen_closing_quote and not self.escape:
            return ('bounded_words',self.words,self.in_word)
        return None


class WordIndex:
    def __init__(self, data):
        self.safe = {}
        signatures = []
        for token,text,_ in data.regular_tokens:
            if text and not any(c in text for c in ('"','\\')) and not any(ord(c)<32 for c in text):
                signatures.append((token,len(re.findall(r'\S+',text)),not text[0].isspace()))
        for words in range(9):
            for inside in (False,True):
                self.safe[words,inside] = [token for token,n,starts in signatures if words+n-int(inside and starts)<=8]
        # Only explore vocabulary paths containing quotes/escapes/control chars;
        # ordinary field tokens above are a cached finite word-count calculation.
        marked = set();stack=[(data.tokenizer_tree.root,False,False)]
        while stack:
            node,special,done=stack.pop()
            if done:
                if (special and node.tokens) or any(id(child) in marked for child in node.children.values()):marked.add(id(node))
            else:
                stack.append((node,special,True))
                for c,child in node.children.items():stack.append((child,special or c in ('"','\\') or ord(c)<32,False))
        self.special_nodes = marked


class WordEnforcer(TokenEnforcer):
    def __init__(self, data, parser, index):
        approved_config=parser.config
        super().__init__(data,parser)
        parser.config=approved_config
        self.index=index

    def _collect_allowed_tokens(self, parser, node, allowed, shortcut):
        if isinstance(shortcut,tuple) and shortcut[0]=='bounded_words':
            allowed.extend(self.index.safe[shortcut[1],shortcut[2]])
            self.collect_special(parser,node,allowed)
        else:
            super()._collect_allowed_tokens(parser,node,allowed,shortcut)

    def collect_special(self, parser, node, allowed):
        allowed.extend(node.tokens)
        characters=parser.get_allowed_characters()
        for c,child in node.children.items():
            if id(child) in self.index.special_nodes and c in characters:
                self.collect_special(parser.add_character(c),child,allowed)


class BoundedGrammar:
    def __init__(self, original):
        self.original=original
        started=time.perf_counter();self.index=WordIndex(original.data)
        self.initialization_seconds=original.initialization_seconds+time.perf_counter()-started

    def __getattr__(self,name):return getattr(self.original,name)
    def prefix(self,parser):
        from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
        return TransformersPrefixAllowedTokensFn(WordEnforcer(self.original.data,BoundedJSON(parser),self.index))


class SpatialBoundedDiscovery(SpatialSyntaxDiscovery):
    def discover_batch(self,*args,**kwargs):
        results=super().discover_batch(*args,**kwargs)
        for result in results:
            result['spatial_syntax'].update(policy='bounded_discovery_fields_v2',name_word_limit=8,
                description_word_limit=8,exact_box_duplicates='forbidden during generation; no identity merge')
        return results
