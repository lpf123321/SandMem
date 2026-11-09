"""Equivalent parser traversal acceleration for the frozen v2 language."""
from types import MethodType
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import UnionParser
from lmformatenforcer.jsonschemaparser import StringParsingState
from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import (
    ClosedStringJSON,ClosedEnforcer,QwenClosedContractBundle)
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedUnion
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON,PlainHeadIndex
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext


class CachedClosedJSON(ClosedStringJSON):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._top=super().string_top();self._key=super().current_key()
    def string_top(self):return self._top
    def current_key(self):return self._key
    def add_character(self,c):
        role,words,inside,escape=self.transition(c);raw=self.raw.add_character(c)
        if isinstance(raw,UnionParser):
            return BoundedUnion([type(self)(p,self.text+c,role,words,inside,escape,self.alphabets)
                                 for p in raw.parsers],self.config)
        return type(self)(raw,self.text+c,role,words,inside,escape,self.alphabets)


def advance_plain_closed(parser,text):
    if not text:return parser
    base=parser.base;raw=base.raw;top=base.string_top();words,inside=base.words,base.in_word
    for c in text:
        if c not in base.alphabet:return None
        if c.isspace():
            if c!=' ' or not inside:return None
            inside=False
        else:
            if base.role=='description':
                words+=int(not inside)
                if words>8:return None
            else:words=0  # OriginalNameJSON leaves names unrestricted.
            inside=True
    clone=top._clone()
    clone.parsed_string+=text.lstrip(' \t\n\r') if not top.parsed_string else text
    whitespaces=len(text)-len(text.rstrip(' \t\n\r'))
    if whitespaces==len(text):
        whitespaces+=raw.num_consecutive_whitespaces;last=raw.last_non_whitespace_character
    else:last=text.rstrip(' \t\n\r')[-1]
    updated=JsonSchemaParser(raw.context,raw.config,raw.object_stack[:-1]+[clone],whitespaces)
    updated.last_parsed_string=raw.last_parsed_string;updated.last_non_whitespace_character=last
    updated.context.active_parser=updated
    return FastBudgetJSON(CachedClosedJSON(updated,base.text+text,base.role,words,inside,'',base.alphabets),parser.context)


class FastClosedEnforcer(ClosedEnforcer):
    def __init__(self,data,parser,index,context,heads):
        super().__init__(data,parser,index,context);self.heads=heads
    def _collect_allowed_tokens(self,parser,node,allowed,shortcut):
        if (isinstance(shortcut,tuple) and shortcut[0]=='closed_string'
                and isinstance(parser,FastBudgetJSON) and isinstance(parser.base,CachedClosedJSON)
                and isinstance(parser.base.raw,JsonSchemaParser)):
            top=parser.base.string_top()
            if (isinstance(top,StringParsingState) and not top.pattern and not top.regex_parser
                    and not top.allowed_strings and top.max_length is None):
                allowed.extend(self.index.safe[shortcut[1:]])
                for head,c,child in self.heads.heads:
                    advanced=advance_plain_closed(parser,head)
                    if advanced is not None and c in advanced.get_allowed_characters():
                        self.collect_special(advanced.add_character(c),child,allowed)
                return
        return super()._collect_allowed_tokens(parser,node,allowed,shortcut)


def fast_closed_prefix(self,parser):
    fid=parser.context.model_class.properties['frame_id'].enum[0]
    context=BudgetContext(self.tokenizer,self.cap,fid);self.contexts.append(context)
    root=FastBudgetJSON(CachedClosedJSON(parser),context)
    fn=TransformersPrefixAllowedTokensFn(FastClosedEnforcer(self.original.data,root,self.closed_index,context,self.closed_heads))
    width=[None]
    def allowed(batch_id,sent):
        if width[0] is None:width[0]=len(sent)
        context.generated_tokens=len(sent)-width[0]
        return fn(batch_id,sent)
    return allowed


class QwenFastClosedContractBundle(QwenClosedContractBundle):
    def __init__(self,config,*,device='cuda:0'):
        super().__init__(config,device=device)
        grammar=self.discovery.grammar
        grammar.closed_heads=PlainHeadIndex(grammar.original.data,grammar.closed_index)
        grammar.prefix=MethodType(fast_closed_prefix,grammar)
