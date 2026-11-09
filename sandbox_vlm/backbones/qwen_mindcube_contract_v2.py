"""Close original bounded descriptions; exclude nonterminating whitespace paths.

This isolated repair keeps names unrestricted and original field semantics.
The decoder can emit one separator, then must continue text or close its value;
at the original eight-word description boundary it must close the value.
"""
import re
from types import MethodType
from lmformatenforcer.characterlevelparser import UnionParser
from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
from sandbox_vlm.backbones.qwen_mindcube_contract import OriginalNameJSON, QwenContractBundle, ContractDiscovery
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedUnion
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext, BudgetEnforcer
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON

POLICY='qwen_mindcube_output_contract_v2'


def decoded_character(escape,c):
    if escape=='\\':
        return None if c=='u' else dict(n='\n',r='\r',t='\t',b='\b',f='\f').get(c,c)
    if escape.startswith('u'):
        return chr(int((escape+c)[1:],16)) if len(escape+c)==5 else None
    return c if c not in ('"','\\') else None


class ClosedStringJSON(OriginalNameJSON):
    def transition(self,c):
        if self.role:
            value=decoded_character(self.escape,c)
            if value is not None and value.isspace() and not self.in_word:
                raise ValueError('no leading or repeated separator in compact discovery fields')
        return super().transition(c)

    def get_allowed_characters(self):
        allowed=super().get_allowed_characters()
        if self.role:
            if self.role=='description' and self.words>=8 and not self.in_word and not self.escape:
                return frozenset('"').intersection(allowed)
            if self.escape:
                checked=set()
                for c in allowed:
                    try:self.transition(c);checked.add(c)
                    except ValueError:pass
                return frozenset(checked)
            return frozenset(c for c in allowed if not c.isspace() or (c==' ' and self.in_word))
        # The original prompt requests compact JSON: structural whitespace has
        # no meaning, and must not form an endless padding loop either.
        return frozenset(c for c in allowed if not c.isspace() or (c==' ' and not self.text[-1:].isspace()))

    def shortcut_key(self):
        if self.role and not self.escape:
            if self.role=='description' and self.words>=8 and not self.in_word:return None
            return ('closed_string',self.role,self.words,self.in_word)
        return None

    def add_character(self,c):
        role,words,inside,escape=self.transition(c);raw=self.raw.add_character(c)
        if isinstance(raw,UnionParser):
            return BoundedUnion([type(self)(p,self.text+c,role,words,inside,escape,self.alphabets)
                                 for p in raw.parsers],self.config)
        return type(self)(raw,self.text+c,role,words,inside,escape,self.alphabets)


class ClosedIndex:
    def __init__(self,data,original):
        self.special_nodes=original.special_nodes;self.safe={}
        signatures=[]
        for token,text,_ in data.regular_tokens:
            if (not text or any(c in text for c in ('"','\\')) or any(ord(c)<32 for c in text)
                    or re.search(r'\s{2}',text) or any(c.isspace() and c!=' ' for c in text)):continue
            signatures.append((token,len(re.findall(r'\S+',text)),text[0].isspace()))
        for role in ('name','description'):
            for words in range(9):
                for inside in (False,True):
                    self.safe[role,words,inside]=[token for token,n,starts_space in signatures
                        if not (starts_space and not inside)
                        and (role=='name' or words+n-int(inside and not starts_space)<=8)]


class ClosedEnforcer(BudgetEnforcer):
    def _collect_allowed_tokens(self,parser,node,allowed,shortcut):
        if isinstance(shortcut,tuple) and shortcut[0]=='closed_string':
            allowed.extend(self.index.safe[shortcut[1:]])
            self.collect_special(parser,node,allowed)
            return
        return super()._collect_allowed_tokens(parser,node,allowed,shortcut)


def closed_prefix(self,parser):
    fid=parser.context.model_class.properties['frame_id'].enum[0]
    context=BudgetContext(self.tokenizer,self.cap,fid);self.contexts.append(context)
    root=FastBudgetJSON(ClosedStringJSON(parser),context)
    fn=TransformersPrefixAllowedTokensFn(ClosedEnforcer(self.original.data,root,self.closed_index,context))
    width=[None]
    def allowed(batch_id,sent):
        if width[0] is None:width[0]=len(sent)
        context.generated_tokens=len(sent)-width[0]
        return fn(batch_id,sent)
    return allowed


class ClosedDiscovery(ContractDiscovery):
    def discover_batch(self,*args,**kwargs):
        results=super().discover_batch(*args,**kwargs)
        for row in results:
            row['output_contract'].update(policy=POLICY,
                string_separator='single nonleading separator; no repeated literal/escaped whitespace',
                description_boundary='close quote after separator following original eighth word',
                field_text_filled=False,raw_output_repaired=False)
        return results


class QwenClosedContractBundle(QwenContractBundle):
    def __init__(self,config,*,device='cuda:0'):
        super().__init__(config,device=device)
        self.discovery.__class__=ClosedDiscovery
        grammar=self.discovery.grammar
        grammar.closed_index=ClosedIndex(grammar.original.data,grammar.index)
        grammar.prefix=MethodType(closed_prefix,grammar)
