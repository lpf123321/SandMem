"""Prevent incomplete numeric prefixes being stranded by whitespace/duplicates."""
import math,re
from functools import lru_cache
from types import MethodType
from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.jsonschemaparser import StringParsingState
from sandbox_vlm.backbones.qwen_mindcube_contract_fast import (
    QwenFastClosedContractBundle,CachedClosedJSON,FastClosedEnforcer,advance_plain_closed)
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext,number_prefix_feasible,unique_float_completion
from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedDiscovery


@lru_cache(maxsize=32768)
def feasible_number(prefix,lower,inclusive,forbidden):
    if not number_prefix_feasible(prefix,lower,inclusive=inclusive):return False
    blocked=set(forbidden)
    if lower>=0:blocked.add(lower)
    if not inclusive:blocked.add(1000.)
    if '.' in prefix:
        return unique_float_completion(prefix,blocked,lower)
    # Jointly check bounds AND uniqueness. Testing these independently wrongly
    # permits 100... when its only in-bounds integer extension, 1000, is banned.
    stems=[int(prefix)]
    if prefix!='0':
        for power in (10,100,1000):
            start=int(prefix)*power;end=min(1000,(int(prefix)+1)*power-1)
            if start<=end:stems.extend(range(start,end+1))
    for stem in stems:
        if stem>1000:continue
        if lower<stem<=1000 and stem not in blocked:return True
        if stem>=1000:continue
        candidate=math.nextafter(max(float(stem),lower),math.inf)
        for _ in range(len(blocked)+2):
            if candidate>stem+1 or candidate>1000:break
            if candidate>lower and candidate not in blocked:return True
            candidate=math.nextafter(candidate,math.inf)
    return False


class SafeBudgetJSON(FastBudgetJSON):
    def get_allowed_characters(self):
        allowed=super().get_allowed_characters()
        if self.closure() is not None:return allowed
        parts=self.box_parts()
        if parts is None or len(parts)>4:return allowed
        index=len(parts)-1;prefix=parts[-1].strip()
        lower=float(parts[index-2]) if index>=2 else -1.
        blocked=set()
        if index==3:
            import json
            head=tuple(float(v) for v in parts[:3])
            for match in re.finditer(r'"bbox_2d"\s*:\s*(\[[^\[\]]*\])',self.text):
                try:box=json.loads(match.group(1))
                except ValueError:continue
                if len(box)==4 and tuple(box[:3])==head:blocked.add(box[3])
        result=set()
        valid_value=False
        if re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',prefix):
            value=float(prefix)
            valid_value=lower<value<=1000 and (index>=2 or value<1000) and value not in blocked
        for c in allowed:
            if c in '0123456789.':
                if feasible_number(prefix+c,lower,index>=2,tuple(sorted(blocked))):result.add(c)
            elif c.isspace():
                # A single separator before a number is legal JSON and still
                # leaves all valid values available. Padding after an already
                # started number is legal only if that number may terminate.
                if not prefix or valid_value:result.add(c)
            elif c in ',]':
                if valid_value:result.add(c)
            else:result.add(c)
        return frozenset(result)


class SafeFastEnforcer(FastClosedEnforcer):
    def _collect_allowed_tokens(self,parser,node,allowed,shortcut):
        if (isinstance(shortcut,tuple) and shortcut[0]=='closed_string'
                and isinstance(parser,SafeBudgetJSON) and isinstance(parser.base,CachedClosedJSON)
                and isinstance(parser.base.raw,JsonSchemaParser)):
            top=parser.base.string_top()
            if (isinstance(top,StringParsingState) and not top.pattern and not top.regex_parser
                    and not top.allowed_strings and top.max_length is None):
                allowed.extend(self.index.safe[shortcut[1:]])
                for head,c,child in self.heads.heads:
                    advanced=advance_plain_closed(parser,head)
                    if advanced is not None:
                        advanced=SafeBudgetJSON(advanced.base,advanced.context,advanced.tail)
                        if c in advanced.get_allowed_characters():
                            self.collect_special(advanced.add_character(c),child,allowed)
                return
        # Bypass the fast parent's bulk path, which returns the old base class.
        from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedEnforcer
        return ClosedEnforcer._collect_allowed_tokens(self,parser,node,allowed,shortcut)


def safe_prefix(self,parser):
    fid=parser.context.model_class.properties['frame_id'].enum[0]
    ctx=BudgetContext(self.tokenizer,self.cap,fid);self.contexts.append(ctx)
    root=SafeBudgetJSON(CachedClosedJSON(parser),ctx)
    fn=TransformersPrefixAllowedTokensFn(SafeFastEnforcer(self.original.data,root,self.closed_index,ctx,self.closed_heads))
    width=[None]
    def allowed(batch_id,sent):
        if width[0] is None:width[0]=len(sent)
        ctx.generated_tokens=len(sent)-width[0]
        return fn(batch_id,sent)
    return allowed


class SafeDiscovery(ClosedDiscovery):
    def discover_batch(self,*args,**kwargs):
        rows=super().discover_batch(*args,**kwargs)
        for row in rows:
            row['output_contract']['numeric_prefix']='joint bounds/uniqueness completion; whitespace only after legal finished coordinate; no precision limit'
        return rows


class QwenSafeContractBundle(QwenFastClosedContractBundle):
    def __init__(self,config,*,device='cuda:0'):
        super().__init__(config,device=device)
        self.discovery.__class__=SafeDiscovery
        self.discovery.grammar.prefix=MethodType(safe_prefix,self.discovery.grammar)
