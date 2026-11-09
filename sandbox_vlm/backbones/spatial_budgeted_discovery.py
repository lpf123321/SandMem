"""Online closure at complete-record boundaries under the original token cap.

No partial record is recovered or filled. Output is constrained before sampling.
The available budget and longest completed record determine whether another
record may start; no fixed object-count cap or answer information is used.
"""
import json
import re
import math
from decimal import Decimal,localcontext
from lmformatenforcer.characterlevelparser import CharacterLevelParser
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedJSON, BoundedUnion, BoundedGrammar, WordEnforcer, SpatialBoundedDiscovery


def number_prefix_feasible(prefix, lower, upper=1000., inclusive=True):
    """Canonical nonnegative decimal prefixes with some completion in bounds."""
    if not prefix or any(c not in '0123456789.' for c in prefix) or prefix.count('.')>1:
        return False
    if '.' in prefix:
        whole,fraction=prefix.split('.')
        if not whole:return False
        if len(whole)>1 and whole.startswith('0'):return False
        value=float(prefix+'0' if not fraction else prefix)
        if value>upper or (value==upper and not inclusive):return False
        if value>lower:return True
        maximum=min(upper,value+10.**(-min(len(fraction),309)))
        return maximum>lower
    if len(prefix)>1 and prefix.startswith('0'):return False
    value=int(prefix)
    for power in ([1] if prefix=='0' else [1,10,100,1000]):
        start,end=value*power,(value+1)*power
        if max(start,lower)<min(end,upper):return True
        if inclusive and start<=upper<end and upper>lower:return True
    return False


def unique_float_completion(prefix,forbidden,lower):
    """Some decimal continuation must parse to a NEW legal Python float.

    Once the decimal interval rounds entirely to an already emitted value,
    appending digits cannot escape that value. Reject that dead prefix before
    it is selected; no precision cutoff or numeric rounding is introduced.
    """
    if not forbidden:return True
    value=float(prefix+'0' if prefix.endswith('.') else prefix)
    if lower<value<=1000 and value not in forbidden:return True
    if '.' not in prefix:
        return value<1000
    with localcontext() as precision:
        precision.prec=max(40,len(prefix)+20)
        start=Decimal(prefix+'0' if prefix.endswith('.') else prefix)
        end=start+Decimal(1).scaleb(-len(prefix.split('.')[1]))
        candidate=float(start);maximum=float(end)
        for _ in range(len(forbidden)+3):
            if candidate>maximum or candidate>1000:return False
            if candidate>lower and candidate not in forbidden:
                previous=math.nextafter(candidate,-math.inf);following=math.nextafter(candidate,math.inf)
                low=(Decimal.from_float(previous)+Decimal.from_float(candidate))/2
                high=(Decimal.from_float(candidate)+Decimal.from_float(following))/2
                if max(start,low)<min(end,high):return True
            candidate=math.nextafter(candidate,math.inf)
    return False


class BudgetContext:
    def __init__(self,tokenizer,cap,frame_id):
        self.tokenizer,self.cap,self.frame_id=tokenizer,cap,frame_id
        self.generated_tokens=0;self.costs={};self.events={}

    def cost(self,text):
        if text not in self.costs:self.costs[text]=len(self.tokenizer.encode(text,add_special_tokens=False))
        return self.costs[text]

    def close_tail(self,text):
        if not text.endswith('}'):return None
        match=re.search(r'"objects"\s*:\s*\[',text)
        if match is None:return None
        decoder=json.JSONDecoder();position=match.end();costs=[]
        while position<len(text):
            while position<len(text) and text[position] in ' ,\n\r\t':position+=1
            start=position
            try:obj,position=decoder.raw_decode(text,position)
            except ValueError:return None
            if not isinstance(obj,dict):return None
            costs.append(self.cost(text[start:position]))
        if not costs:return None
        try:root=json.loads(text+']}')
        except ValueError:return None
        tail=']}' if 'frame_id' in root else '],"frame_id":'+str(self.frame_id)+'}'
        # One prospective token plus a comma for a new record and EOS are
        # charged. The longest actually completed record is the next-record
        # reserve; its length is not a tuned parameter or a physical count.
        available=self.cap-self.generated_tokens-1
        closing=self.cost(tail)+1
        reserve=max(costs)+closing+1
        if available<=reserve:
            self.events[text]=dict(trigger_output_tokens=self.generated_tokens,
                available_tokens=available,longest_record_tokens=max(costs),closing_tokens=closing,
                complete_records=len(costs),forced_syntax_tail=tail)
            return tail
        return None


class BudgetJSON(CharacterLevelParser):
    def __init__(self,base,context,tail=None):
        super().__init__(base.config);self.base,self.context,self.tail=base,context,tail

    @property
    def text(self):
        base=self.base
        while isinstance(base,BoundedUnion):base=base.parsers[0]
        return base.text

    def closure(self):
        if self.tail is not None:return self.tail
        if self.base.can_end():return ''
        # If every value is already present, only brackets/known source-frame
        # metadata remain. Reserve their character count plus EOS (not a tuned
        # token threshold). Never manufacture a missing name, description or box.
        short=['} ',']}','}]}',']}]}']
        short[0]='}'
        frame=',"frame_id":'+str(self.context.frame_id)+'}'
        tails=short+[prefix+frame for prefix in ('',']','}]',']}]')]
        remaining=self.context.cap-self.context.generated_tokens
        for tail in sorted(tails,key=len):
            if remaining>len(tail)+2:continue
            base=self.base
            try:
                for c in tail:
                    if c not in base.get_allowed_characters():raise ValueError('not a legal grammar completion')
                    base=base.add_character(c)
                if not base.can_end():continue
                text=self.text+tail
                from sandbox_vlm.perception.discovery import parse_discovery
                detections=parse_discovery(text,self.context.frame_id,(1000,1000))
                if not detections:continue
            except (ValueError,TypeError):continue
            self.context.events[self.text]=dict(trigger_output_tokens=self.context.generated_tokens,
                available_tokens=remaining,forced_syntax_tail=tail,complete_records=len(detections),
                reason='all field values already complete; reserve syntax characters and EOS',
                values_invented=False,closing_character_reserve=len(tail)+1)
            return tail
        return self.context.close_tail(self.text)

    def box_parts(self):
        base=self.base
        while isinstance(base,BoundedUnion):base=base.parsers[0]
        if base.current_key()[0]!='bbox_2d':return None
        match=re.search(r'"bbox_2d"\s*:\s*\[([^\[\]]*)$',self.text)
        return match.group(1).split(',') if match else None

    def get_allowed_characters(self):
        tail=self.closure()
        if tail is not None:return frozenset(tail[:1]).intersection(self.base.get_allowed_characters())
        allowed=self.base.get_allowed_characters();parts=self.box_parts()
        if parts is None or len(parts)>4:return allowed
        index=len(parts)-1;prefix=parts[-1].strip()
        lower=float(parts[index-2]) if index>=2 else -1.
        forbidden=set()
        if index==3:
            first_three=tuple(float(v) for v in parts[:3])
            for match in re.finditer(r'"bbox_2d"\s*:\s*(\[[^\[\]]*\])',self.text):
                try:box=json.loads(match.group(1))
                except ValueError:continue
                if len(box)==4 and tuple(box[:3])==first_three:forbidden.add(box[3])
        result=set()
        for c in allowed:
            if c in '-+eE':continue
            if c in '0123456789.':
                if number_prefix_feasible(prefix+c,lower,inclusive=index>=2) and (
                        not forbidden or unique_float_completion(prefix+c,forbidden,lower)):result.add(c)
            elif c in ',]' and prefix:
                if re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',prefix) is None:continue
                try:value=float(prefix)
                except ValueError:continue
                if lower<value<=1000 and (index>=2 or value<1000):result.add(c)
            elif c.isspace() and prefix and re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',prefix) is None:continue
            else:result.add(c)
        return frozenset(result)

    def add_character(self,c):
        tail=self.closure()
        if tail is not None:
            if not tail or c!=tail[0]:raise ValueError('only grammar closure fits current budget')
            tail=tail[1:]
        return BudgetJSON(self.base.add_character(c),self.context,tail)

    def can_end(self):return self.base.can_end()
    def cache_key(self):return None
    def shortcut_key(self):return self.base.shortcut_key() if self.closure() is None else None


class BudgetEnforcer(WordEnforcer):
    def __init__(self,data,parser,index,context):
        super().__init__(data,parser,index);self.context=context
    def _apply_new_characters(self,state,token_sequence):
        # Replay the token under the budget at which it was selected, then
        # evaluate its successor with the updated remaining budget. Otherwise
        # a comma allowed one step ago could be rejected during token replay.
        current=self.context.generated_tokens
        self.context.generated_tokens=max(0,current-1)
        try:return super()._apply_new_characters(state,token_sequence)
        finally:self.context.generated_tokens=current


class BudgetGrammar(BoundedGrammar):
    def __init__(self,original,cap,tokenizer):
        super().__init__(original);self.cap=cap;self.tokenizer=tokenizer;self.contexts=[]
    def prefix(self,parser):
        fid=parser.context.model_class.properties['frame_id'].enum[0]
        context=BudgetContext(self.tokenizer,self.cap,fid)
        self.contexts.append(context)
        root=BudgetJSON(BoundedJSON(parser),context)
        function=TransformersPrefixAllowedTokensFn(BudgetEnforcer(self.original.data,root,self.index,context))
        first_width=[None]
        def allowed(batch_id,sent):
            if first_width[0] is None:first_width[0]=len(sent)
            context.generated_tokens=len(sent)-first_width[0]
            return function(batch_id,sent)
        return allowed


class SpatialBudgetedDiscovery(SpatialBoundedDiscovery):
    def annotate(self,result,context):
        result['spatial_syntax'].update(policy='budget_closed_discovery_v3',
            closure='at complete-object boundary when remaining budget cannot reserve the longest completed record plus syntax/EOS',
            object_count_cap=None,original_output_cap=context.cap,
            closure_events_at_actual_prefix=[v for text,v in context.events.items() if result['raw_output'].startswith(text)],
            normalized_coordinate_syntax='canonical nonnegative decimals; original 0..1000 bounds and positive box dimensions',
            float_uniqueness='prune only prefixes having no distinct valid float completion; no precision cap',
            partial_record_recovery=False)
        return result
    def discover_batch(self,*args,**kwargs):
        self.grammar.contexts=[]
        results=super().discover_batch(*args,**kwargs)
        for result,context in zip(results,self.grammar.contexts):self.annotate(result,context)
        return results
    def discover(self,*args,**kwargs):
        self.grammar.contexts=[]
        result=super().discover(*args,**kwargs)
        return self.annotate(result,self.grammar.contexts[0])
