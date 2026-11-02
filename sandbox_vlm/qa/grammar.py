"""Generation-time JSON/citation syntax only; never repairs or chooses answers."""
import json
import re


def unique_id_array(ids):
    """Quadratic regex for all sorted subsets, including [] (not 2**n literals)."""
    ids=sorted(set(ids))
    if any(type(i) is not int or i<0 for i in ids):
        raise ValueError('citation IDs must be nonnegative integers')
    alternatives=[]
    for index,last in enumerate(ids):
        alternatives.append(''.join(f'(?:{prior},)?' for prior in ids[:index])+str(last))
    return r'\[(?:'+('|'.join(alternatives))+r')?\]' if ids else r'\[\]'


def answer_grammar(metadata):
    selected=set(metadata['selection']['instance_ids'])
    # A cited identity must have an actual link to an image supplied this time.
    eligible=[entry['instance_id'] for entry in metadata['legend']
              if entry['instance_id'] in selected and entry['selected_actual_frame_ids']]
    frames=metadata['selection']['frame_ids']
    pending=[n['local_observation_id'] for n in metadata.get('unresolved_notes', [])]
    instance_array=unique_id_array(eligible);frame_array=unique_id_array(frames)
    json_string=r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})+"'
    requests=r'\[\]'
    if pending and selected:
        oid='(?:'+'|'.join(re.escape(json.dumps(p,ensure_ascii=False)) for p in sorted(set(pending)))+')'
        gid='(?:'+'|'.join(str(gid) for gid in sorted(selected))+')'
        item=r'\{"local_observation_id":'+oid+r',"instance_id":'+gid+r'\}'
        requests=r'\[(?:'+item+r'(?:,'+item+r'){0,'+str(len(pending)-1)+r'})?\]'
    pattern=(r'\{"answer":'+json_string+r',"used_instance_ids":'+instance_array
        +r',"used_frame_ids":'+frame_array+r',"match_requests":'+requests+r'\}')
    return dict(type='json_unique_citations',pattern=pattern,
        prefix='{"answer":',suffix_pattern=r',"used_instance_ids":'+instance_array+r',"used_frame_ids":'+frame_array+r',"match_requests":'+requests+r'\}',
        eligible_instance_ids=sorted(eligible),allowed_frame_ids=sorted(set(frames)),
        instance_frame_support={str(e['instance_id']):e['selected_actual_frame_ids'] for e in metadata['legend'] if e['instance_id'] in eligible},
        instance_array_pattern=instance_array,requests_pattern=requests,
        canonical_key_order=True,canonical_citation_order=True,
        semantic_answer_constrained=False,repair_or_retry=False)


def build_answer_parser(constraint,alphabet):
    """Use the library's cached JSON-string path instead of regex free-text traversal."""
    from lmformatenforcer import JsonSchemaParser,RegexParser,CharacterLevelParserConfig
    from lmformatenforcer.characterlevelparser import CharacterLevelParser,StringParser
    config=CharacterLevelParserConfig(alphabet=alphabet)
    linked_parser_cache={}
    class FrameCitations(CharacterLevelParser):
        """Sorted subsets with support checks, without enumerating 2**N subsets."""
        def __init__(self, chosen_instances, text=''):
            super().__init__(config)
            self.chosen_instances=tuple(chosen_instances);self.text=text
        def options(self):
            prefix=',"used_frame_ids":['
            if not self.text.startswith(prefix):
                return [prefix] if prefix.startswith(self.text) else []
            body=self.text[len(prefix):]
            if body.endswith(']'):return []
            parts=body.split(',');done=[int(x) for x in parts[:-1]]
            current=parts[-1];last=done[-1] if done else -1
            frames=[f for f in constraint['allowed_frame_ids'] if f>last]
            def supported(selected):
                return all(set(selected)&set(constraint['instance_frame_support'][str(g)])
                           for g in self.chosen_instances)
            options=[]
            if not body and supported(done):options.append(self.text+']')
            start=prefix+(''.join(str(f)+',' for f in done))
            for fid in frames:
                if not str(fid).startswith(current):continue
                selected=done+[fid]
                future=[f for f in frames if f>fid]
                # A comma is possible only when some completion can support IDs.
                if future and supported(selected+future):options.append(start+str(fid)+',')
                if supported(selected):options.append(start+str(fid)+']')
            return options
        def add_character(self,character):
            if character not in self.get_allowed_characters():raise ValueError('invalid linked frame citation')
            return FrameCitations(self.chosen_instances,self.text+character)
        def get_allowed_characters(self):
            return ''.join(sorted({x[len(self.text)] for x in self.options() if len(x)>len(self.text)}))
        def can_end(self):return self.text.endswith(']')
        def cache_key(self):return ('bounded_frame_citations',self.chosen_instances,self.text)
    class LinkedTail(CharacterLevelParser):
        def __init__(self,phase='instances',body='',parser=None):
            super().__init__(config);self.phase=phase;self.body=body
            self.parser=parser or RegexParser(r',"used_instance_ids":'+constraint['instance_array_pattern'],config=config)
        def add_character(self,character):
            import itertools
            parser=self.parser.add_character(character);body=self.body+character
            if self.phase=='instances' and character==']' and parser.can_end():
                chosen=json.loads(body.split(':',1)[1])
                frames=constraint['allowed_frame_ids']
                if len(frames)>31:raise ValueError('citation subset grammar exceeds bounded image budget')
                if len(frames)>12:
                    return LinkedTail('frames','',FrameCitations(chosen))
                subsets=[]
                for n in range(len(frames)+1):
                    for subset in itertools.combinations(frames,n):
                        if all(set(subset)&set(constraint['instance_frame_support'][str(gid)]) for gid in chosen):
                            subsets.append(re.escape(json.dumps(subset,separators=(',',':'))))
                key=tuple(chosen)
                if key not in linked_parser_cache:
                    pattern=r',"used_frame_ids":(?:'+'|'.join(subsets)+')'
                    linked_parser_cache[key]=RegexParser(pattern,config=config)
                return LinkedTail('frames','',linked_parser_cache[key])
            if self.phase=='frames' and character==']' and parser.can_end():
                if 'requests' not in linked_parser_cache:
                    linked_parser_cache['requests']=RegexParser(r',"match_requests":'+constraint['requests_pattern']+r'\}',config=config)
                return LinkedTail('requests','',linked_parser_cache['requests'])
            return LinkedTail(self.phase,body,parser)
        def get_allowed_characters(self):return self.parser.get_allowed_characters()
        def can_end(self):return self.phase=='requests' and self.parser.can_end()
        def cache_key(self):return ('linked_citations',self.phase,self.body,self.parser.cache_key())
    class StreamingSequence(CharacterLevelParser):
        def __init__(self,parsers):
            super().__init__(config);self.parsers=tuple(parsers)
        def add_character(self,character):
            for index,parser in enumerate(self.parsers):
                if character in parser.get_allowed_characters():
                    return StreamingSequence((parser.add_character(character),*self.parsers[index+1:]))
                if not parser.can_end():break
            raise ValueError('invalid structured-output character')
        def get_allowed_characters(self):
            allowed=[]
            for parser in self.parsers:
                allowed.append(parser.get_allowed_characters())
                if not parser.can_end():break
            return ''.join(allowed)
        def can_end(self):
            return all(p.can_end() for p in self.parsers)
        def shortcut_key(self):
            return self.parsers[0].shortcut_key() if not self.parsers[0].can_end() else None
        def cache_key(self):
            key=self.parsers[0].cache_key()
            return ('answer_sequence',len(self.parsers),key) if key is not None else None
    return StreamingSequence([StringParser(constraint['prefix']),
        JsonSchemaParser({'type':'string','minLength':1},config=config),
        LinkedTail()])
