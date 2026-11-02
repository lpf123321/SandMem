"""Frozen Qwen3-VL-2B joint-image answering with measured processor token budgets."""
import time

from sandbox_vlm.qa.budget import check_budget


class QwenAnswerer:
    @classmethod
    def from_discovery(cls, config, discovery):
        """Reuse the same frozen weights/processor in a sequential full run."""
        if any(config[k] != discovery.config[k] for k in ('model_id', 'revision', 'dtype')):
            raise ValueError('perception and QA backbone mismatch')
        if config.get('temperature', 0) != 0:
            raise ValueError('QA requires greedy generation')
        instance = cls.__new__(cls)
        instance.config, instance.device = dict(config), discovery.device
        instance.model, instance.processor = discovery.model, discovery.processor
        instance.generation_attempts = 0
        return instance

    def __init__(self, config, *, device='cuda'):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if config['model_id'] != 'Qwen/Qwen3-VL-2B-Instruct' or not config.get('revision'):
            raise ValueError('only the pinned user-approved 2B multimodal checkpoint is allowed')
        if config.get('temperature', 0) != 0:
            raise ValueError('initial QA requires deterministic greedy generation')
        kwargs = dict(revision=config['revision'], local_files_only=True)
        self.config, self.device = dict(config), device
        self.generation_attempts = 0
        self.processor = AutoProcessor.from_pretrained(config['model_id'], **kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config['model_id'],
            dtype=getattr(torch, config['dtype']), **kwargs).to(device).eval().requires_grad_(False)

    def answer(self, messages, images, limits, *, format_constraint=None, note_boundary=None):
        import torch
        started = time.perf_counter()
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')
        input_tokens = int(inputs['input_ids'].shape[-1])
        visual_tokens = int((inputs['input_ids'] == self.model.config.image_token_id).sum().item())
        grid = inputs['image_grid_thw'].tolist()
        patch = self.processor.image_processor.patch_size
        processed_hw = [[r[1] * patch, r[2] * patch] for r in grid]
        context = int(self.model.config.text_config.max_position_embeddings)
        budget = check_budget(limits, image_sizes_hw=[[i.height, i.width] for i in images],
                              processed_image_sizes_hw=processed_hw, input_tokens=input_tokens,
                              visual_tokens=visual_tokens, context_limit=context)
        result = dict(status=budget['status'], budget=budget, image_grid_thw=grid,
                      processor_seconds=time.perf_counter() - started, model_calls=0,
                      generation_kwargs=dict(do_sample=False, max_new_tokens=limits.max_output_tokens))
        expected_visual = sum(r[0] * r[1] * r[2] // self.processor.image_processor.merge_size ** 2 for r in grid)
        if len(grid) != len(images) or visual_tokens != expected_visual:
            raise ValueError('processor image identity/token accounting mismatch')
        if budget['status'] != 'ok':
            return result  # No model generation, retries, truncation or silent image removal.
        generation_kwargs=dict(result['generation_kwargs'])
        boundary = None
        if note_boundary is not None:
            if format_constraint is not None:
                raise ValueError('do not combine independent decoding constraints')
            from transformers import LogitsProcessorList
            from sandbox_vlm.qa.note_boundary import NoteBoundary
            eos = self.model.generation_config.eos_token_id
            boundary = NoteBoundary(self.processor.tokenizer, input_tokens, limits.max_output_tokens,
                                    eos if isinstance(eos, list) else [eos], **note_boundary)
            generation_kwargs['logits_processor'] = LogitsProcessorList([boundary])
        if format_constraint is not None:
            from lmformatenforcer import RegexParser,CharacterLevelParserConfig
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
            import hashlib
            import json
            import importlib.metadata
            constrained_started=time.perf_counter()
            if not hasattr(self,'_format_tokenizer_data'):
                self._format_tokenizer_data=build_token_enforcer_tokenizer_data(self.processor.tokenizer)
                self._format_alphabet=''.join(sorted(set(''.join(t[1] for t in self._format_tokenizer_data.regular_tokens))))
            from sandbox_vlm.qa.grammar import build_answer_parser
            parser=build_answer_parser(format_constraint,self._format_alphabet)
            generation_kwargs['prefix_allowed_tokens_fn']=build_transformers_prefix_allowed_tokens_fn(self._format_tokenizer_data,parser)
            result['output_syntax_constraint']=dict(**format_constraint,
                pattern_sha256=hashlib.sha256(format_constraint['pattern'].encode()).hexdigest(),
                constraint_sha256=hashlib.sha256(json.dumps(format_constraint,sort_keys=True).encode()).hexdigest(),
                library='lm-format-enforcer',version=importlib.metadata.version('lm-format-enforcer'),
                construction_seconds=time.perf_counter()-constrained_started)
        inputs = inputs.to(self.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.generation_attempts += 1
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        torch.cuda.synchronize()
        output_ids = generated[0, input_tokens:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        reached_cap = len(output_ids) >= limits.max_output_tokens and int(output_ids[-1]) not in eos_ids
        result.update(raw_output=self.processor.decode(output_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False), output_tokens=len(output_ids),
            generation_seconds=time.perf_counter() - started, model_calls=1,
            reached_output_limit=reached_cap, status='answer_failed' if reached_cap else 'generated')
        if boundary is not None:
            result['note_boundary'] = boundary.report()
            result['output_token_ids'] = output_ids.tolist()
        return result
