"""Spatial fused tokens with the unmodified official CausalMem FOSSCache."""
from official_components import causal_cache


class SpatialCausalMem:
    def __init__(self, runtime, repo, *, budget=12000, sample_fps=1):
        import torch
        cache_class, self.source_receipt = causal_cache(repo)
        self.runtime = runtime
        self.sample_fps = sample_fps
        self.cache = cache_class(budget=budget, dim=runtime.model.config.hidden_size,
            k_max=64, decay=0.9, device=runtime.device, dtype=torch.bfloat16,
            max_new_basis=8, time_weight=0, time_power=1)
        self.token_count = 0
        self.pair_count = 0
        self.positions = torch.empty((3, 0), device=runtime.device, dtype=torch.long)
        self.position_token_ids = torch.empty((0,), device=runtime.device, dtype=torch.long)

    def add(self, features, grid):
        import torch
        t, h, w = grid
        h //= 2
        w //= 2
        pair_features = features.reshape(t, h * w, -1)
        for index, values in enumerate(pair_features):
            positions = torch.arange(self.token_count, self.token_count + h * w, device=features.device)
            self.cache.process_frame(values, frame_idx=self.pair_count, token_positions=positions)
            # Preserve original uncompressed Qwen2.5 temporal/spatial coordinates.
            temporal = torch.full((h * w,), int(self.pair_count * 2 / self.sample_fps *
                self.runtime.model.config.vision_config.tokens_per_second), device=features.device)
            height = torch.arange(h, device=features.device).repeat_interleave(w)
            width = torch.arange(w, device=features.device).repeat(h)
            candidates = torch.cat([self.position_token_ids, positions])
            coordinates = torch.cat([self.positions, torch.stack([temporal, height, width])], dim=1)
            selection = torch.searchsorted(candidates, self.cache.mem1_positions)
            if not torch.equal(candidates[selection], self.cache.mem1_positions):
                raise ValueError('retained token/position metadata mismatch')
            self.positions = coordinates.index_select(1, selection)
            self.position_token_ids = self.cache.mem1_positions.clone()
            self.token_count += h * w
            self.pair_count += 1

    def answer(self, question, *, max_new_tokens=16):
        import time
        import torch
        runtime = self.runtime
        tokenizer = runtime.processor.tokenizer
        prefix = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>'
        suffix = '<|vision_end|>\n' + question + '<|im_end|>\n<|im_start|>assistant\n'
        head = tokenizer(prefix, add_special_tokens=False, return_tensors='pt').input_ids.to(runtime.device)
        tail = tokenizer(suffix, add_special_tokens=False, return_tensors='pt').input_ids.to(runtime.device)
        memory, retained = self.cache._assemble_and_sort()
        selection = torch.searchsorted(self.position_token_ids, retained)
        positions = self.positions.index_select(1, selection)
        # Qwen's first visual temporal/spatial coordinate follows vision_start.
        positions = positions + head.shape[1]
        text_start = int(positions.max()) + 1
        head_positions = torch.arange(head.shape[1], device=runtime.device).expand(3, -1)
        tail_positions = torch.arange(text_start, text_start + tail.shape[1], device=runtime.device).expand(3, -1)
        all_positions = torch.cat([head_positions, positions, tail_positions], dim=1).unsqueeze(1)
        embeddings = torch.cat([runtime.model.get_input_embeddings()(head), memory.unsqueeze(0),
                                 runtime.model.get_input_embeddings()(tail)], dim=1)
        started = time.perf_counter()
        output_ids = []
        eos = runtime.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        with torch.inference_mode():
            output = runtime.model.model(inputs_embeds=embeddings, position_ids=all_positions,
                                         use_cache=True, return_dict=True)
            cache = output.past_key_values
            hidden = output.last_hidden_state[:, -1:]
            next_position = int(all_positions.max()) + 1
            for step in range(max_new_tokens):
                token = int(runtime.model.lm_head(hidden)[0, -1].argmax())
                output_ids.append(token)
                if token in eos or step + 1 == max_new_tokens:
                    break
                output = runtime.model.model(input_ids=torch.tensor([[token]], device=runtime.device),
                    position_ids=torch.full((3, 1, 1), next_position, device=runtime.device, dtype=torch.long),
                    past_key_values=cache, use_cache=True, return_dict=True)
                cache, hidden = output.past_key_values, output.last_hidden_state
                next_position += 1
        torch.cuda.synchronize()
        return dict(raw_output=tokenizer.decode(output_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False), output_token_ids=output_ids,
                    output_tokens=len(output_ids), seconds=time.perf_counter() - started,
                    retained_visual_tokens=len(memory), observed_visual_tokens=self.token_count,
                    retained_position_metadata_bytes=self.positions.numel() * self.positions.element_size()
                        + self.position_token_ids.numel() * self.position_token_ids.element_size(),
                    retained_original_token_positions=retained.cpu().tolist(),
                    text_prompt=prefix + '[retained Spatial fused video tokens]' + suffix,
                    input_text_token_ids=dict(prefix=head[0].tolist(), suffix=tail[0].tolist()),
                    reached_output_limit=len(output_ids) == max_new_tokens and output_ids[-1] not in eos)
