"""Load selected upstream definitions without importing incompatible model stacks.

Definitions are compiled from the pinned files, without editing their ASTs.
Upstream repositories and their licenses remain alongside the run artifacts.
"""
import ast
import hashlib
import logging
from pathlib import Path
import re
import time
import types


def source_definitions(path, names, namespace):
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('upstream definitions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}, dict(
        path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        definitions=list(names), ast_modified=False)


def causal_cache(repo):
    import torch
    definitions, receipt = source_definitions(
        Path(repo) / 'qwen2_5_vl/modeling_qwen2_5_vl_streaming_time.py',
        ['FOSSCache'], {'torch': torch})
    return definitions['FOSSCache'], receipt


def hermes_class(repo):
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
    repo = Path(repo)
    namespace = dict(torch=torch, Tuple=tuple, DynamicCache=DynamicCache,
                     apply_multimodal_rotary_pos_emb=apply_multimodal_rotary_pos_emb)
    # reindex_3d has no incompatible model imports beyond the supplied symbols.
    names = [node.name for node in ast.parse((repo / 'inference/reindex_3d.py').read_text()).body
             if isinstance(node, ast.FunctionDef)]
    _, reindex_receipt = source_definitions(repo / 'inference/reindex_3d.py', names, namespace)
    definitions, abstract_receipt = source_definitions(
        repo / 'inference/abstract_hermes.py', ['Abstract_Hermes'], {'torch': torch})
    namespace.update(F=F, re=re, time=time, logger=logging.getLogger('HERMES'),
                     Qwen2_5_VLForConditionalGeneration=Qwen2_5_VLForConditionalGeneration,
                     Abstract_Hermes=definitions['Abstract_Hermes'])
    definitions, receipt = source_definitions(
        repo / 'inference/qwenvl_hermes.py',
        ['get_qwen2_5_vl_position_ids', 'QwenVL_Hermes'], namespace)
    upstream = definitions['QwenVL_Hermes']

    class SpatialHermes(upstream):
        @property
        def language_model(self):
            # The pinned Spatial decoder is self.model (transformers 4.51).
            return self.model

        def _register_forward_hooks(self):
            def make_hook(index):
                def hook(module, args, kwargs):
                    if index in self._layer_position_ids:
                        positions = self._layer_position_ids[index]
                        kwargs['position_ids'] = positions
                        hidden = args[0] if args else kwargs['hidden_states']
                        # In 4.51 RoPE is prepared before decoder-layer hooks.
                        kwargs['position_embeddings'] = self.language_model.rotary_emb(hidden, positions)
                    return args, kwargs
                return hook
            for index, layer in enumerate(self.language_model.layers):
                self._hook_handles.append(layer.register_forward_pre_hook(make_hook(index), with_kwargs=True))

        @torch.inference_mode()
        def encode_features(self, features, grid, frame_count):
            self._ensure_dynamic_cache()
            offsets = self._get_next_global_offset_per_layer()
            positions = definitions['get_qwen2_5_vl_position_ids'](
                grid, features.shape[0], offset=offsets[0],
                vision_config=self.config.vision_config, sample_fps=self.sample_fps).to(self.device)
            self._layer_position_ids.clear()
            for index in range(self.num_layers):
                current = positions + offsets[index] - offsets[0]
                self._layer_position_ids[index] = self._build_position_ids_3d_for_vision(current, 1)
            output = self.language_model(
                inputs_embeds=features.unsqueeze(0), past_key_values=self.kv_cache,
                use_cache=True, return_dict=True,
                position_ids=self._build_position_ids_3d_for_vision(positions, 1))
            self.kv_cache = output.past_key_values
            for index in range(self.num_layers):
                self._append_position_ids_layer_explicit(index, positions + offsets[index] - offsets[0])
            self.last_encoded_frames = frame_count
            self.total_processed_frames += frame_count
            self._layer_position_ids.clear()

    return SpatialHermes, namespace, [reindex_receipt, abstract_receipt, receipt]


def attach_hermes(runtime, repo, *, kv_size, sample_fps):
    cls, namespace, receipts = hermes_class(repo)
    controller = cls.__new__(cls)
    controller.__dict__ = runtime.model.__dict__.copy()
    prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    initial_ids = runtime.processor.tokenizer(prompt, return_tensors='pt').input_ids.to(runtime.device)
    namespace['Abstract_Hermes'].__init__(controller, runtime.processor, initial_ids, kv_size)
    controller.streaming = False  # VSI-Bench questions observe the full video.
    controller.sample_fps = sample_fps
    controller.num_layers = runtime.model.config.num_hidden_layers
    controller.short_term_ratio, controller.long_term_ratio = 0.1, 0.3  # official load_model
    controller.short_term_threshold = int(controller.num_layers * controller.short_term_ratio)
    controller.long_term_threshold = int(controller.num_layers * (1 - controller.long_term_ratio))
    controller._position_ids_cache = [None] * controller.num_layers
    controller._layer_position_ids, controller._hook_handles = {}, []
    controller._mrope_section = namespace['_get_mrope_section'](controller.language_model)
    controller.total_processed_frames = 0
    controller._register_forward_hooks()
    return controller, receipts
