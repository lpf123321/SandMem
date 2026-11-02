"""Frozen open-vocabulary discovery; optional D040/D041 raw-question focus.

The original question-independent prompt remains the default. Neither mode
extracts a normalized target field, uses a fixed label vocabulary, or assigns IDs.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    frame_id: int
    name: str
    short_description: str
    bbox_xyxy: tuple[float, float, float, float]  # original image pixels

    def to_dict(self) -> dict:
        return asdict(self)


def discovery_prompt(frame_id: int, *, question=None) -> str:
    original = (
        "Identify the distinct visible physical objects in this image. Use open-ended names; "
        "describe each in at most eight words to distinguish similar objects. "
        "Do not invent hidden objects. Return only one JSON object with this schema: "
        '{"frame_id": ' + str(frame_id) + ', "objects": ['
        '{"name": "object name", "description": "brief appearance", '
        '"bbox_2d": [x_min, y_min, x_max, y_max]}]}. '
        "Every object entry must include all three keys: name, description, bbox_2d. "
        "A name or description alone is not a valid object entry. "
        "Use 2D bounding-box coordinates normalized to 0..1000 over this entire image, "
        "with origin at the top left. Use one entry per visible object. "
        "List each physical instance exactly once. Never repeat an object or an identical bounding box. "
        "Use compact JSON without markdown or indentation. Once the distinct objects are listed, "
        "close the objects array and JSON object, then stop immediately. "
        "Return an empty objects list if no objects are identifiable."
    )
    if question is None:
        return original
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question-focused discovery requires the unchanged original question')
    focused = original.replace('Identify the distinct visible physical objects in this image.',
        'Identify every distinct visible physical object relevant to the main entity asked about '
        'in the original question. Ignore entities mentioned only in answer choices. '
        'List all visible instances of that entity, not one representative per category. '
        'Do not answer the question or count across frames.')
    return focused + '\nOriginal question (verbatim):\n' + question


def parse_discovery(raw: str, frame_id: int, image_size: tuple[int, int]) -> list[Detection]:
    """Reject malformed, truncated, nonfinite and out-of-image boxes without repair."""
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("discovery output is not complete JSON") from exc
    if not isinstance(payload, dict) or type(payload.get("frame_id")) is not int or payload["frame_id"] != frame_id:
        raise ValueError("discovery output has wrong frame identity")
    if not isinstance(payload.get("objects"), list):
        raise ValueError("discovery output must contain an objects list")
    height, width = image_size
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive")
    detections = []
    for obj in payload["objects"]:
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"].strip():
            raise ValueError("object must have an open-ended nonempty name")
        if not isinstance(obj.get("description"), str):
            raise ValueError("object description must be a string")
        box = obj.get("bbox_2d")
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
            raise ValueError("bbox_2d must contain four finite numbers")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("invalid normalized bounding box")
        detections.append(Detection(frame_id, obj["name"].strip(), obj["description"].strip(),
                                    (x1 * width / 1000, y1 * height / 1000,
                                     x2 * width / 1000, y2 * height / 1000)))
    return detections


def discovery_generation_kwargs(config: dict) -> dict:
    """Explicit deterministic generation settings, preserved in the run config."""
    cap = config["max_new_tokens"]
    penalty = config.get("repetition_penalty", 1.0)
    if type(cap) is not int or cap < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
    return {"do_sample": False, "max_new_tokens": cap, "repetition_penalty": float(penalty)}


class QwenDiscovery:
    """One frozen 2B model, single-image discovery calls with explicit cost records."""

    def __init__(self, config: dict, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if config["model_id"] != "Qwen/Qwen3-VL-2B-Instruct":
            raise ValueError("only the user-approved Qwen3-VL-2B-Instruct is allowed")
        if not config.get("revision"):
            raise ValueError("a fixed model/processor revision is required")
        self.config = dict(config)
        self.generation_kwargs = discovery_generation_kwargs(config)
        self.device = device
        kwargs = dict(revision=config["revision"], local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(config["model_id"], **kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model_id"], dtype=getattr(torch, config["dtype"]), **kwargs,
        ).to(device).eval().requires_grad_(False)

    def _repeat_guard(self, input_width, batch_size):
        setting = self.config.get('duplicate_stop_repeats', 0)
        if type(setting) is not int or setting not in (0, 3):
            raise ValueError('duplicate_stop_repeats must be disabled (0) or the approved 3')
        if not setting: return None, {}
        from transformers import StoppingCriteriaList
        from sandbox_vlm.perception.repetition import RepeatedDiscoveryStop
        guard = RepeatedDiscoveryStop(self.processor.tokenizer, input_width, batch_size)
        return guard, dict(stopping_criteria=StoppingCriteriaList([guard]))

    def discover(self, image_path: str | Path, frame_id: int, *, question=None) -> dict:
        import torch
        from PIL import Image
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            height, width = image.height, image.width
            prompt = discovery_prompt(frame_id, question=question)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt},
            ]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        visual_tokens = int((inputs["input_ids"] == self.model.config.image_token_id).sum().item())
        guard, extra = self._repeat_guard(input_tokens, 1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **self.generation_kwargs, **extra)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        output_ids = generated[0, input_tokens:]
        raw = self.processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        reached_cap = len(output_ids) >= self.config["max_new_tokens"] and int(output_ids[-1]) not in eos_ids
        result = {
            "frame_id": frame_id, "image_size_hw": [height, width], "prompt": prompt,
            "raw_output": raw, "status": "ok", "detections": [],
            "input_tokens": input_tokens, "visual_tokens": visual_tokens,
            "text_and_special_input_tokens": input_tokens - visual_tokens,
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "output_tokens": len(output_ids), "generation_seconds": elapsed,
            "reached_output_limit": reached_cap,
        }
        try:
            if guard is not None:
                from sandbox_vlm.perception.repetition import repeated_complete_record
                repeat = guard.stops[0] or repeated_complete_record(raw)
                result.update(duplicate_guard_check_seconds=guard.check_seconds, duplicate_stop=repeat)
                if repeat is not None: raise ValueError('repeated_complete_discovery_record')
            if reached_cap:
                raise ValueError("generation reached output limit without EOS")
            result["detections"] = [d.to_dict() for d in parse_discovery(raw, frame_id, (height, width))]
        except ValueError as exc:
            result.update(status="perception_failed", error=str(exc))
        return result

    def discover_batch(self, image_paths, frame_ids, *, question=None):
        """Independent single-image prompts, sharing a bounded generation batch.

        No image or text from one request is visible to another request. EOS and
        truncation are checked per sequence; padding is never parsed as output.
        generation_seconds is amortized batch time; latency is recorded separately.
        """
        import torch
        from PIL import Image
        maximum = self.config.get('max_batch_size', 4)
        if type(maximum) is not int or not 1 <= maximum <= 32:
            raise ValueError('discovery maximum batch size must be explicitly bounded at 32')
        if not 1 <= len(frame_ids) <= maximum or len(image_paths) != len(frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError('discovery batch exceeds explicit independent-frame bound')
        messages, sizes, prompts = [], [], []
        for path, fid in zip(image_paths, frame_ids):
            with Image.open(path) as opened: image = opened.convert('RGB')
            sizes.append((image.height, image.width)); prompts.append(discovery_prompt(fid, question=question))
            messages.append([{'role': 'user', 'content': [
                {'type': 'image', 'image': image}, {'type': 'text', 'text': prompts[-1]}]}])
        previous_padding = self.processor.tokenizer.padding_side
        processor_started = time.perf_counter()
        try:
            self.processor.tokenizer.padding_side = 'left'
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                padding=True, return_dict=True, return_tensors='pt').to(self.device)
        finally:
            self.processor.tokenizer.padding_side = previous_padding
        torch.cuda.synchronize()
        processor_seconds = time.perf_counter() - processor_started
        started = time.perf_counter()
        guard, extra = self._repeat_guard(inputs['input_ids'].shape[1], len(frame_ids))
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **self.generation_kwargs, **extra)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        width = inputs['input_ids'].shape[1]
        results = []
        for index, fid in enumerate(frame_ids):
            output = generated[index, width:].tolist()
            if guard is not None and guard.stops[index] is not None:
                output = output[:guard.stops[index]['output_tokens']]
            eos_pos = next((n for n, token in enumerate(output) if token in eos_ids), None)
            if eos_pos is not None: output = output[:eos_pos+1]
            raw = self.processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            cap = len(output) >= self.config['max_new_tokens'] and eos_pos is None
            ntokens = int(inputs['attention_mask'][index].sum().item())
            visual = int((inputs['input_ids'][index] == self.model.config.image_token_id).sum().item())
            row = dict(frame_id=fid, image_size_hw=list(sizes[index]), prompt=prompts[index], raw_output=raw,
                status='ok', detections=[], input_tokens=ntokens, visual_tokens=visual,
                text_and_special_input_tokens=ntokens-visual,
                image_grid_thw=inputs['image_grid_thw'][index:index+1].tolist(),
                output_tokens=len(output), generation_seconds=elapsed/len(frame_ids), reached_output_limit=cap,
                batch_frame_ids=list(frame_ids), batch_index=index, batch_padded_input_width=int(width),
                batch_generation_seconds=elapsed, generation_latency_seconds=elapsed,
                processor_seconds=processor_seconds/len(frame_ids), generation_invocations=int(index == 0),
                logical_single_image_requests=1)
            try:
                if guard is not None:
                    from sandbox_vlm.perception.repetition import repeated_complete_record
                    repeat = guard.stops[index] or repeated_complete_record(raw)
                    row.update(duplicate_guard_check_seconds=guard.check_seconds/len(frame_ids), duplicate_stop=repeat)
                    if repeat is not None: raise ValueError('repeated_complete_discovery_record')
                if cap: raise ValueError('generation reached output limit without EOS')
                row['detections'] = [d.to_dict() for d in parse_discovery(raw, fid, sizes[index])]
            except ValueError as exc: row.update(status='perception_failed', error=str(exc))
            results.append(row)
        return results
