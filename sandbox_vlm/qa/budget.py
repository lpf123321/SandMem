"""Joint sandbox-image and real-frame accounting, before model generation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class QABudget:
    max_images: int
    max_image_side: int
    max_visual_tokens: int
    max_text_tokens: int
    max_output_tokens: int

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in vars(self).values()):
            raise ValueError('QA limits must be explicit positive integers')


def check_budget(limits, *, image_sizes_hw, input_tokens, visual_tokens, context_limit,
                 processed_image_sizes_hw=None):
    if (type(input_tokens) is not int or type(visual_tokens) is not int
            or input_tokens < 0 or not 0 <= visual_tokens <= input_tokens
            or type(context_limit) is not int or context_limit < 1):
        raise ValueError('invalid measured token counts/context limit')
    text_tokens = input_tokens - visual_tokens
    reasons = []
    if len(image_sizes_hw) > limits.max_images:
        reasons.append('combined_image_count_exceeded')
    sizes = list(image_sizes_hw) + list(processed_image_sizes_hw or [])
    if any(len(size) != 2 or min(size) < 1 for size in sizes):
        raise ValueError('invalid measured image dimensions')
    if any(max(size) > limits.max_image_side for size in sizes):
        reasons.append('image_resolution_exceeded')
    if visual_tokens > limits.max_visual_tokens:
        reasons.append('combined_visual_tokens_exceeded')
    if text_tokens > limits.max_text_tokens:
        reasons.append('text_input_tokens_exceeded')
    if input_tokens + limits.max_output_tokens > context_limit:
        reasons.append('context_limit_exceeded')
    return dict(status='budget_exceeded' if reasons else 'ok', reasons=reasons,
                combined_images=len(image_sizes_hw), input_image_sizes_hw=image_sizes_hw,
                processed_image_sizes_hw=processed_image_sizes_hw,
                input_tokens=input_tokens, visual_tokens=visual_tokens,
                text_and_special_input_tokens=text_tokens,
                reserved_output_tokens=limits.max_output_tokens, model_context_limit=context_limit,
                limits=vars(limits))
