"""D095 supplied-pixel limits and measured processor token accounting."""
from sandbox_vlm.qa.budget import check_budget


def check_unit_budget(limits, *, image_sizes_hw, processed_image_sizes_hw,
                      input_tokens, visual_tokens, context_limit):
    # The approved 384/256 sides describe supplied frame/crop images, not an
    # extra side cap on the frozen processor's internal minimum-size grid.
    result=check_budget(limits,image_sizes_hw=image_sizes_hw,
        input_tokens=input_tokens,visual_tokens=visual_tokens,context_limit=context_limit)
    if len(image_sizes_hw)!=2:
        result['reasons'].append('unit_requires_frame_and_exact_crop')
    elif max(image_sizes_hw[0])>384 or max(image_sizes_hw[1])>256:
        result['reasons'].append('supplied_unit_image_resolution_exceeded')
    result.update(processed_image_sizes_hw=processed_image_sizes_hw,
        image_side_limit_applies_to='supplied_images_before_frozen_processor',
        processed_side_limit=None)
    result['status']='budget_exceeded' if result['reasons'] else 'ok'
    return result
