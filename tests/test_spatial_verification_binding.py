"""Protect exact D116 messages while retaining live image order checks."""
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_spatial_vsibench import verification_with_original_images
from sandbox_vlm.qa.d116_full import joint_messages
from sandbox_vlm.qa.artifact import pixel_sha256
from sandbox_vlm.qa.map_only import indexed_messages
from sandbox_vlm.qa.two_pass import verification_messages


def test_verifier_exact_serialized_content_and_original_images():
    diagram=Image.new('RGB',(20,16),'red');frame=Image.new('RGB',(18,16),'blue')
    base,images=joint_messages('Original input',True,[7],{},[frame],diagram)
    candidate=dict(status='ok',answer='A',raw_output='FINAL_ANSWER: A')
    hashes=[pixel_sha256(im) for im in images]
    before=indexed_messages(base,images,hashes)
    intended=verification_messages(before,candidate)
    actual=verification_with_original_images(base,images,candidate)
    blocks=[c for m in actual for c in m['content'] if c['type']=='image']
    assert all(block['image'] is image for block,image in zip(blocks,images))
    assert indexed_messages(actual,images,hashes)==intended
    assert indexed_messages(base,images,hashes)==before


def test_verifier_still_rejects_swapped_images():
    diagram=Image.new('RGB',(20,16),'red');frame=Image.new('RGB',(18,16),'blue')
    base,images=joint_messages('Original input',True,[7],{},[frame],diagram)
    with pytest.raises(ValueError,match='image order differs'):
        verification_with_original_images(base,list(reversed(images)),dict(status='ok',answer='A'))
