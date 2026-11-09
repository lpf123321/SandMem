"""Name-independent public aliases for the frozen method; no algorithm changes."""
from sandbox_vlm.memory.coverage import CoverageSandbox
from sandbox_vlm.memory.visual import BoundedVisualBackend
from sandbox_vlm.qa.d116_full import compose, draw, joint_messages
from sandbox_vlm.qa.two_pass import eligible, candidate_from_generation, verification_messages, decide


def create_memory(video_id, question, *, limit_bytes=64 * 1024 * 1024):
    return CoverageSandbox(video_id, question, retain_ambiguities=False,
                           memory_limit_bytes=limit_bytes)


__all__ = ['create_memory', 'BoundedVisualBackend', 'compose', 'draw', 'joint_messages',
           'eligible', 'candidate_from_generation', 'verification_messages', 'decide']


# Current continuous candidate; historical aliases above remain frozen.
from sandbox_vlm.continuous_model import ContinuousSandMem

def create_continuous_model(video_id, config):
    return ContinuousSandMem(video_id, config)

__all__.extend(['ContinuousSandMem', 'create_continuous_model'])
