"""Training-free long-video spatial understanding with a persistent-instance sandbox.

Modules:
  data        dataset loaders, frame mapping, config
  perception  MLLM object discovery, SAM segmentation, VGGT geometry (frozen)
  alignment   cross-viewpoint geometric alignment and instance association
  memory      unified sandbox (spatial extents, instances, viewpoints, ambiguity)
  qa          evidence selection, sandbox rendering, joint answering
  evaluation  official benchmark scoring
"""

__version__ = "0.0.1"
