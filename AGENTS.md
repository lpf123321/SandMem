# Repository instructions

- The provisional display name is in `project.json`; keep the internal `sandbox_vlm` namespace independent of that name.
- The frozen method is D116 plus first/last observation compaction and oldest-observed instance eviction. Verify original source hashes with `python tools/verify_release.py` before changing method files.
- Preserve actual 3D image input, actual source-frame provenance, full selected record fields, streaming causality and the existing two-pass gate. Do not add pending pools, count solvers, new identity rules or implicit model substitutions.
- `configs/reference_qwen.json` describes the historical Qwen reference. Spatial adapters are isolated migration work and must not be labeled a completed full-model migration or scored result.
- Keep data, weights, credentials, environments and generated experiment artifacts outside Git. Preserve third-party source notices.
- Do not overwrite original reference workspaces or historical results. Model runs and changes to comparison protocols need their own explicit scope; packaging verification is CPU-only.
