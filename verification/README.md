# Release verification

Verified on 2026-11-03 from this independent checkout.

- **147 original project files** (118 core source files, 16 test files, 9 reference scripts and 4 configurations) match the accepted `sandmem_20261102_01` snapshot byte-for-byte.
- **101 CPU tests passed** in this checkout, including capacity overflow, first/last retention, observation-recency eviction, stable IDs, checkpoint restoration, source-region slots, stale-cache exclusion, two-pass answering and input contracts. Dependencies were taken from the existing tested interpreter; imports of the method itself resolve to this checkout.
- **294/294 D116 input-equivalence replays passed**: native instance payloads, rendered sandbox pixel hashes, and complete model messages match the original accepted D116 inputs. The old artifact workspace was read-only; no model calls or new benchmark scores were produced.
- The packaged baseline configuration is the historical Qwen reference. The isolated Spatial adapters are not activated as a completed full Spatial release.

Machine-readable records:

- `release_check.json`: frozen-file and dependency hashes, plus module-origin checks.
- `d116_input_equivalence.json`: all 294 per-query equivalence checks.
- `regression_tests.txt`: the final complete CPU test summary.
- `reference_summary.json`: aggregate historical Qwen scores, clearly labeled as predating the new capacity mechanism.

The 294-query replay validates the unchanged D116 path. Capacity-triggered behavior is additionally covered by the synthetic regression cases; the replay is not a new accuracy evaluation of memory eviction or a claim of cold-start 64 MiB certification.

Packaging adds only the name-independent API aliases, verification tools, documentation and provenance files. The original method modules have not been rewritten or renamed. Historical source modules that are not the final D116 entry path remain for import compatibility and are not promoted as an alternative final method.
