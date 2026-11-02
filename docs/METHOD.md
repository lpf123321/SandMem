# Frozen method: D116 + instance-memory retention

The accepted method uses a frozen MLLM for question-conditioned video coverage, object discovery and answering, and existing frozen SAM/VGGT for segmentation, appearance and local geometry. Instance identities retain the approved existing rules. There is no pending-observation pool, VGGT track mechanism, task-label routing, count-specific solver or QA identity-revision loop.

## Memory and retrieval

Each instance stores its ID, name, complete description and representative appearance. Actual source-frame/region links are the evidence of observation. The original question is normalized into lexical terms. Any name overlap selects an anchor; description overlap is used only if no names match. If neither matches, all records are retained. Actual same-frame context from the supplied images is added, and selected records keep their full fields. The selected real-frame set and chronological order are preserved.

## Local 3D evidence

Reliable points inside actual segmentation masks produce estimated local AABBs with VGGT depth and cameras. The rendered diagram includes instance IDs, observed ranges and cameras. Each local coordinate reference remains independent and has arbitrary scale. This method does not claim a metrically calibrated global scene or verified full-video physical identities.

The original question, rendered 3D image, related complete records, actual observation provenance and all selected real frames are jointly supplied to the MLLM.

## Conditional answering

The gate is the existing sparse-input flag and the presence of actual observation links. Eligible inputs first produce an independent draft from the original question, real frames and original compact text. A second call verifies the candidate with the joint sandbox evidence. A change to a valid draft must cite a supplied source-frame ID; failed verification or an uncited change retains a valid draft. A valid verifier can answer when the draft failed. Citation checks provenance rather than visual truth. Other inputs use a joint single pass. The same frozen backbone is shared across roles.

## Capacity handling

Only overflow triggers compaction. For every instance, retain the minimum and maximum actual source-frame IDs, deduplicating the single-observation case. If the complete structure still exceeds the limit, evict by `(last_actual_observation_frame, instance_id)` ascending. This is observation recency, not QA-access recency.

Remove the corresponding descriptions, appearance, geometry, observation links, dedup keys, mask entries and local correspondence endpoints. Shared viewpoints survive while another retained instance uses them. Keep surviving instance IDs and source-region slots stable, and persist a monotonic allocator. Do not recover forgotten records through historical caches or disk paging.

Reachable CPU structure, complete serialized state and backend replacement overlap are accounted for. Mandatory independent auxiliary state remains charged. If even an empty instance set cannot fit the mandatory structure, explicitly reject the transaction and preserve the prior committed state. This is not a claim that model weights, all transient tensors or total process RSS fit in 64 MiB.

## Source identity

`provenance/frozen_files.json` binds each shipped original source/config/test file to the accepted `sandmem_20261102_01` snapshot. `provenance/d116_comparison.json` records the difference from D118's frozen runtime. The nine changed D116 core files and new retention module implement capacity handling and its stable-ID/QA compatibility. Other differences are isolated Spatial migration adapters, not changes to the active D116 answering rule.
