# Output Schema

Each run directory contains:

- `run_meta.json`: run-level metadata and model/lens/prompt fields.
- `attention_records.jsonl`: one row per record.
- `patch_attention_values.jsonl`: 576 rows per record, with `patch_id = patch_row * 24 + patch_col`.
- `jvlens_mapping.jsonl`: top attention patches with JLens and logit-lens token lists.
- `records/<record_id>/attention_map.json`: sidecar for attention-map rendering.
- `records/<record_id>/record_readouts.json`: record-level top patches and token readouts.
- `records/<record_id>/index.html`: static record page.
- `index.html`: run index page.
- `validation_summary.json`: machine-readable validation result.
- `static_share.zip`: portable static report.

Required validation invariants:

- one 24x24 token grid per image record;
- 576 patch rows;
- top patch ids sorted by `raw_attention` descending and `patch_id` ascending for ties;
- nonempty JLens and logit-lens token lists for mapped patches;
- no token interpolation in the attention map;
- no stale model or layer labels.
