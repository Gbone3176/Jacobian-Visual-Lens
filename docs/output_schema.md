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

VG attention mode fields:

- `vg_attention_mode`: default `attribute_raw`;
- `q_type`: resolved from the mode;
- `attention_value_source`, `heatmap_value_source`, `colorbar_value_source`, and `top_patch_rank_source`: `raw_attention` or `normalized_attention`;
- `vg_attention_mode_contract`: the four allowed mappings:
  `attribute_raw -> attribute/raw_attention`,
  `attribute_normalized -> attribute/normalized_attention`,
  `localization_raw -> localization/raw_attention`,
  `localization_normalized -> localization/normalized_attention`.

Required validation invariants:

- one 24x24 token grid per image record;
- 576 patch rows;
- top patch ids sorted by `raw_attention` descending and `patch_id` ascending for ties;
- nonempty JLens and logit-lens token lists for mapped patches;
- no token interpolation in the attention map;
- no stale model or layer labels.
