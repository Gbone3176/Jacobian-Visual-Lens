# Visualization Contract

The attention map is not a smooth heatmap. It is a patch-level view of a 24x24 image-token grid.

Default VG mode:

- `vg_attention_mode = attribute_raw`
- `q_type = attribute`
- `value_source = raw_attention`

Allowed VG modes:

| mode | q_type | value_source |
| --- | --- | --- |
| `attribute_raw` | `attribute` | `raw_attention` |
| `attribute_normalized` | `attribute` | `normalized_attention` |
| `localization_raw` | `localization` | `raw_attention` |
| `localization_normalized` | `localization` | `normalized_attention` |

Contract:

- `attention_map_display_mode = patch_grid_image_aspect`
- `attention_map_alignment = q_type_aligned`
- `logical_patch_grid = [24, 24]`
- `attention_map_token_interpolation = false`
- `attention_map_patch_blocks_independent = true`
- colormap: viridis
- color scaling: display-only per-map minmax
- image aspect: matches the input image size

Each token patch is drawn as one solid rectangle:

```text
x0 = col * width / 24
x1 = (col + 1) * width / 24
y0 = row * height / 24
y1 = (row + 1) * height / 24
```

Optional bounding boxes are displayed in the same image-relative coordinate system.
