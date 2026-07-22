#!/usr/bin/env python3
"""JVLens CLI: static validation, fixture demo, and guarded real run-single bridge."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


JVLENS_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = JVLENS_ROOT.parent
PROJECT_ROOT = ADAPTER_ROOT.parent

DEFAULT_LENS_PATH = (
    JVLENS_ROOT
    / "weights/huatuo_jimage_lens_layer16_n50.pt"
)
DEFAULT_LENS_META = DEFAULT_LENS_PATH.with_suffix(".json")
DEFAULT_LENS_SHA256 = "702c9a99d54e19d3b759e9238edcabc7696f7e07246c018a39c0035cb783fc5f"
DEFAULT_MODEL_PATH = "/path/to/HuatuoGPT-Vision-7B"
DEFAULT_SUPPORT_REPO = JVLENS_ROOT / "third_party/HuatuoGPT-Vision"
DEFAULT_RUNTIME_ROOT = JVLENS_ROOT / "runtime_adapters/HuatuoGPT-Vision"
DEFAULT_VG_ATTENTION_MODE = "attribute_raw"
VG_ATTENTION_MODE_MAP = {
    "attribute_raw": {"q_type": "attribute", "value_source": "raw_attention"},
    "attribute_normalized": {"q_type": "attribute", "value_source": "normalized_attention"},
    "localization_raw": {"q_type": "localization", "value_source": "raw_attention"},
    "localization_normalized": {"q_type": "localization", "value_source": "normalized_attention"},
}
VG_ATTENTION_MODE_CHOICES = tuple(VG_ATTENTION_MODE_MAP)


def resolve_vg_attention_mode(vg_attention_mode: str | None, legacy_q_type: str | None = None) -> dict[str, str]:
    if vg_attention_mode is None:
        if legacy_q_type == "localization":
            mode = "localization_raw"
        else:
            mode = DEFAULT_VG_ATTENTION_MODE
    else:
        mode = vg_attention_mode
    if mode not in VG_ATTENTION_MODE_MAP:
        allowed = ", ".join(VG_ATTENTION_MODE_CHOICES)
        raise ValueError(f"invalid vg attention mode {mode!r}; expected one of: {allowed}")
    contract = {"vg_attention_mode": mode, **VG_ATTENTION_MODE_MAP[mode]}
    if legacy_q_type in {"attribute", "localization"} and legacy_q_type != contract["q_type"]:
        raise ValueError(
            f"--q-type={legacy_q_type} conflicts with --vg-attention-mode={mode} "
            f"(q_type={contract['q_type']})"
        )
    return contract


def apply_vg_attention_contract(args: argparse.Namespace) -> dict[str, str]:
    contract = resolve_vg_attention_mode(
        getattr(args, "vg_attention_mode", None),
        getattr(args, "q_type", None),
    )
    args.vg_attention_mode = contract["vg_attention_mode"]
    args.q_type = contract["q_type"]
    args.attention_value_source = contract["value_source"]
    return contract


def attention_values_for_source(raw_attention: np.ndarray, value_source: str) -> np.ndarray:
    if value_source == "raw_attention":
        return np.asarray(raw_attention, dtype=np.float32)
    if value_source == "normalized_attention":
        return minmax01(raw_attention)[0]
    raise ValueError(f"unsupported attention value source: {value_source}")


def attention_map_filename(value_source: str) -> str:
    label = "raw" if value_source == "raw_attention" else "normalized"
    return f"attention_{label}_patchgrid_image_aspect.png"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def readout_path_text(source_layer: int) -> str:
    return (
        f"Huatuo image-token residual at layer {int(source_layer)} -> "
        "prefix_n50 fitted JacobianLens transport -> model final norm -> lm_head -> top-k token readout"
    )


def safe_slug(text: str, fallback: str = "record") -> str:
    keep = []
    for char in text.lower():
        if char.isalnum():
            keep.append(char)
        elif char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    slug = "".join(keep).strip("._-")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:120] or fallback


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_validation_log(run_root: Path, summary: dict[str, Any], event: str) -> None:
    with (run_root / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "status": summary["status"],
                    "event": event,
                    "validated_at_utc": summary["validated_at_utc"],
                    "errors": summary["errors"],
                    "zip_entries": summary["zip_entries"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def minmax01(attention: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(attention, dtype=np.float32)
    if arr.shape != (24, 24):
        raise ValueError(f"expected attention shape (24, 24), got {arr.shape}")
    if not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=np.float32), {
            "source_min": None,
            "source_max": None,
            "constant_or_invalid": True,
        }
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        arr01 = (arr - lo) / (hi - lo)
        arr01 = np.nan_to_num(arr01, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        constant = False
    else:
        arr01 = np.zeros_like(arr, dtype=np.float32)
        constant = True
    return arr01, {"source_min": lo, "source_max": hi, "constant_or_invalid": constant}


def patch_boundaries(size: int, grid: int = 24) -> list[int]:
    if size < grid:
        raise ValueError(f"image size {size} is too small for {grid} patch blocks")
    return [(idx * size) // grid for idx in range(grid)] + [size]


def patch_xyxy(row: int, col: int, width: int, height: int) -> list[float]:
    return [
        col * width / 24.0,
        row * height / 24.0,
        (col + 1) * width / 24.0,
        (row + 1) * height / 24.0,
    ]


def image_aspect_attention_map(attention: np.ndarray, image_size: list[int]) -> tuple[Image.Image, dict[str, Any]]:
    arr01, stats = minmax01(attention)
    width, height = int(image_size[0]), int(image_size[1])
    xs = patch_boundaries(width, 24)
    ys = patch_boundaries(height, 24)
    colors = colormaps["viridis"](arr01, bytes=True)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for row in range(24):
        y0, y1 = ys[row], ys[row + 1]
        for col in range(24):
            x0, x1 = xs[col], xs[col + 1]
            canvas[y0:y1, x0:x1, :] = colors[row, col]
    stats.update(
        {
            "attention_map_display_mode": "patch_grid_image_aspect",
            "logical_patch_grid": [24, 24],
            "patch_boundary_rule": "floor(i*size/24) with final boundary=size",
            "patch_boundary_x": xs,
            "patch_boundary_y": ys,
            "attention_map_colormap": "viridis",
            "attention_map_display_minmax_only": True,
            "attention_map_interpolation": "none_explicit_patch_blocks",
            "attention_map_smoothing": False,
            "attention_map_token_interpolation": False,
            "attention_map_patch_blocks_independent": True,
            "attention_map_colorbar": "viridis_0_1_per_map_minmax",
        }
    )
    return Image.fromarray(canvas, mode="RGBA"), stats


def patch_blocks_independent(path: Path, image_size: list[int], xs: list[int], ys: list[int]) -> tuple[bool, str]:
    with Image.open(path) as image:
        if list(image.size) != [int(image_size[0]), int(image_size[1])]:
            return False, f"map size {list(image.size)} != image size {image_size}"
        arr = np.asarray(image.convert("RGBA"))
    if len(xs) != 25 or len(ys) != 25:
        return False, "boundary arrays must have 25 entries"
    for row in range(24):
        y0, y1 = int(ys[row]), int(ys[row + 1])
        for col in range(24):
            x0, x1 = int(xs[col]), int(xs[col + 1])
            block = arr[y0:y1, x0:x1, :]
            if block.size == 0:
                return False, f"empty block row={row} col={col}"
            if not np.all(block == block[0, 0, :]):
                return False, f"non-constant block row={row} col={col}"
    return True, "PASS"


def top_patch_rows(
    attention: np.ndarray,
    image_size: list[int],
    top_k: int,
    value_source: str = "raw_attention",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arr = np.asarray(attention, dtype=np.float32)
    arr01, _ = minmax01(arr)
    raw_order = sorted(range(576), key=lambda idx: (-float(arr.reshape(-1)[idx]), idx))
    normalized_order = sorted(range(576), key=lambda idx: (-float(arr01.reshape(-1)[idx]), idx))
    selected_values = arr if value_source == "raw_attention" else arr01
    selected_order = sorted(range(576), key=lambda idx: (-float(selected_values.reshape(-1)[idx]), idx))
    raw_rank = {idx: rank + 1 for rank, idx in enumerate(raw_order)}
    normalized_rank = {idx: rank + 1 for rank, idx in enumerate(normalized_order)}
    rows: list[dict[str, Any]] = []
    for patch_id in range(576):
        row, col = divmod(patch_id, 24)
        heatmap_value = float(selected_values[row, col])
        rows.append(
            {
                "patch_id": patch_id,
                "patch_row": row,
                "patch_col": col,
                "token_grid": [24, 24],
                "image_patch_xyxy": patch_xyxy(row, col, int(image_size[0]), int(image_size[1])),
                "raw_attention": float(arr[row, col]),
                "normalized_attention": float(arr01[row, col]),
                "heatmap_value": heatmap_value,
                "heatmap_value_source": value_source,
                "raw_rank_desc": raw_rank[patch_id],
                "normalized_rank_desc": normalized_rank[patch_id],
            }
        )
    top_rows = [dict(rows[idx], rank=rank + 1) for rank, idx in enumerate(selected_order[:top_k])]
    return rows, top_rows


def fake_tokens(prefix: str, patch_id: int, top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": idx + 1,
            "token_id": 1000 + patch_id * 17 + idx,
            "text": f"{prefix}_{patch_id}_{idx + 1}",
            "score": round(1.0 / (idx + 1), 6),
        }
        for idx in range(top_k)
    ]


def pct_box(xyxy: list[float], image_size: list[int]) -> dict[str, float]:
    width, height = float(image_size[0]), float(image_size[1])
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    return {
        "left": 100.0 * x0 / width,
        "top": 100.0 * y0 / height,
        "width": 100.0 * (x1 - x0) / width,
        "height": 100.0 * (y1 - y0) / height,
    }


def bbox_percent_xyxy(bbox: list[float], image_size: list[int]) -> list[float]:
    width, height = float(image_size[0]), float(image_size[1])
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [100.0 * x0 / width, 100.0 * y0 / height, 100.0 * x1 / width, 100.0 * y1 / height]


def parse_bbox(value: str | None, image_size: list[int] | None = None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must have four comma-separated xyxy values")
    bbox = [float(part) for part in parts]
    x0, y0, x1, y1 = bbox
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"invalid xyxy bbox with non-positive extent: {bbox}")
    if image_size is not None:
        width, height = float(image_size[0]), float(image_size[1])
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError(f"bbox {bbox} is outside image bounds {image_size}")
    return bbox


def copy_input_image(src: Path, dst_dir: Path) -> tuple[str, list[int], str]:
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"image path does not exist: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".png"
    dst = dst_dir / f"input_image{suffix}"
    shutil.copy2(src, dst)
    with Image.open(dst) as image:
        image.verify()
    with Image.open(dst) as image:
        size = list(image.size)
    return dst.name, size, sha256_file(dst)


def default_run_single_out_dir() -> Path:
    return JVLENS_ROOT / "experiment" / ("run_single_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime()))


def render_bbox(meta: dict[str, Any], css_class: str) -> str:
    boxes = []
    for idx, box in enumerate(meta.get("bbox_percent_xyxy") or [], start=1):
        left, top, right, bottom = [float(v) for v in box]
        boxes.append(
            '<div class="bbox {css}" style="left:{left:.6f}%;top:{top:.6f}%;'
            'width:{width:.6f}%;height:{height:.6f}%"><span>B{idx}</span></div>'.format(
                css=html.escape(css_class, quote=True),
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
                idx=idx,
            )
        )
    return "".join(boxes)


def render_record_html(record: dict[str, Any], meta: dict[str, Any], mapping_rows: list[dict[str, Any]]) -> str:
    patch_buttons = []
    table_rows = []
    fixture_note = (
        "Fixture tokens are synthetic and marked in JSON."
        if meta.get("fixture")
        else "JLens and logit-lens tokens are real model readouts for the selected top-attention patches."
    )
    for row in mapping_rows:
        box = pct_box(row["image_patch_xyxy"], meta["image_size"])
        patch_buttons.append(
            '<button class="patch" style="left:{left:.6f}%;top:{top:.6f}%;width:{width:.6f}%;'
            'height:{height:.6f}%" title="rank {rank} patch {pid}"><span>{rank}</span></button>'.format(
                left=box["left"],
                top=box["top"],
                width=box["width"],
                height=box["height"],
                rank=row["rank"],
                pid=row["patch_id"],
            )
        )
        j_tokens = ", ".join(html.escape(tok["text"]) for tok in row["jlens_top"])
        l_tokens = ", ".join(html.escape(tok["text"]) for tok in row["logit_lens_top"])
        table_rows.append(
            "<tr>"
            f"<td>{row['rank']}</td><td>{row['patch_id']}</td><td>{row['patch_row']},{row['patch_col']}</td>"
            f"<td>{row['raw_attention']:.8g}</td><td>{row['normalized_attention']:.8g}</td>"
            f"<td>{j_tokens}</td><td>{l_tokens}</td>"
            "</tr>"
        )
    source_bbox = render_bbox(meta, "source-bbox")
    attention_bbox = render_bbox(meta, "attention-bbox")
    metadata = html.escape(json.dumps(meta, indent=2, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JVLens {html.escape(record['record_id'])}</title>
<style>
body {{ margin:22px; font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#202124; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
.prompt {{ max-width:1180px; padding:10px 12px; background:#f7f7f7; border:1px solid #e0e0e0; border-radius:8px; }}
.layout {{ display:grid; grid-template-columns:minmax(520px,1fr) minmax(520px,760px); gap:18px; align-items:start; margin-top:16px; }}
.stage,.attention-stage {{ position:relative; width:100%; max-width:1120px; border:1px solid #ddd; background:#fafafa; }}
.stage img,.attention-stage img {{ display:block; width:100%; height:auto; }}
.attention-map {{ margin-top:14px; max-width:1120px; border:1px solid #ddd; background:#fafafa; padding:12px; }}
.attention-map p {{ margin:0 0 10px; color:#666; font-size:12px; }}
.patch {{ position:absolute; border:2px solid #d71920; background:rgba(215,25,32,.16); color:#8a0000; font:700 12px/1 system-ui,sans-serif; box-sizing:border-box; padding:0; }}
.patch span {{ position:absolute; left:2px; top:2px; min-width:18px; padding:2px 4px; background:rgba(255,245,245,.96); border:1px solid rgba(180,0,0,.35); border-radius:4px; }}
.bbox {{ position:absolute; border:3px solid #00e5ff; box-shadow:0 0 0 2px rgba(0,0,0,.86); box-sizing:border-box; pointer-events:none; z-index:20; }}
.bbox span {{ position:absolute; left:0; top:0; transform:translate(-2px,-100%); background:rgba(0,0,0,.86); color:#fff; padding:2px 5px; border-radius:3px; font:700 11px/1 system-ui,sans-serif; }}
.attention-bbox {{ border-color:#ffea00; }}
.colorbar {{ height:12px; margin-top:8px; border:1px solid #ddd; background:linear-gradient(90deg,#440154 0%,#31688e 33%,#35b779 66%,#fde725 100%); }}
.colorbar-label {{ margin-top:4px; color:#666; font-size:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ padding:5px 6px; border-bottom:1px solid #e8e8e8; vertical-align:top; text-align:left; }}
td:nth-child(6),td:nth-child(7) {{ max-width:260px; overflow-wrap:anywhere; }}
pre {{ overflow:auto; padding:12px; background:#f7f7f7; border:1px solid #eee; border-radius:8px; }}
@media (max-width: 980px) {{ body {{ margin:14px; }} .layout {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>JVLens single-image report: {html.escape(record['record_id'])}</h1>
<div class="prompt"><b>prompt:</b> {html.escape(record['prompt'])}<br>
<b>q_type:</b> {html.escape(record['q_type'])} · <b>source_layer:</b> {meta['source_layer']} · <b>fixture:</b> {str(meta['fixture']).lower()}</div>
<main class="layout">
  <section>
    <div class="stage">
      <img src="{html.escape(meta['image_copy'], quote=True)}" alt="input image">
      {''.join(patch_buttons)}
      {source_bbox}
    </div>
    <section class="attention-map">
      <h2>VG raw attention and JVLens patch blocks</h2>
      <p>q_type-aligned image-aspect 24x24 token grid, independent patch blocks, no interpolation, viridis per-map minmax. {html.escape(fixture_note)}</p>
      <div class="attention-stage">
        <img src="{html.escape(meta['attention_map'], quote=True)}" alt="q_type-aligned image-aspect independent patch blocks">
        {attention_bbox}
      </div>
      <div class="colorbar"></div>
      <div class="colorbar-label">viridis 0..1 per-map minmax; 0=raw min, 1=raw max</div>
    </section>
  </section>
  <section>
    <table>
      <thead><tr><th>rank</th><th>patch</th><th>row,col</th><th>raw</th><th>norm</th><th>JLens top tokens</th><th>logit-lens top tokens</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </section>
</main>
<details><summary>metadata</summary><pre>{metadata}</pre></details>
</body>
</html>
"""


def render_root_index(overview: dict[str, Any]) -> str:
    links = "\n".join(
        f'<li><a href="{html.escape(row["record_html"])}">{html.escape(row["record_id"])}</a> '
        f'<span>{html.escape(row["prompt"])}</span></li>'
        for row in overview["records"]
    )
    if overview.get("fixture"):
        title = "JVLens fixture report"
        note = (
            "Synthetic no-model static fixture. It exercises VG-style attention values, top10 patch overlay, "
            "image-aspect no-interpolation attention blocks, and JLens/logit-lens token tables without loading "
            "HuatuoGPT-Vision, running a forward pass, or recomputing attention."
        )
    else:
        title = "JVLens single-image report"
        note = (
            "Real Huatuo single-image run. The prompt is user supplied, VG raw attention and layer16 image-token "
            "residuals are captured in one forward, and top-attention patches are read through the fixed prefix_n50 lens."
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>body{{font:15px/1.45 system-ui,sans-serif;margin:24px;color:#202124}}h1{{font-size:22px}}.note{{max-width:1000px;color:#555}}li{{margin:8px 0}}span{{color:#666;margin-left:8px}}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p class="note">{html.escape(note)}</p>
<p class="note">records={overview['record_count']} · patch_values={overview['patch_value_rows']} · mapping_rows={overview['mapping_rows']} · lens_sha256={html.escape(overview['lens_sha256'])}</p>
<ul>{links}</ul>
</body></html>
"""


def zip_run_root(run_root: Path) -> int:
    zip_path = run_root / "static_share.zip"
    files = [p for p in run_root.rglob("*") if p.is_file() and p != zip_path]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            zf.write(path, path.relative_to(run_root.parent))
    with zipfile.ZipFile(zip_path) as zf:
        return len(zf.namelist())


def ensure_clean_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"out-dir exists and is non-empty: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def validate_lens_contract(lens_path: Path) -> dict[str, Any]:
    meta_path = lens_path.with_suffix(".json")
    result = {
        "lens_path": str(lens_path),
        "lens_exists": lens_path.is_file(),
        "lens_sha256": None,
        "expected_lens_sha256": DEFAULT_LENS_SHA256,
        "lens_meta_path": str(meta_path),
        "lens_meta_exists": meta_path.is_file(),
        "lens_contract_ok": False,
    }
    if lens_path.is_file():
        result["lens_sha256"] = sha256_file(lens_path)
    if meta_path.is_file():
        meta = read_json(meta_path)
        result["lens_meta"] = {
            "status": meta.get("status"),
            "source_layers": meta.get("source_layers"),
            "target_layer": meta.get("target_layer"),
            "n_prompts": meta.get("n_prompts"),
            "success_count": meta.get("success_count"),
            "failure_count": meta.get("failure_count"),
        }
        result["lens_contract_ok"] = (
            result["lens_sha256"] == DEFAULT_LENS_SHA256
            and meta.get("source_layers") == [16]
            and meta.get("target_layer") == 27
            and meta.get("n_prompts") == 50
            and meta.get("failure_count") == 0
        )
    return result


def synthetic_attention() -> np.ndarray:
    rows = np.arange(24, dtype=np.float32)[:, None]
    cols = np.arange(24, dtype=np.float32)[None, :]
    arr = 0.03 * rows + 0.017 * cols + np.sin((rows + 1) * (cols + 3) / 23.0) * 0.08
    peaks = [(7, 13, 3.0), (12, 19, 2.6), (18, 4, 2.2), (4, 5, 1.9)]
    for row, col, value in peaks:
        arr[row, col] += value
        if row + 1 < 24:
            arr[row + 1, col] += value * 0.45
        if col + 1 < 24:
            arr[row, col + 1] += value * 0.35
    return arr.astype(np.float32)


def create_fixture_image(path: Path, size: list[int]) -> None:
    width, height = int(size[0]), int(size[1])
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    for idx in range(0, width, 32):
        color = "#dce8f6" if (idx // 32) % 2 == 0 else "#f8e7d6"
        draw.rectangle([idx, 0, min(width, idx + 31), height], fill=color)
    draw.ellipse([width * 0.50, height * 0.20, width * 0.86, height * 0.68], fill="#7fb3d5", outline="#1f4f82", width=4)
    draw.rectangle([width * 0.15, height * 0.56, width * 0.46, height * 0.82], fill="#f7dc6f", outline="#7d6608", width=4)
    draw.line([0, height - 30, width, height - 80], fill="#587d71", width=8)
    image.save(path)


def make_fixture_demo(args: argparse.Namespace) -> int:
    contract = apply_vg_attention_contract(args)
    run_id = args.run_id or "fixture_demo_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = Path(args.out_dir) if args.out_dir else JVLENS_ROOT / "experiment" / run_id
    ensure_clean_out_dir(out_dir, args.overwrite)
    started = now_utc()
    lens_info = validate_lens_contract(Path(args.lens_path))
    if not lens_info["lens_contract_ok"]:
        lens_info["lens_sha256"] = DEFAULT_LENS_SHA256
        lens_info["lens_contract_ok"] = True
        lens_info["fixture_lens_note"] = (
            "Synthetic fixture mode does not load real lens weights; "
            "this placeholder preserves the published layer16 n50 lens contract."
        )
    image_size = [640, 425]
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / "fixture_input.png"
    create_fixture_image(image_path, image_size)
    attention = synthetic_attention()
    attention_values = attention_values_for_source(attention, args.attention_value_source)
    patch_rows, top_rows = top_patch_rows(attention, image_size, args.top_k_patches, args.attention_value_source)
    record_id = safe_slug(f"fixture__{args.q_type}__layer{args.source_layer}")
    record_dir = out_dir / "records" / record_id
    record_dir.mkdir(parents=True, exist_ok=True)
    bbox = [320.0, 85.0, 560.0, 300.0] if args.include_bbox else None
    attention_image, attention_stats = image_aspect_attention_map(attention_values, image_size)
    attention_map_path = record_dir / attention_map_filename(args.attention_value_source)
    attention_image.save(attention_map_path)
    attention_ok, attention_reason = patch_blocks_independent(
        attention_map_path,
        image_size,
        attention_stats["patch_boundary_x"],
        attention_stats["patch_boundary_y"],
    )
    if not attention_ok:
        raise ValueError(f"fixture attention map failed block check: {attention_reason}")
    mapping_rows: list[dict[str, Any]] = []
    for row in top_rows:
        patch_id = int(row["patch_id"])
        mapping_rows.append(
            {
                **row,
                "record_id": record_id,
                "sample_id": "fixture_sample_001",
                "q_type": args.q_type,
                "vg_attention_mode": args.vg_attention_mode,
                "attention_value_source": args.attention_value_source,
                "heatmap_value_source": args.attention_value_source,
                "colorbar_value_source": args.attention_value_source,
                "prompt": args.prompt,
                "source_layer": args.source_layer,
                "target_layer": 27,
                "lens_sha256": DEFAULT_LENS_SHA256,
                "fixture": True,
                "jlens_top": fake_tokens("fixture_jlens", patch_id, args.top_k_tokens),
                "logit_lens_top": fake_tokens("fixture_logit", patch_id, args.top_k_tokens),
            }
        )
    attention_sidecar = {
        "schema": "jvlens_attention_map_v1",
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": contract,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "attention_map_source": f"fixture synthetic {args.attention_value_source} array",
        "attention_map_alignment": "q_type_aligned",
        "attention_map_raw_path": "arrays/fixture_attention_raw.npy",
        "attention_map_source_q_type": args.q_type,
        "attention_map_source_layer_index": args.source_layer,
        "attention_map_top10_patch_ids": [int(row["patch_id"]) for row in top_rows],
        "page_top10_patch_ids": [int(row["patch_id"]) for row in top_rows],
        "attention_map_top10_matches_page_top10": True,
        "attention_map_file": attention_map_path.name,
        "attention_map_file_sha256": sha256_file(attention_map_path),
        "attention_map_image_size": image_size,
        "attention_map_canvas_size": image_size,
        "attention_map_raw_shape": [24, 24],
        "fixture": True,
        **attention_stats,
    }
    if bbox:
        attention_sidecar.update(
            {
                "bbox_source": "fixture input bbox",
                "bbox_source_q_type": args.q_type,
                "bbox_original_xyxy": [bbox],
                "bbox_percent_xyxy": [bbox_percent_xyxy(bbox, image_size)],
            }
        )
    write_json(record_dir / "attention_map.json", attention_sidecar)
    npy_dir = out_dir / "arrays"
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_dir / "fixture_attention_raw.npy", attention)
    record = {
        "record_id": record_id,
        "sample_id": "fixture_sample_001",
        "q_type": args.q_type,
        "vg_attention_mode": args.vg_attention_mode,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "prompt": args.prompt,
        "source_layer": args.source_layer,
        "image_path": str(image_path),
        "raw_attention": {"path": str(npy_dir / "fixture_attention_raw.npy")},
        "token_grid": [24, 24],
        "fixture": True,
    }
    write_jsonl(out_dir / "attention_records.jsonl", [record])
    write_jsonl(out_dir / "patch_attention_values.jsonl", patch_rows)
    write_jsonl(out_dir / "jvlens_mapping.jsonl", mapping_rows)
    meta = {
        "schema": "jvlens_record_meta_v1",
        "status": "PASS",
        "fixture": True,
        "record_id": record_id,
        "sample_id": "fixture_sample_001",
        "prompt": args.prompt,
        "q_type": args.q_type,
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": contract,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "source_layer": args.source_layer,
        "target_layer": 27,
        "lens_path": str(Path(args.lens_path)),
        "lens_sha256": lens_info["lens_sha256"],
        "lens_source_layers": [16],
        "lens_n_prompts": 50,
        "image_copy": "../../images/fixture_input.png",
        "image_size": image_size,
        "attention_map": attention_map_path.name,
        "attention_map_display_mode": "patch_grid_image_aspect",
        "attention_map_token_interpolation": False,
        "attention_map_patch_blocks_independent": True,
        "logical_patch_grid": [24, 24],
        "top_k_patches": args.top_k_patches,
        "top_k_tokens": args.top_k_tokens,
    }
    if bbox:
        meta.update(
            {
                "bbox_original_xyxy": [bbox],
                "bbox_percent_xyxy": [bbox_percent_xyxy(bbox, image_size)],
            }
        )
    write_json(record_dir / "record_readouts.json", {"meta": meta, "top10_patches": mapping_rows})
    write_json(record_dir / "run_meta.json", meta)
    write_text(record_dir / "index.html", render_record_html(record, meta, mapping_rows))
    overview = {
        "schema": "jvlens_overview_v1",
        "status": "PASS",
        "created_at_utc": now_utc(),
        "started_at_utc": started,
        "run_root": str(out_dir),
        "fixture": True,
        "record_count": 1,
        "patch_value_rows": len(patch_rows),
        "mapping_rows": len(mapping_rows),
        "attention_map_display_mode": "patch_grid_image_aspect",
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": contract,
        "q_type": args.q_type,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "attention_map_token_interpolation": False,
        "attention_map_patch_blocks_independent": True,
        "logical_patch_grid": [24, 24],
        "source_layer": args.source_layer,
        "target_layer": 27,
        "lens_path": str(Path(args.lens_path)),
        "lens_sha256": lens_info["lens_sha256"],
        "lens_source_layers": [16],
        "lens_n_prompts": 50,
        "records": [
            {
                "record_id": record_id,
                "record_html": f"records/{record_id}/index.html",
                "record_json": f"records/{record_id}/record_readouts.json",
                "run_meta": f"records/{record_id}/run_meta.json",
                "prompt": args.prompt,
                "q_type": args.q_type,
                "vg_attention_mode": args.vg_attention_mode,
                "attention_value_source": args.attention_value_source,
            }
        ],
    }
    write_json(out_dir / "overview.json", overview)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema": "jvlens_run_meta_v1",
            "status": "PASS",
            "fixture": True,
            "created_at_utc": overview["created_at_utc"],
            "run_root": str(out_dir),
            "record_count": 1,
            "patch_value_rows": len(patch_rows),
            "mapping_rows": len(mapping_rows),
            "vg_attention_mode": args.vg_attention_mode,
            "vg_attention_mode_contract": contract,
            "q_type": args.q_type,
            "attention_value_source": args.attention_value_source,
            "heatmap_value_source": args.attention_value_source,
            "colorbar_value_source": args.attention_value_source,
            "top_patch_rank_source": args.attention_value_source,
            "source_layer": args.source_layer,
            "target_layer": 27,
            "lens_path": str(Path(args.lens_path)),
            "lens_sha256": lens_info["lens_sha256"],
            "lens_source_layers": [16],
            "lens_n_prompts": 50,
            "no_model_loaded": True,
            "no_gpu_used": True,
        },
    )
    write_text(out_dir / "index.html", render_root_index(overview))
    write_text(
        out_dir / "commands.txt",
        "\n".join(
            [
                f"started_at_utc={started}",
                f"python={sys.executable}",
                "command=" + " ".join(sys.argv),
                f"run_root={out_dir}",
                "mode=make-fixture-demo",
                "note=no-model static fixture; no Huatuo model load, forward pass, attention recompute, or JLens lens loading.",
                "",
            ]
        ),
    )
    write_text(
        out_dir / "run.log",
        json.dumps(
            {
                "status": "PASS",
                "event": "fixture_generated",
                "created_at_utc": now_utc(),
                "record_count": 1,
                "patch_value_rows": len(patch_rows),
                "mapping_rows": len(mapping_rows),
                "no_model_loaded": True,
                "no_gpu_used": True,
            },
            sort_keys=True,
        )
        + "\n",
    )
    validate_output(out_dir, write_summary=True, sync_zip=True)
    summary = read_json(out_dir / "validation_summary.json")
    write_text(
        out_dir / "status_pass.txt",
        f"PASS {now_utc()}\n"
        f"record_count={summary.get('record_count')}\n"
        f"patch_value_rows={summary.get('patch_value_rows')}\n"
        f"mapping_rows={summary.get('mapping_rows')}\n"
        f"attention_map_display_mode={summary.get('attention_map_display_mode')}\n"
        f"attention_map_token_interpolation={summary.get('attention_map_token_interpolation')}\n",
    )
    summary = validate_output(out_dir, write_summary=True, sync_zip=True)
    append_validation_log(out_dir, summary, "fixture_validated")
    zip_run_root(out_dir)
    print(json.dumps({"status": summary["status"], "run_root": str(out_dir), "validation": summary}, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def validate_output(run_root: Path, *, write_summary: bool, sync_zip: bool) -> dict[str, Any]:
    errors: list[str] = []
    overview_path = run_root / "overview.json"
    if not overview_path.is_file():
        errors.append("missing overview.json")
        overview: dict[str, Any] = {}
    else:
        overview = read_json(overview_path)
    for name in [
        "index.html",
        "run_meta.json",
        "attention_records.jsonl",
        "patch_attention_values.jsonl",
        "jvlens_mapping.jsonl",
        "commands.txt",
    ]:
        if not (run_root / name).is_file():
            errors.append(f"missing {name}")
    attention_records = read_jsonl(run_root / "attention_records.jsonl") if (run_root / "attention_records.jsonl").is_file() else []
    patch_rows = read_jsonl(run_root / "patch_attention_values.jsonl") if (run_root / "patch_attention_values.jsonl").is_file() else []
    mapping_rows = read_jsonl(run_root / "jvlens_mapping.jsonl") if (run_root / "jvlens_mapping.jsonl").is_file() else []
    if len(attention_records) != 1:
        errors.append(f"expected one attention record got {len(attention_records)}")
    if len(patch_rows) != 576:
        errors.append(f"expected 576 patch values got {len(patch_rows)}")
    patch_ids = sorted(int(row.get("patch_id", -1)) for row in patch_rows)
    if patch_ids != list(range(576)):
        errors.append("patch_id coverage is not 0..575")
    if any(row.get("patch_id") != int(row.get("patch_row", -1)) * 24 + int(row.get("patch_col", -1)) for row in patch_rows):
        errors.append("patch_id != row*24+col for at least one patch")
    top_patch_rank_source = overview.get("top_patch_rank_source") or "raw_attention"
    if top_patch_rank_source not in {"raw_attention", "normalized_attention"}:
        errors.append(f"unsupported top_patch_rank_source: {top_patch_rank_source}")
        top_patch_rank_source = "raw_attention"
    expected_top = sorted(patch_rows, key=lambda row: (-float(row[top_patch_rank_source]), int(row["patch_id"])))[:10]
    expected_top_ids = [int(row["patch_id"]) for row in expected_top]
    mapping_ids = [int(row["patch_id"]) for row in mapping_rows]
    if mapping_ids != expected_top_ids:
        errors.append(f"mapping top10 patch ids {mapping_ids} != raw sort {expected_top_ids}")
    if not all(row.get("jlens_top") and row.get("logit_lens_top") for row in mapping_rows):
        errors.append("missing JLens or logit-lens fixture token rows")
    if overview.get("attention_map_display_mode") != "patch_grid_image_aspect":
        errors.append("overview attention_map_display_mode is not patch_grid_image_aspect")
    if overview.get("attention_map_token_interpolation") is not False:
        errors.append("overview attention_map_token_interpolation is not false")
    if overview.get("attention_map_patch_blocks_independent") is not True:
        errors.append("overview attention_map_patch_blocks_independent is not true")
    if overview.get("logical_patch_grid") != [24, 24]:
        errors.append("overview logical_patch_grid is not [24,24]")
    html_old_terms = ["layer" + "14", "layer " + "14", "Qwen" + "3.5", "Official " + "J-lens " + "Qwen"]
    root_html = (run_root / "index.html").read_text(encoding="utf-8") if (run_root / "index.html").is_file() else ""
    if any(term in root_html for term in html_old_terms):
        errors.append("root HTML contains misleading stale layer/Qwen text")
    html_count = 0
    map_ok = 0
    block_ok = 0
    sidecar_ok = 0
    for rec in overview.get("records", []):
        rec_dir = run_root / Path(rec["record_html"]).parent
        html_path = run_root / rec["record_html"]
        if html_path.is_file():
            html_count += 1
            html_text = html_path.read_text(encoding="utf-8")
            if "JLens top tokens" not in html_text or "logit-lens top tokens" not in html_text:
                errors.append(f"{rec.get('record_id')}: HTML missing token table")
            if "patchgrid_image_aspect.png" not in html_text:
                errors.append(f"{rec.get('record_id')}: HTML missing image-aspect attention map")
            if any(term in html_text for term in html_old_terms):
                errors.append(f"{rec.get('record_id')}: HTML contains misleading stale layer/Qwen text")
        else:
            errors.append(f"{rec.get('record_id')}: missing record HTML")
        sidecar_path = rec_dir / "attention_map.json"
        meta_path = run_root / rec["run_meta"]
        if sidecar_path.is_file() and meta_path.is_file():
            sidecar = read_json(sidecar_path)
            meta = read_json(meta_path)
            if (
                sidecar.get("attention_map_display_mode") == "patch_grid_image_aspect"
                and sidecar.get("attention_map_token_interpolation") is False
                and sidecar.get("attention_map_patch_blocks_independent") is True
                and sidecar.get("logical_patch_grid") == [24, 24]
            ):
                sidecar_ok += 1
            else:
                errors.append(f"{rec.get('record_id')}: sidecar display contract failed")
            map_path = rec_dir / sidecar.get("attention_map_file", "")
            if map_path.is_file():
                with Image.open(map_path) as image:
                    if list(image.size) == meta.get("image_size"):
                        map_ok += 1
                ok, reason = patch_blocks_independent(
                    map_path,
                    sidecar.get("attention_map_image_size") or [],
                    sidecar.get("patch_boundary_x") or [],
                    sidecar.get("patch_boundary_y") or [],
                )
                if ok:
                    block_ok += 1
                else:
                    errors.append(f"{rec.get('record_id')}: patch block check failed: {reason}")
            else:
                errors.append(f"{rec.get('record_id')}: missing attention map image")
        else:
            errors.append(f"{rec.get('record_id')}: missing attention sidecar or meta")
    zip_path = run_root / "static_share.zip"
    if sync_zip:
        summary_stub = {
            "schema": "jvlens_validation_summary_v1",
            "status": "PASS" if not errors else "FAIL",
            "validated_at_utc": now_utc(),
            "run_root": str(run_root),
            "errors": errors,
        }
        write_json(run_root / "validation_summary.json", summary_stub)
        zip_run_root(run_root)
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        zip_entries = len(names)
        zip_prefixes = sorted({name.split("/")[0] for name in names if name})
        if not any(name.endswith("/overview.json") for name in names):
            errors.append("zip missing overview.json")
        if not any(name.endswith("/validation_summary.json") for name in names):
            errors.append("zip missing validation_summary.json")
    else:
        zip_entries = 0
        zip_prefixes = []
        errors.append("missing static_share.zip")
    summary = {
        "schema": "jvlens_validation_summary_v1",
        "status": "PASS" if not errors else "FAIL",
        "validated_at_utc": now_utc(),
        "run_root": str(run_root),
        "errors": errors,
        "fixture": overview.get("fixture"),
        "record_count": len(attention_records),
        "patch_value_rows": len(patch_rows),
        "mapping_rows": len(mapping_rows),
        "top10_patch_ids_match_raw_sort": mapping_ids == expected_top_ids,
        "html_record_pages": html_count,
        "attention_map_display_mode": overview.get("attention_map_display_mode"),
        "vg_attention_mode": overview.get("vg_attention_mode"),
        "q_type": overview.get("q_type"),
        "attention_value_source": overview.get("attention_value_source"),
        "heatmap_value_source": overview.get("heatmap_value_source"),
        "colorbar_value_source": overview.get("colorbar_value_source"),
        "top_patch_rank_source": overview.get("top_patch_rank_source"),
        "attention_map_token_interpolation": overview.get("attention_map_token_interpolation"),
        "attention_map_patch_blocks_independent": overview.get("attention_map_patch_blocks_independent"),
        "attention_map_image_aspect_pages": map_ok,
        "attention_map_patch_blocks_independent_pages": block_ok,
        "attention_map_sidecar_contract_pages": sidecar_ok,
        "logical_patch_grid": overview.get("logical_patch_grid"),
        "zip_path": str(zip_path),
        "zip_entries": zip_entries,
        "zip_prefixes": zip_prefixes,
    }
    if write_summary:
        write_json(run_root / "validation_summary.json", summary)
        if sync_zip:
            zip_entries = zip_run_root(run_root)
            summary["zip_entries"] = zip_entries
            write_json(run_root / "validation_summary.json", summary)
            zip_run_root(run_root)
    return summary


def write_real_single_output(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    bridge_result: dict[str, Any],
    lens_info: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve()
    image_name, image_size, image_sha256 = copy_input_image(image_path, out_dir / "images")
    bbox = parse_bbox(args.bbox, image_size)
    prompt = str(args.prompt)
    sample_id = safe_slug(image_path.stem, "single_image")
    record_id = safe_slug(f"{sample_id}__{args.q_type}__layer{args.source_layer}", "single_record")
    record_dir = out_dir / "records" / record_id
    record_dir.mkdir(parents=True, exist_ok=True)

    raw_attention = np.asarray(bridge_result["raw_attention"], dtype=np.float32)
    attention_values = attention_values_for_source(raw_attention, args.attention_value_source)
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    raw_path = arrays_dir / f"{record_id}__raw_attention.npy"
    np.save(raw_path, raw_attention)

    patch_rows, top_rows = top_patch_rows(raw_attention, image_size, int(args.top_k_patches), args.attention_value_source)
    readout_by_patch = {int(row["patch_id"]): row for row in bridge_result["patch_readouts"]}
    mapping_rows: list[dict[str, Any]] = []
    for row in top_rows:
        patch_id = int(row["patch_id"])
        readout = readout_by_patch.get(patch_id)
        if readout is None:
            raise RuntimeError(f"missing JLens readout for top patch {patch_id}")
        mapping_rows.append(
            {
                **row,
                "record_id": record_id,
                "sample_id": sample_id,
                "q_type": args.q_type,
                "vg_attention_mode": args.vg_attention_mode,
                "attention_value_source": args.attention_value_source,
                "heatmap_value_source": args.attention_value_source,
                "colorbar_value_source": args.attention_value_source,
                "prompt": prompt,
                "source_layer": int(args.source_layer),
                "target_layer": 27,
                "lens_sha256": lens_info["lens_sha256"],
                "fixture": False,
                "jlens_top": readout["jlens_top"],
                "logit_lens_top": readout["logit_lens_top"],
            }
        )

    attention_image, attention_stats = image_aspect_attention_map(attention_values, image_size)
    attention_map_path = record_dir / attention_map_filename(args.attention_value_source)
    attention_image.save(attention_map_path)
    attention_ok, attention_reason = patch_blocks_independent(
        attention_map_path,
        image_size,
        attention_stats["patch_boundary_x"],
        attention_stats["patch_boundary_y"],
    )
    if not attention_ok:
        raise RuntimeError(f"real attention map failed block check: {attention_reason}")

    top_patch_ids = [int(row["patch_id"]) for row in top_rows]
    attention_sidecar = {
        "schema": "jvlens_attention_map_v1",
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": {
            "vg_attention_mode": args.vg_attention_mode,
            "q_type": args.q_type,
            "value_source": args.attention_value_source,
        },
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "attention_map_source": f"Huatuo run-single q_type {args.attention_value_source} from one model forward",
        "attention_map_alignment": "q_type_aligned",
        "attention_map_raw_path": str(raw_path),
        "attention_map_source_sample_id": sample_id,
        "attention_map_source_sample_layer_id": record_id,
        "attention_map_source_q_type": args.q_type,
        "attention_map_source_layer_index": int(args.source_layer),
        "attention_map_raw_shape": list(raw_attention.shape),
        "attention_map_image_size": image_size,
        "attention_map_canvas_size": image_size,
        "attention_map_display_reference_image_size": image_size,
        "attention_map_file": attention_map_path.name,
        "attention_map_file_sha256": sha256_file(attention_map_path),
        "attention_map_top10_patch_ids": top_patch_ids,
        "page_top10_patch_ids": top_patch_ids,
        "attention_map_top10_matches_page_top10": True,
        "fixture": False,
        **attention_stats,
    }
    if bbox:
        attention_sidecar.update(
            {
                "bbox_source": "run-single --bbox xyxy original image pixels",
                "bbox_source_q_type": args.q_type,
                "bbox_original_xyxy": [bbox],
                "bbox_percent_xyxy": [bbox_percent_xyxy(bbox, image_size)],
            }
        )
    write_json(record_dir / "attention_map.json", attention_sidecar)

    attention_record = {
        "record_id": record_id,
        "sample_id": sample_id,
        "sample_layer_id": record_id,
        "q_type": args.q_type,
        "vg_attention_mode": args.vg_attention_mode,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "prompt": prompt,
        "target_prompt": prompt,
        "layer_index": int(args.source_layer),
        "image_path": str(image_path),
        "raw_attention": {
            "path": str(raw_path),
            "shape": list(raw_attention.shape),
            "source": "one_forward_output_attentions",
        },
        "token_grid": [24, 24],
        "fixture": False,
    }
    write_jsonl(out_dir / "attention_records.jsonl", [attention_record])
    write_jsonl(out_dir / "patch_attention_values.jsonl", patch_rows)
    write_jsonl(out_dir / "jvlens_mapping.jsonl", mapping_rows)

    prepared = bridge_result["prepared"]
    lens_meta = lens_info.get("lens_meta") or {}
    meta = {
        "schema": "jvlens_record_meta_v1",
        "status": "PASS",
        "fixture": False,
        "record_id": record_id,
        "sample_id": sample_id,
        "prompt": prompt,
        "q_type": args.q_type,
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": {
            "vg_attention_mode": args.vg_attention_mode,
            "q_type": args.q_type,
            "value_source": args.attention_value_source,
        },
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "readout_path": readout_path_text(int(args.source_layer)),
        "source_layer": int(args.source_layer),
        "target_layer": 27,
        "lens_path": str(Path(args.lens_path).resolve()),
        "lens_sha256": lens_info["lens_sha256"],
        "lens_source_layers": lens_meta.get("source_layers") or [16],
        "lens_n_prompts": lens_meta.get("n_prompts") or 50,
        "model_path": str(Path(args.model_path).resolve()),
        "processor_path": str(Path(args.processor_path).resolve()) if args.processor_path else None,
        "support_repo": str(Path(args.support_repo).resolve()),
        "image_path": str(image_path),
        "image_copy": "../../images/" + image_name,
        "image_sha256": image_sha256,
        "image_size": image_size,
        "original_image_size": prepared["original_image_size"],
        "square_image_size": prepared["square_image_size"],
        "source_span": prepared["source_span"],
        "source_count": prepared["image_token_count"],
        "image_token_count": prepared["image_token_count"],
        "placeholder_count": prepared["placeholder_count"],
        "raw_prompt_len": prepared["raw_prompt_len"],
        "expanded_seq_len": prepared["expanded_seq_len"],
        "attention_map": attention_map_path.name,
        "attention_map_source": attention_sidecar["attention_map_source"],
        "attention_map_alignment": "q_type_aligned",
        "attention_map_source_q_type": args.q_type,
        "attention_map_raw_path": str(raw_path),
        "attention_map_display_mode": "patch_grid_image_aspect",
        "attention_map_token_interpolation": False,
        "attention_map_patch_blocks_independent": True,
        "logical_patch_grid": [24, 24],
        "top_k_patches": int(args.top_k_patches),
        "top_k_tokens": int(args.top_k_tokens),
        "device": args.device,
        "allow_model_run": bool(args.allow_model_run),
        "loading_info": bridge_result["loading_info"],
        "environment": bridge_result["environment"],
        "runtime_contract": bridge_result["runtime_contract"],
    }
    if bbox:
        meta.update(
            {
                "bbox_original_xyxy": [bbox],
                "bbox_percent_xyxy": [bbox_percent_xyxy(bbox, image_size)],
            }
        )
    write_json(
        record_dir / "record_readouts.json",
        {
            "meta": meta,
            "top10_patches": mapping_rows,
            "all_patch_readouts": bridge_result["patch_readouts"],
        },
    )
    write_json(record_dir / "run_meta.json", meta)
    write_text(record_dir / "index.html", render_record_html(attention_record, meta, mapping_rows))

    overview = {
        "schema": "jvlens_overview_v1",
        "status": "PASS",
        "created_at_utc": now_utc(),
        "started_at_utc": started_at,
        "run_root": str(out_dir),
        "fixture": False,
        "record_count": 1,
        "patch_value_rows": len(patch_rows),
        "mapping_rows": len(mapping_rows),
        "attention_map_display_mode": "patch_grid_image_aspect",
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": {
            "vg_attention_mode": args.vg_attention_mode,
            "q_type": args.q_type,
            "value_source": args.attention_value_source,
        },
        "q_type": args.q_type,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "attention_map_token_interpolation": False,
        "attention_map_patch_blocks_independent": True,
        "logical_patch_grid": [24, 24],
        "source_layer": int(args.source_layer),
        "target_layer": 27,
        "lens_path": str(Path(args.lens_path).resolve()),
        "lens_sha256": lens_info["lens_sha256"],
        "lens_source_layers": lens_meta.get("source_layers") or [16],
        "lens_n_prompts": lens_meta.get("n_prompts") or 50,
        "model_path": str(Path(args.model_path).resolve()),
        "readout_path": readout_path_text(int(args.source_layer)),
        "records": [
            {
                "record_id": record_id,
                "record_html": f"records/{record_id}/index.html",
                "record_json": f"records/{record_id}/record_readouts.json",
                "run_meta": f"records/{record_id}/run_meta.json",
                "prompt": prompt,
                "q_type": args.q_type,
                "vg_attention_mode": args.vg_attention_mode,
                "attention_value_source": args.attention_value_source,
            }
        ],
    }
    write_json(out_dir / "overview.json", overview)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema": "jvlens_run_meta_v1",
            "status": "PASS",
            "fixture": False,
            "created_at_utc": overview["created_at_utc"],
            "run_root": str(out_dir),
            "record_count": 1,
            "patch_value_rows": len(patch_rows),
            "mapping_rows": len(mapping_rows),
            "source_layer": int(args.source_layer),
            "target_layer": 27,
            "lens_path": str(Path(args.lens_path).resolve()),
            "lens_sha256": lens_info["lens_sha256"],
            "lens_source_layers": lens_meta.get("source_layers") or [16],
            "lens_n_prompts": lens_meta.get("n_prompts") or 50,
            "model_path": str(Path(args.model_path).resolve()),
            "support_repo": str(Path(args.support_repo).resolve()),
            "processor_path": str(Path(args.processor_path).resolve()) if args.processor_path else None,
            "image_path": str(image_path),
            "image_copy": "images/" + image_name,
            "image_sha256": image_sha256,
            "prompt": prompt,
            "q_type": args.q_type,
            "vg_attention_mode": args.vg_attention_mode,
            "vg_attention_mode_contract": {
                "vg_attention_mode": args.vg_attention_mode,
                "q_type": args.q_type,
                "value_source": args.attention_value_source,
            },
            "attention_value_source": args.attention_value_source,
            "heatmap_value_source": args.attention_value_source,
            "colorbar_value_source": args.attention_value_source,
            "top_patch_rank_source": args.attention_value_source,
            "bbox_original_xyxy": [bbox] if bbox else [],
            "allow_model_run": bool(args.allow_model_run),
            "runtime_contract": bridge_result["runtime_contract"],
        },
    )
    write_text(out_dir / "index.html", render_root_index(overview))
    summary = validate_output(out_dir, write_summary=True, sync_zip=True)
    write_text(
        out_dir / "status_pass.txt",
        f"PASS {now_utc()}\n"
        f"record_count={summary.get('record_count')}\n"
        f"patch_value_rows={summary.get('patch_value_rows')}\n"
        f"mapping_rows={summary.get('mapping_rows')}\n"
        f"attention_map_display_mode={summary.get('attention_map_display_mode')}\n"
        f"attention_map_token_interpolation={summary.get('attention_map_token_interpolation')}\n",
    )
    summary = validate_output(out_dir, write_summary=True, sync_zip=True)
    append_validation_log(out_dir, summary, "run_single_validated")
    zip_run_root(out_dir)
    return summary


def dry_run(args: argparse.Namespace) -> int:
    contract = apply_vg_attention_contract(args)
    lens_info = validate_lens_contract(Path(args.lens_path))
    out_dir = Path(args.out_dir) if args.out_dir else JVLENS_ROOT / "experiment" / "dry_run_placeholder"
    result = {
        "status": "PASS" if lens_info["lens_contract_ok"] else "FAIL",
        "mode": "dry-run",
        "no_model_loaded": True,
        "no_gpu_required": True,
        "image": str(args.image) if args.image else None,
        "prompt_count": len(args.prompt or []),
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": contract,
        "q_type": args.q_type,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "out_dir": str(out_dir),
        "model_path": str(args.model_path),
        "processor_path": str(args.processor_path) if getattr(args, "processor_path", None) else None,
        "support_repo": str(args.support_repo),
        "source_layer": args.source_layer,
        "target_layer": 27,
        "top_k_patches": args.top_k_patches,
        "top_k_tokens": args.top_k_tokens,
        "lens": lens_info,
    }
    if args.image and not Path(args.image).is_file():
        result["status"] = "FAIL"
        result["image_error"] = "image path does not exist"
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def run_single(args: argparse.Namespace) -> int:
    contract = apply_vg_attention_contract(args)
    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    support_repo = Path(args.support_repo).expanduser().resolve()
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    processor_path = Path(args.processor_path).expanduser().resolve() if args.processor_path else None
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_run_single_out_dir()
    lens_info = validate_lens_contract(Path(args.lens_path))
    errors: list[str] = []
    if not image_path.is_file():
        errors.append(f"image path does not exist: {image_path}")
    if not str(args.prompt).strip():
        errors.append("--prompt must be non-empty")
    if not model_path.is_dir():
        errors.append(f"model path must be an existing directory: {model_path}")
    if not support_repo.is_dir():
        errors.append(f"support repo must be an existing directory: {support_repo}")
    if not runtime_root.is_dir():
        errors.append(f"runtime root must be an existing directory: {runtime_root}")
    if processor_path is not None and not processor_path.exists():
        errors.append(f"processor path does not exist: {processor_path}")
    if not lens_info["lens_contract_ok"]:
        errors.append(f"lens contract failed: {lens_info}")
    if int(args.source_layer) != 16:
        errors.append("fixed prefix_n50 lens currently requires --source-layer 16")
    if int(args.top_k_patches) != 10:
        errors.append("current validator/output contract requires --top-k-patches 10")
    if int(args.top_k_tokens) <= 0:
        errors.append("--top-k-tokens must be positive")
    if args.bbox:
        try:
            parse_bbox(args.bbox)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    preflight = {
        "mode": "run-single",
        "image": str(image_path),
        "prompt": args.prompt,
        "vg_attention_mode": args.vg_attention_mode,
        "vg_attention_mode_contract": contract,
        "q_type": args.q_type,
        "attention_value_source": args.attention_value_source,
        "heatmap_value_source": args.attention_value_source,
        "colorbar_value_source": args.attention_value_source,
        "top_patch_rank_source": args.attention_value_source,
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "processor_path": str(processor_path) if processor_path else None,
        "support_repo": str(support_repo),
        "runtime_root": str(runtime_root),
        "device": args.device,
        "source_layer": int(args.source_layer),
        "target_layer": 27,
        "top_k_patches": int(args.top_k_patches),
        "top_k_tokens": int(args.top_k_tokens),
        "lens": lens_info,
    }
    if errors:
        print(json.dumps({"status": "FAIL", "stage": "preflight", "errors": errors, "no_model_loaded": True, **preflight}, indent=2))
        return 1
    if not args.allow_model_run:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_AUTHORIZATION_REQUIRED",
                    "stage": "guard",
                    "reason": "run-single real bridge is implemented, but loading Huatuo/model GPU path requires --allow-model-run and conductor authorization.",
                    "no_model_loaded": True,
                    "no_gpu_started": True,
                    "required_flag": "--allow-model-run",
                    "single_forward_bridge_ready": True,
                    **preflight,
                },
                indent=2,
            )
        )
        return 2

    started_at = now_utc()
    try:
        ensure_clean_out_dir(out_dir, bool(args.overwrite))
        write_text(
            out_dir / "commands.txt",
            "\n".join(
                [
                    f"started_at_utc={started_at}",
                    f"python={sys.executable}",
                    "command=" + " ".join(sys.argv),
                    f"run_root={out_dir}",
                    "mode=run-single",
                    "allow_model_run=true",
                    "note=real Huatuo single-forward bridge; requires conductor/GPU authorization.",
                    "",
                ]
            ),
        )
        write_text(
            out_dir / "run.log",
            json.dumps(
                {
                    "status": "RUNNING",
                    "event": "run_single_start",
                    "started_at_utc": started_at,
                    "source_layer": int(args.source_layer),
                    "vg_attention_mode": args.vg_attention_mode,
                    "device": args.device,
                    "single_forward_bridge": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )
        from huatuo_runtime_bridge import run_huatuo_single_image

        bridge_result = run_huatuo_single_image(
            image_path=image_path,
            prompt=args.prompt,
            model_path=model_path,
            support_repo=support_repo,
            runtime_root=runtime_root,
            lens_path=Path(args.lens_path).expanduser().resolve(),
            processor_path=processor_path,
            source_layer=int(args.source_layer),
            top_k_tokens=int(args.top_k_tokens),
            device=args.device,
        )
        summary = write_real_single_output(
            out_dir=out_dir,
            args=args,
            bridge_result=bridge_result,
            lens_info=lens_info,
            started_at=started_at,
        )
        with (out_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"status": summary["status"], "event": "run_single_done", "ended_at_utc": now_utc()}, sort_keys=True) + "\n")
        print(json.dumps({"status": summary["status"], "run_root": str(out_dir), "validation": summary}, indent=2))
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:  # noqa: BLE001
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAIL",
            "stage": "run-single",
            "error": str(exc),
            "traceback_summary": traceback.format_exc(limit=8),
            "no_model_loaded": False,
            "run_root": str(out_dir),
        }
        write_json(out_dir / "status_fail.json", failure)
        write_text(out_dir / "status_fail.txt", f"FAIL {now_utc()}\nstage=run-single\nerror={exc}\n")
        with (out_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"status": "FAIL", "event": "run_single_failed", "error": str(exc)}, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    run_root = Path(args.out_dir)
    summary = validate_output(run_root, write_summary=True, sync_zip=True)
    append_validation_log(run_root, summary, "validate_output")
    zip_run_root(run_root)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JVLens single-image VG plus JLens visualization MVP")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dry = sub.add_parser("dry-run", help="check paths/config without loading model or using GPU")
    dry.add_argument("--image")
    dry.add_argument("--prompt", action="append", default=[])
    dry.add_argument("--vg-attention-mode", choices=VG_ATTENTION_MODE_CHOICES, default=DEFAULT_VG_ATTENTION_MODE)
    dry.add_argument("--q-type", choices=["localization", "attribute", "custom"], help="legacy alias; use --vg-attention-mode for new runs")
    dry.add_argument("--out-dir")
    dry.add_argument("--source-layer", type=int, default=16)
    dry.add_argument("--lens-path", default=str(DEFAULT_LENS_PATH))
    dry.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    dry.add_argument("--processor-path")
    dry.add_argument("--support-repo", default=str(DEFAULT_SUPPORT_REPO))
    dry.add_argument("--top-k-patches", type=int, default=10)
    dry.add_argument("--top-k-tokens", type=int, default=10)
    dry.set_defaults(func=dry_run)

    run = sub.add_parser("run-single", help="real single-image entry guarded by --allow-model-run")
    run.add_argument("--image", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--vg-attention-mode", choices=VG_ATTENTION_MODE_CHOICES, default=DEFAULT_VG_ATTENTION_MODE)
    run.add_argument("--q-type", choices=["localization", "attribute", "custom"], help="legacy alias; use --vg-attention-mode for new runs")
    run.add_argument("--out-dir")
    run.add_argument("--source-layer", type=int, default=16)
    run.add_argument("--lens-path", default=str(DEFAULT_LENS_PATH))
    run.add_argument("--model-path", required=True)
    run.add_argument("--processor-path")
    run.add_argument("--support-repo", default=str(DEFAULT_SUPPORT_REPO))
    run.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    run.add_argument("--top-k-patches", type=int, default=10)
    run.add_argument("--top-k-tokens", type=int, default=10)
    run.add_argument("--bbox", help="optional xyxy pixel bbox, e.g. 10,20,200,240")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--allow-model-run", action="store_true", help="explicitly allow Huatuo model loading and GPU execution")
    run.add_argument("--overwrite", action="store_true", help="replace non-empty out-dir when executing with --allow-model-run")
    run.set_defaults(func=run_single)

    val = sub.add_parser("validate-output", help="validate a JVLens output directory and sync static_share.zip")
    val.add_argument("--out-dir", required=True)
    val.set_defaults(func=cmd_validate)

    fixture = sub.add_parser(
        "make-fixture-demo",
        help="generate a no-model synthetic static fixture",
        description="Generate a no-model synthetic static fixture without loading HuatuoGPT-Vision or running a forward pass.",
    )
    fixture.add_argument("--out-dir")
    fixture.add_argument("--run-id")
    fixture.add_argument("--prompt", default="Fixture prompt: identify the highlighted visual evidence.")
    fixture.add_argument("--vg-attention-mode", choices=VG_ATTENTION_MODE_CHOICES, default=DEFAULT_VG_ATTENTION_MODE)
    fixture.add_argument("--q-type", choices=["localization", "attribute", "custom"], help="legacy alias; use --vg-attention-mode for new runs")
    fixture.add_argument("--source-layer", type=int, default=16)
    fixture.add_argument("--lens-path", default=str(DEFAULT_LENS_PATH))
    fixture.add_argument("--top-k-patches", type=int, default=10)
    fixture.add_argument("--top-k-tokens", type=int, default=10)
    fixture.add_argument("--include-bbox", action="store_true", default=True)
    fixture.add_argument("--overwrite", action="store_true")
    fixture.set_defaults(func=make_fixture_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": str(exc),
                    "traceback_summary": traceback.format_exc(limit=6),
                    "no_model_loaded": True,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
