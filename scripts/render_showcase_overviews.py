#!/usr/bin/env python3
"""Render sanitized showcase overview PNGs from checked-in metadata.

The script is intentionally CPU-only and does not read model outputs beyond the
static showcase metadata. Set JVLENS_CJK_FONT_PATH to an external CJK-capable
font when the host does not provide one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_DIR = ROOT / "examples" / "showcase"
CANVAS_SIZE = (1500, 1120)
FONT_ENV = "JVLENS_CJK_FONT_PATH"
CJK_PROBE = "中文颜色粉色痣疣疙近距离外观轮廓"
CJK_PROBE_TEXT = "中文测试 皮肤镜 结直肠"
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def font_family(path: Path) -> str:
    stem = path.stem
    if stem.lower() == "simhei":
        return "SimHei"
    return stem


def find_font(explicit_font_path: str | None = None) -> Path:
    candidates = []
    if explicit_font_path:
        candidates.append(Path(explicit_font_path).expanduser())
    env_path = os.environ.get(FONT_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        Path(path)
        for path in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        )
    )
    for candidate in candidates:
        if candidate.is_file() and cjk_render_check(candidate):
            return candidate
    raise SystemExit(
        f"No CJK-capable font found. Set {FONT_ENV} to an external .ttf/.ttc/.otf font path."
    )


def cjk_render_check(font_path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(font_path), 28)
    except OSError:
        return False
    masks = []
    for char in CJK_PROBE[:8]:
        mask = font.getmask(char)
        bbox = mask.getbbox()
        if bbox is None:
            return False
        masks.append(hashlib.sha256(bytes(mask)).hexdigest())
    return len(set(masks)) >= 4


def cjk_probe_evidence(font_path: Path) -> dict[str, Any]:
    font = ImageFont.truetype(str(font_path), 28)
    mask = font.getmask(CJK_PROBE_TEXT)
    bbox = mask.getbbox()
    return {
        "overview_cjk_probe_text": CJK_PROBE_TEXT,
        "overview_cjk_probe_bbox": list(bbox or (0, 0, 0, 0)),
        "overview_cjk_probe_sha256": hashlib.sha256(bytes(mask)).hexdigest(),
    }


def cjk_source_string_issues(payload: dict[str, Any]) -> dict[str, Any]:
    fields: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, str):
            if "\ufffd" in value:
                fields.append(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")
    return {
        "replacement_char_count": len(fields),
        "replacement_char_fields": fields,
    }


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def latin_font_path(cjk_font_path: Path) -> Path:
    return LATIN_FONT_PATH if LATIN_FONT_PATH.is_file() else cjk_font_path


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> str:
    if text_size(draw, text, font)[0] <= width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_size(draw, text[:mid] + suffix, font)[0] <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def draw_card(draw: ImageDraw.ImageDraw, xyxy: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(xyxy, radius=6, fill=(255, 255, 255), outline=(190, 202, 218), width=1)


def draw_rank_badge(draw: ImageDraw.ImageDraw, center: tuple[int, int], label: str, font: ImageFont.ImageFont) -> None:
    x, y = center
    w, h = text_size(draw, label, font)
    pad_x, pad_y = 5, 3
    box = (x - w // 2 - pad_x, y - h // 2 - pad_y, x + w // 2 + pad_x, y + h // 2 + pad_y)
    draw.rectangle(box, fill=(30, 101, 224), outline=(34, 211, 238), width=2)
    draw.text((box[0] + pad_x, box[1] + pad_y - 1), label, fill=(255, 255, 255), font=font)


def draw_source_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    old_overview: Image.Image,
    metadata: dict[str, Any],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    draw_card(draw, (23, 194, 625, 667))
    draw.text((34, 211), "Source image with raw_attention-ranked patches", fill=(10, 18, 35), font=font)
    src = old_overview.crop((35, 235, 612, 629))
    canvas.paste(src, (35, 235))
    draw.text((34, 642), "Orange=bbox, red=top patches, blue=rank", fill=(63, 78, 104), font=small)


def draw_heatmap_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    metadata: dict[str, Any],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    draw_card(draw, (638, 194, 1240, 667))
    draw.text((650, 211), "Attribute raw_attention patch map", fill=(10, 18, 35), font=font)
    arr = np.asarray(metadata["heatmap_matrix_24x24"], dtype=np.float32)
    lo = float(metadata["heatmap_display_min"])
    hi = float(metadata["heatmap_display_max"])
    norm = np.clip((arr - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    rgba = colormaps["viridis"](norm)
    heat = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8), mode="RGB").resize((576, 394), Image.Resampling.NEAREST)
    canvas.paste(heat, (650, 235))
    for bbox in metadata.get("bbox_percent_xyxy", []):
        x0 = 650 + round(bbox[0] / 100 * 576)
        y0 = 235 + round(bbox[1] / 100 * 394)
        x1 = 650 + round(bbox[2] / 100 * 576)
        y1 = 235 + round(bbox[3] / 100 * 394)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 149, 24), width=4)
        label_w, label_h = text_size(draw, "bbox", small)
        draw.rectangle((x0, y0 - label_h - 7, x0 + label_w + 8, y0), fill=(8, 10, 15), outline=(255, 149, 24), width=2)
        draw.text((x0 + 4, y0 - label_h - 5), "bbox", fill=(255, 255, 255), font=small)
    draw.text((650, 642), "raw_attention colorbar; bbox only; no top patch annotations", fill=(63, 78, 104), font=small)


def draw_table(
    draw: ImageDraw.ImageDraw,
    metadata: dict[str, Any],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
    tiny: ImageFont.ImageFont,
) -> None:
    draw_card(draw, (23, 666, 1447, 1098))
    draw.text((34, 682), "All 10 raw_attention-ranked patches with raw_attention heatmap values", fill=(10, 18, 35), font=font)
    headers = [
        ("rank", 34, 55),
        ("patch", 88, 75),
        ("row,col", 164, 88),
        ("raw attn", 260, 95),
        ("norm attn", 364, 95),
        ("JLens top tokens", 488, 510),
        ("logit-lens top tokens", 1012, 410),
    ]
    header_y = 720
    for label, x, _ in headers:
        draw.text((x, header_y), label, fill=(63, 78, 104), font=tiny)
    draw.line((34, 739, 1435, 739), fill=(205, 215, 228), width=1)
    row_h = 34
    for i, patch in enumerate(metadata["top_patches"][:10]):
        y0 = 743 + i * row_h
        if i % 2:
            draw.rectangle((34, y0, 1435, y0 + row_h), fill=(246, 248, 251))
        values = [
            str(patch["rank"]),
            str(patch["patch_id"]),
            f"{patch['patch_row']},{patch['patch_col']}",
            f"{patch['raw_attention']:.8g}",
            f"{patch['normalized_attention']:.4g}",
            ", ".join(patch.get("jlens_top_texts", [])),
            ", ".join(patch.get("logit_lens_top_texts", [])),
        ]
        for value, (_, x, width) in zip(values, headers):
            draw.text((x, y0 + 9), fit_text(draw, value, small, width), fill=(10, 18, 35), font=small)


def render_one(metadata_path: Path, font_path: Path, family: str) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    overview_path = ROOT / metadata["files"]["overview_png"]
    old_overview = Image.open(overview_path).convert("RGB")
    canvas = Image.new("RGB", CANVAS_SIZE, (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    latin_path = latin_font_path(font_path)
    title_font = load_font(latin_path, 29)
    font = load_font(latin_path, 20)
    small = load_font(latin_path, 14)
    tiny = load_font(latin_path, 12)
    table_font = load_font(font_path, 14)

    draw.rectangle((0, 0, CANVAS_SIZE[0], 93), fill=(15, 23, 42))
    title = f"JVLens Attribute Showcase - {metadata['sample_type']}"
    draw.text((34, 29), title, fill=(255, 255, 255), font=title_font)
    subtitle = "Layer16 attribute attention: raw_attention-ranked patches + raw_attention heatmap + prefix_n50 JLens readout"
    draw.text((35, 68), subtitle, fill=(229, 236, 247), font=small)
    draw.text((34, 122), "Attribute prompt", fill=(10, 18, 35), font=font)
    draw.text((34, 156), fit_text(draw, metadata["prompt"], small, 1080), fill=(63, 78, 104), font=small)
    contract = "q_type=attribute   heatmap_value_source=raw_attention   top_patch_rank_source=raw_attention   coordinate_system=patch_id=row*24+col"
    draw.text((34, 184), contract, fill=(63, 78, 104), font=small)

    draw_source_panel(canvas, draw, old_overview, metadata, font, small)
    draw_heatmap_panel(canvas, draw, metadata, font, small)
    draw_table(draw, metadata, font, table_font, tiny)

    canvas.save(overview_path)
    sha = sha256_file(overview_path)
    byte_count = overview_path.stat().st_size
    font_fields = {
        "overview_font_family": family,
        "overview_font_source": "external_cjk_font",
        "overview_font_path": "<external>",
        "overview_font_basename": font_path.name,
        "overview_cjk_font_fallback": True,
        "overview_cjk_render_check": cjk_render_check(font_path),
        **cjk_probe_evidence(font_path),
    }
    metadata.update(
        {
            "overview_png_sha256": sha,
            "overview_bytes": byte_count,
            **font_fields,
            "overview_cjk_source_string_issues": cjk_source_string_issues(metadata),
        }
    )
    write_json(metadata_path, metadata)
    return {
        "slug": metadata["slug"],
        "overview_png": metadata["files"]["overview_png"],
        "overview_bytes": byte_count,
        "overview_png_sha256": sha,
        "overview_font_family": family,
        "overview_font_source": "external_cjk_font",
        "overview_font_path": "<external>",
        "overview_font_basename": font_path.name,
        "overview_cjk_font_fallback": True,
        "overview_cjk_render_check": font_fields["overview_cjk_render_check"],
        "overview_cjk_probe_text": font_fields["overview_cjk_probe_text"],
        "overview_cjk_probe_bbox": font_fields["overview_cjk_probe_bbox"],
        "overview_cjk_probe_sha256": font_fields["overview_cjk_probe_sha256"],
    }


def sync_index(showcase_dir: Path, font_fields: dict[str, Any], rendered: dict[str, dict[str, Any]]) -> None:
    index_path = showcase_dir / "showcase_index.json"
    index = read_json(index_path)
    index.update(font_fields)
    for sample in index.get("samples", []):
        sample.update(rendered[sample["slug"]])
    write_json(index_path, index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showcase-dir", type=Path, default=SHOWCASE_DIR)
    parser.add_argument("--font-path", help=f"External CJK font path. Defaults to {FONT_ENV}.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    font_path = find_font(args.font_path)
    family = font_family(font_path)
    rendered: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(args.showcase_dir.glob("*/metadata.json")):
        result = render_one(metadata_path, font_path, family)
        rendered[result["slug"]] = result
    font_fields = {
        "overview_font_family": family,
        "overview_font_source": "external_cjk_font",
        "overview_font_path": "<external>",
        "overview_font_basename": font_path.name,
        "overview_cjk_font_fallback": True,
        "overview_cjk_render_check": cjk_render_check(font_path),
        **cjk_probe_evidence(font_path),
    }
    sync_index(args.showcase_dir, font_fields, rendered)
    print(json.dumps({"font_family": family, "rendered": rendered}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
