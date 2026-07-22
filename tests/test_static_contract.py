#!/usr/bin/env python3
"""CPU-only static checks for the JVLens staging package."""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_cjk_font_evidence(payload: dict, label: str) -> None:
    assert_ok(payload.get("overview_font_family"), f"missing CJK overview font family: {label}")
    assert_ok(payload.get("overview_font_source") == "external_cjk_font", f"unexpected CJK font source: {label}")
    assert_ok(payload.get("overview_font_path") == "<external>", f"CJK font path must be sanitized: {label}")
    assert_ok(payload.get("overview_font_basename"), f"missing CJK font basename: {label}")
    assert_ok("/" not in str(payload.get("overview_font_basename")), f"CJK font basename must not contain a path: {label}")
    assert_ok(payload.get("overview_cjk_font_fallback") is True, f"CJK fallback must be enabled: {label}")
    assert_ok(payload.get("overview_cjk_render_check") is True, f"CJK render check must pass: {label}")
    assert_ok(payload.get("overview_cjk_probe_text") == "中文测试 皮肤镜 结直肠", f"CJK probe text mismatch: {label}")
    probe_bbox = payload.get("overview_cjk_probe_bbox")
    assert_ok(isinstance(probe_bbox, list) and len(probe_bbox) == 4, f"CJK probe bbox missing: {label}")
    assert_ok(probe_bbox[2] > probe_bbox[0] and probe_bbox[3] > probe_bbox[1], f"CJK probe bbox is empty: {label}")
    probe_sha = payload.get("overview_cjk_probe_sha256", "")
    assert_ok(isinstance(probe_sha, str) and len(probe_sha) == 64, f"CJK probe hash missing: {label}")


def main() -> int:
    help_result = run([sys.executable, "run_jvlens.py", "--help"])
    assert_ok(help_result.returncode == 0, help_result.stderr)
    dry_help = run([sys.executable, "run_jvlens.py", "dry-run", "--help"])
    assert_ok(dry_help.returncode == 0, dry_help.stderr)

    fit_help = run([sys.executable, "scripts/fit/huatuo_fit_jlens.py", "--help"])
    assert_ok(fit_help.returncode == 0, fit_help.stderr)
    eval_help = run([sys.executable, "scripts/eval/huatuo_eval_fitted_lens.py", "--help"])
    assert_ok(eval_help.returncode == 0, eval_help.stderr)
    f1_help = run([sys.executable, "scripts/eval/huatuo_single_sample_f1.py", "--help"])
    assert_ok(f1_help.returncode == 0, f1_help.stderr)
    assert_ok("--vg-attention-mode" in dry_help.stdout, "CLI help must expose --vg-attention-mode")

    expected_modes = {
        "attribute_raw": ("attribute", "raw_attention"),
        "attribute_normalized": ("attribute", "normalized_attention"),
        "localization_raw": ("localization", "raw_attention"),
        "localization_normalized": ("localization", "normalized_attention"),
    }
    default_dry_run = run([sys.executable, "run_jvlens.py", "dry-run"])
    assert_ok(default_dry_run.returncode == 0, default_dry_run.stderr)
    default_payload = json.loads(default_dry_run.stdout)
    assert_ok(default_payload.get("vg_attention_mode") == "attribute_raw", "default VG attention mode must be attribute_raw")
    assert_ok(default_payload.get("q_type") == "attribute", "default VG mode q_type must be attribute")
    assert_ok(default_payload.get("attention_value_source") == "raw_attention", "default VG mode value source must be raw_attention")
    for mode, (q_type, value_source) in expected_modes.items():
        mode_result = run([sys.executable, "run_jvlens.py", "dry-run", "--vg-attention-mode", mode])
        assert_ok(mode_result.returncode == 0, mode_result.stderr)
        payload = json.loads(mode_result.stdout)
        assert_ok(payload.get("vg_attention_mode") == mode, f"dry-run mode mismatch for {mode}")
        assert_ok(payload.get("q_type") == q_type, f"dry-run q_type mismatch for {mode}")
        assert_ok(payload.get("attention_value_source") == value_source, f"dry-run value source mismatch for {mode}")
        assert_ok(payload.get("heatmap_value_source") == value_source, f"dry-run heatmap source mismatch for {mode}")
        assert_ok(payload.get("colorbar_value_source") == value_source, f"dry-run colorbar source mismatch for {mode}")
    invalid_mode = run([sys.executable, "run_jvlens.py", "dry-run", "--vg-attention-mode", "bad_mode"])
    assert_ok(invalid_mode.returncode != 0, "invalid VG attention mode must fail")
    assert_ok("invalid choice" in invalid_mode.stderr, "invalid VG attention mode must emit argparse choice error")

    fixture_outputs = [
        ROOT / "examples/fixture_demo/validation_summary.json",
        ROOT / "examples/fixture_demo/static_share.zip",
    ]
    preserved_fixture_outputs = {
        path: path.read_bytes() if path.exists() else None
        for path in fixture_outputs
    }
    try:
        validation = run([sys.executable, "run_jvlens.py", "validate-output", "--out-dir", "examples/fixture_demo"])
        assert_ok(validation.returncode == 0, validation.stdout + validation.stderr)
        assert_ok('"status": "PASS"' in validation.stdout, validation.stdout)
    finally:
        for path, data in preserved_fixture_outputs.items():
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(data)

    cli_source = (ROOT / "src/jvlens_cli.py").read_text(encoding="utf-8")
    guard_pos = cli_source.index("if not args.allow_model_run:")
    import_pos = cli_source.index("from huatuo_runtime_bridge import run_huatuo_single_image")
    assert_ok(guard_pos < import_pos, "run-single must guard before importing the real runtime bridge")

    fit_script = ROOT / "scripts/fit/huatuo_fit_jlens.py"
    f1_script = ROOT / "scripts/eval/huatuo_single_sample_f1.py"
    eval_script = ROOT / "scripts/eval/huatuo_eval_fitted_lens.py"
    assert_ok(fit_script.is_file(), "missing open-source Huatuo fitting script")
    assert_ok(f1_script.is_file(), "missing open-source token-level F1 script")
    assert_ok(eval_script.is_file(), "missing open-source fitted-lens eval script")
    f1_source = f1_script.read_text(encoding="utf-8")
    for needle in ["precision", "recall", "f1", "sec9_raw", "word_bag", "prf"]:
        assert_ok(needle in f1_source, f"F1 source missing {needle!r}")
    fit_source = fit_script.read_text(encoding="utf-8")
    assert_ok("--out-dir" in fit_source and "required=True" in fit_source, "fit script must require explicit --out-dir")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert_ok("raw_attention attribute maps" in readme, "README must describe raw_attention showcase maps")
    assert_ok("blue badges" in readme, "README must describe non-yellow rank badges")
    assert_ok("right heatmap shows raw_attention + bbox only" in readme, "README must state right heatmap annotation contract")
    assert_ok("CJK-capable fallback font" in readme, "README must state CJK-capable preview font fallback")
    readme_zh = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert_ok("支持 CJK 的 fallback 字体" in readme_zh, "Chinese README must state CJK preview font fallback")
    renderer_source = (ROOT / "scripts/render_showcase_overviews.py").read_text(encoding="utf-8")
    assert_ok("JVLENS_CJK_FONT_PATH" in renderer_source, "showcase renderer must support JVLENS_CJK_FONT_PATH")
    assert_ok("/" + "cpfs01" not in renderer_source, "showcase renderer must not embed internal font paths")
    showcase_index_path = ROOT / "examples/showcase/showcase_index.json"
    assert_ok(showcase_index_path.is_file(), "missing showcase index")
    showcase_index = json.loads(showcase_index_path.read_text(encoding="utf-8"))
    showcase_samples = showcase_index.get("samples", [])
    assert_ok(len(showcase_samples) == 3, "expected exactly three showcase samples")
    assert_ok(showcase_index.get("overview_table_visible_rows") == 10, "showcase index must expose all 10 table rows")
    assert_ok(
        showcase_index.get("overview_rank_label_style") == "blue_badge_white_text_cyan_outline",
        "showcase index rank labels must avoid yellow/viridis-high styling",
    )
    assert_ok(showcase_index.get("attention_prompt_type") == "attribute", "showcase index must use attribute prompts")
    assert_ok(showcase_index.get("vg_attention_mode") == "attribute_raw", "showcase index must use attribute_raw")
    assert_ok(showcase_index.get("attention_value_source") == "raw_attention", "showcase index must use raw_attention")
    assert_ok(showcase_index.get("heatmap_value_source") == "raw_attention", "showcase heatmap must use raw_attention")
    assert_ok(showcase_index.get("colorbar_value_source") == "raw_attention", "showcase colorbar must use raw_attention")
    assert_ok(showcase_index.get("top_patch_rank_source") == "raw_attention", "showcase top patches must preserve raw_attention-ranked image patches")
    assert_ok(showcase_index.get("overview_canvas_size") == [1500, 1120], "showcase overview canvas must be tall enough for all table rows")
    assert_ok(showcase_index.get("overview_table_visible_rows") == 10, "showcase overview must expose all 10 table rows")
    assert_ok(showcase_index.get("overview_rank_label_style") == "blue_badge_white_text_cyan_outline", "showcase rank labels must not use yellow badges")
    assert_ok(showcase_index.get("right_heatmap_patch_annotations") is False, "right heatmap must not draw patch annotations")
    assert_ok(showcase_index.get("right_heatmap_rank_labels") is False, "right heatmap must not draw rank labels")
    assert_ok(showcase_index.get("right_heatmap_bbox_overlay") is True, "right heatmap must keep bbox overlay")
    assert_cjk_font_evidence(showcase_index, "showcase index")
    index_transform = showcase_index.get("attention_map_transform", {})
    assert_ok(index_transform.get("patch_id_formula") == "patch_id = patch_row * 24 + patch_col", "showcase index patch id formula mismatch")
    assert_ok(index_transform.get("origin") == "top_left", "showcase index origin mismatch")
    assert_ok(index_transform.get("row_axis") == "y", "showcase index row axis mismatch")
    assert_ok(index_transform.get("col_axis") == "x", "showcase index col axis mismatch")
    assert_ok(index_transform.get("transpose") is False, "showcase index must not transpose heatmap")
    assert_ok(index_transform.get("flip_x") is False and index_transform.get("flip_y") is False, "showcase index must not flip heatmap")
    assert_ok("top raw-ranked attribute patches" not in readme, "README must not mention raw-ranked showcase patches")
    expected_showcase_types = {"COCO natural image", "Dermoscopy / skin lesion", "Colorectal-Endoscopy / polyp"}
    assert_ok({sample.get("sample_type") for sample in showcase_samples} == expected_showcase_types, "unexpected showcase sample types")
    for sample in showcase_samples:
        assert_ok(sample.get("q_type") == "attribute", f"showcase index sample must be attribute: {sample.get('slug')}")
        assert_ok(sample.get("vg_attention_mode") == "attribute_raw", f"showcase sample must use attribute_raw: {sample.get('slug')}")
        assert_ok(sample.get("attention_value_source") == "raw_attention", f"showcase sample must use raw attention: {sample.get('slug')}")
        assert_ok(sample.get("heatmap_value_source") == "raw_attention", f"showcase heatmap must use raw attention: {sample.get('slug')}")
        assert_ok(sample.get("colorbar_value_source") == "raw_attention", f"showcase colorbar must use raw attention: {sample.get('slug')}")
        assert_ok(sample.get("overview_canvas_size") == [1500, 1120], f"showcase sample canvas mismatch: {sample.get('slug')}")
        assert_ok(sample.get("overview_table_visible_rows") == 10, f"showcase sample must expose all 10 table rows: {sample.get('slug')}")
        assert_ok(sample.get("overview_rank_label_style") == "blue_badge_white_text_cyan_outline", f"showcase sample rank label style mismatch: {sample.get('slug')}")
        assert_ok(sample.get("right_heatmap_patch_annotations") is False, f"right heatmap must not draw patch annotations: {sample.get('slug')}")
        assert_ok(sample.get("right_heatmap_rank_labels") is False, f"right heatmap must not draw rank labels: {sample.get('slug')}")
        assert_ok(sample.get("right_heatmap_bbox_overlay") is True, f"right heatmap must keep bbox overlay: {sample.get('slug')}")
        assert_cjk_font_evidence(sample, f"showcase index sample {sample.get('slug')}")
        assert_ok(sample.get("overview_panel_note_layout") == "short_non_overlapping_panel_notes", f"showcase sample panel notes should be short: {sample.get('slug')}")
        overview_rel = sample["overview_png"]
        metadata_rel = sample["metadata_json"]
        assert_ok(overview_rel in readme, f"README does not reference {overview_rel}")
        overview_path = ROOT / overview_rel
        metadata_path = ROOT / metadata_rel
        assert_ok(overview_path.is_file(), f"missing showcase image {overview_rel}")
        assert_ok(metadata_path.is_file(), f"missing showcase metadata {metadata_rel}")
        with Image.open(overview_path) as image:
            assert_ok(image.size == (1500, 1120), f"showcase image size mismatch: {overview_rel}")
            assert_ok(image.format == "PNG", f"showcase image must be PNG: {overview_rel}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert_cjk_font_evidence(metadata, metadata_rel)
        source_issues = metadata.get("overview_cjk_source_string_issues", {})
        assert_ok(source_issues.get("replacement_char_count") == 0, f"unexpected replacement chars in showcase strings: {metadata_rel}")
        assert_ok(source_issues.get("replacement_char_fields") == [], f"unexpected replacement char fields: {metadata_rel}")
        assert_ok(metadata.get("overview_table_visible_rows") == 10, f"showcase table must expose 10 rows: {metadata_rel}")
        assert_ok(metadata.get("overview_table_no_clipping") is True, f"showcase table must not clip rows: {metadata_rel}")
        assert_ok(
            metadata.get("overview_rank_label_style") == "blue_badge_white_text_cyan_outline",
            f"showcase rank labels must avoid yellow/viridis-high styling: {metadata_rel}",
        )
        assert_ok(metadata.get("right_heatmap_patch_annotations") is False, f"right heatmap must not draw patch annotations: {metadata_rel}")
        assert_ok(metadata.get("right_heatmap_rank_labels") is False, f"right heatmap must not draw rank labels: {metadata_rel}")
        assert_ok(metadata.get("right_heatmap_bbox_overlay") is True, f"right heatmap must keep bbox overlay: {metadata_rel}")
        assert_ok(metadata.get("q_type") == "attribute", f"showcase metadata must be attribute: {metadata_rel}")
        assert_ok(metadata.get("vg_attention_mode") == "attribute_raw", f"showcase metadata must use attribute_raw: {metadata_rel}")
        assert_ok("__attribute__layer16" in metadata.get("sample_layer_id", ""), f"showcase sample_layer_id must be attribute: {metadata_rel}")
        assert_ok(metadata.get("attention_value_source") == "raw_attention", f"showcase metadata must use raw attention: {metadata_rel}")
        assert_ok(metadata.get("heatmap_value_source") == "raw_attention", f"showcase heatmap must use raw attention: {metadata_rel}")
        assert_ok(metadata.get("colorbar_value_source") == "raw_attention", f"showcase colorbar must use raw attention: {metadata_rel}")
        assert_ok(metadata.get("top_patch_rank_source") == "raw_attention", f"showcase top patches must preserve raw_attention rank: {metadata_rel}")
        assert_ok(metadata.get("source_layer") == 16, f"showcase source layer mismatch: {metadata_rel}")
        assert_ok(metadata.get("target_layer") == 27, f"showcase target layer mismatch: {metadata_rel}")
        assert_ok(metadata.get("lens") == "prefix_n50", f"showcase lens mismatch: {metadata_rel}")
        assert_ok(metadata.get("raw_source_files_copied") is False, f"showcase should not copy raw source files: {metadata_rel}")
        assert_ok(metadata.get("overview_canvas_size") == [1500, 1120], f"showcase metadata canvas mismatch: {metadata_rel}")
        assert_ok(metadata.get("overview_table_visible_rows") == 10, f"showcase metadata must expose all table rows: {metadata_rel}")
        assert_ok(metadata.get("overview_table_rows_available") == list(range(1, 11)), f"showcase metadata table rows mismatch: {metadata_rel}")
        assert_ok(len(metadata.get("overview_table_patch_ids_available", [])) == 10, f"showcase metadata table patch count mismatch: {metadata_rel}")
        assert_ok(metadata.get("overview_table_no_clipping") is True, f"showcase metadata must mark table as unclipped: {metadata_rel}")
        assert_ok(metadata.get("overview_rank_label_style") == "blue_badge_white_text_cyan_outline", f"showcase rank label style mismatch: {metadata_rel}")
        assert_ok(metadata.get("right_heatmap_patch_annotations") is False, f"showcase right heatmap patch annotation contract mismatch: {metadata_rel}")
        assert_ok(metadata.get("right_heatmap_rank_labels") is False, f"showcase right heatmap rank label contract mismatch: {metadata_rel}")
        assert_ok(metadata.get("right_heatmap_bbox_overlay") is True, f"showcase right heatmap bbox overlay contract mismatch: {metadata_rel}")
        assert_ok(metadata.get("overview_panel_note_layout") == "short_non_overlapping_panel_notes", f"showcase panel note layout mismatch: {metadata_rel}")
        transform = metadata.get("attention_map_transform", {})
        assert_ok(transform.get("patch_id_formula") == "patch_id = patch_row * 24 + patch_col", f"showcase patch id formula mismatch: {metadata_rel}")
        assert_ok(transform.get("origin") == "top_left", f"showcase heatmap origin mismatch: {metadata_rel}")
        assert_ok(transform.get("row_axis") == "y", f"showcase heatmap row axis mismatch: {metadata_rel}")
        assert_ok(transform.get("col_axis") == "x", f"showcase heatmap col axis mismatch: {metadata_rel}")
        assert_ok(transform.get("transpose") is False, f"showcase heatmap must not transpose: {metadata_rel}")
        assert_ok(transform.get("flip_x") is False, f"showcase heatmap must not flip x: {metadata_rel}")
        assert_ok(transform.get("flip_y") is False, f"showcase heatmap must not flip y: {metadata_rel}")
        heatmap_matrix = metadata.get("heatmap_matrix_24x24")
        assert_ok(isinstance(heatmap_matrix, list) and len(heatmap_matrix) == 24, f"showcase heatmap matrix must have 24 rows: {metadata_rel}")
        assert_ok(all(isinstance(row, list) and len(row) == 24 for row in heatmap_matrix), f"showcase heatmap matrix must be 24x24: {metadata_rel}")
        top_patches = metadata.get("top_patches", [])
        assert_ok(len(top_patches) == 10, f"showcase top patch count mismatch: {metadata_rel}")
        assert_ok(
            metadata.get("overview_table_patch_ids_available") == [patch["patch_id"] for patch in top_patches],
            f"showcase visible table patch ids must match top patch list: {metadata_rel}",
        )
        matrix_top10 = sorted(
            (
                {
                    "patch_id": row_idx * 24 + col_idx,
                    "raw_attention": value,
                }
                for row_idx, row in enumerate(heatmap_matrix)
                for col_idx, value in enumerate(row)
            ),
            key=lambda item: (-float(item["raw_attention"]), int(item["patch_id"])),
        )[:10]
        assert_ok(
            [item["patch_id"] for item in matrix_top10] == [patch["patch_id"] for patch in top_patches],
            f"showcase raw top10 must equal matrix global top10: {metadata_rel}",
        )
        assert_ok(abs(top_patches[0]["raw_attention"] - metadata.get("heatmap_display_max")) < 1e-12, f"raw rank1 patch must equal raw heatmap max: {metadata_rel}")
        assert_ok(top_patches[0]["patch_id"] == metadata.get("heatmap_display_max_patch_id"), f"raw rank1 patch id must equal raw heatmap max patch id: {metadata_rel}")
        previous_raw = None
        for index, patch in enumerate(top_patches, start=1):
            assert_ok("raw_attention" in patch and "normalized_attention" in patch, f"showcase patch must keep raw and normalized values: {metadata_rel}")
            assert_ok(patch.get("rank") == index, f"showcase rank sequence mismatch: {metadata_rel}")
            assert_ok(patch.get("raw_rank_desc") == index, f"showcase raw rank mismatch: {metadata_rel}")
            patch_row = patch.get("patch_row")
            patch_col = patch.get("patch_col")
            patch_id = patch.get("patch_id")
            assert_ok(patch_id == patch_row * 24 + patch_col, f"showcase patch id/row/col mismatch: {metadata_rel}")
            assert_ok(0 <= patch_row < 24 and 0 <= patch_col < 24, f"showcase patch row/col out of range: {metadata_rel}")
            assert_ok(patch.get("heatmap_row") == patch_row, f"showcase heatmap row mismatch: {metadata_rel}")
            assert_ok(patch.get("heatmap_col") == patch_col, f"showcase heatmap col mismatch: {metadata_rel}")
            matrix_value = heatmap_matrix[patch_row][patch_col]
            expected_value = patch[metadata["heatmap_value_source"]]
            assert_ok(abs(matrix_value - expected_value) < 1e-6, f"showcase heatmap matrix value mismatch: {metadata_rel}")
            assert_ok(abs(patch.get("heatmap_value") - expected_value) < 1e-6, f"showcase heatmap top patch value mismatch: {metadata_rel}")
            if previous_raw is not None:
                assert_ok(previous_raw >= patch["raw_attention"], f"showcase raw attention must be descending: {metadata_rel}")
            previous_raw = patch["raw_attention"]
        if sample.get("slug") == "dermoscopy_skin_lesion_layer16":
            dermoscopy_patch_ids = [patch["patch_id"] for patch in top_patches]
            assert_ok(dermoscopy_patch_ids == [252, 275, 253, 323, 322, 375, 299, 484, 324, 539], "dermoscopy raw top10 patch ids changed")
            assert_ok(484 in metadata.get("overview_table_patch_ids_available", []), "dermoscopy rank8 patch484 must be table-visible")
            assert_ok(539 in metadata.get("overview_table_patch_ids_available", []), "dermoscopy rank10 patch539 must be table-visible")
            dermoscopy_fix = metadata.get("dermoscopy_layout_fix_evidence", {})
            assert_ok(dermoscopy_fix.get("raw_top10_complete") is True, "dermoscopy fix must state raw top10 completeness")
            assert_ok(dermoscopy_fix.get("rank8_patch", {}).get("patch_id") == 484, "dermoscopy rank8 patch must remain 484")
            assert_ok(dermoscopy_fix.get("rank8_patch", {}).get("patch_row") == 20, "dermoscopy rank8 row must remain 20")
            assert_ok(dermoscopy_fix.get("rank8_patch", {}).get("patch_col") == 4, "dermoscopy rank8 col must remain 4")
            assert_ok(abs(dermoscopy_fix.get("rank8_patch", {}).get("raw_attention") - 0.001220703125) < 1e-12, "dermoscopy rank8 raw attention mismatch")
            assert_ok(dermoscopy_fix.get("rank10_patch", {}).get("patch_id") == 539, "dermoscopy rank10 patch must remain 539")
            assert_ok(dermoscopy_fix.get("rank10_patch", {}).get("patch_row") == 22, "dermoscopy rank10 row must remain 22")
            assert_ok(dermoscopy_fix.get("rank10_patch", {}).get("patch_col") == 11, "dermoscopy rank10 col must remain 11")
            assert_ok(abs(dermoscopy_fix.get("rank10_patch", {}).get("raw_attention") - 0.000946044921875) < 1e-12, "dermoscopy rank10 raw attention mismatch")

    forbidden = [
        "/" + "cpfs01",
        "gbw_" + "21307130160",
        "172" + ".20",
        "127" + ".0.0.1",
        "A" + "100",
        "CUDA_VISIBLE_DEVICES" + "=0",
        "models--" + "FreedomIntelligence",
        "COCO_" + "val" + "2014",
        "val" + "2014",
        "snap" + "shots/",
        ".cache/" + "huggingface",
        "huatuogpt_v7b_adapter/" + "Jlens/experiment",
    ]
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix not in {".png", ".zip", ".npy", ".pt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in forbidden:
                assert_ok(needle not in text, f"forbidden string {needle!r} in {path.relative_to(ROOT)}")

    assert_ok(not ROOT.with_suffix(".zip").exists(), "staging zip delivery should not exist")

    forbidden_names = {"stdout_stderr.log", "full_command.txt", "status_fail.txt"}
    for path in ROOT.rglob("*"):
        if path.is_file():
            name = path.name
            assert_ok(name not in forbidden_names, f"forbidden run artifact in staging package: {path.relative_to(ROOT)}")
            assert_ok(not (name.startswith("gpu_") and name.endswith(".txt")), f"GPU log found: {path.relative_to(ROOT)}")

    allowed_lens = ROOT / "weights/huatuo_jimage_lens_layer16_n50.pt"
    for path in ROOT.rglob("*.pt"):
        assert_ok(path == allowed_lens, f"unexpected .pt file in staging package: {path.relative_to(ROOT)}")
    assert_ok(allowed_lens.is_file(), "expected layer16 n50 lens .pt is missing")

    expected_lens_sha = "702c9a99d54e19d3b759e9238edcabc7696f7e07246c018a39c0035cb783fc5f"
    lens_meta_path = ROOT / "weights/huatuo_jimage_lens_layer16_n50.json"
    lens_meta = json.loads(lens_meta_path.read_text(encoding="utf-8"))
    assert_ok(lens_meta.get("source_layer") == 16, "metadata source_layer alias must be 16")
    assert_ok(lens_meta.get("source_layers") == [16], "metadata source_layers must be [16]")
    assert_ok(lens_meta.get("sha256") == expected_lens_sha, "metadata sha256 alias mismatch")
    assert_ok(lens_meta.get("lens_sha256") == expected_lens_sha, "metadata lens_sha256 mismatch")

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    stale_notice_phrases = [
        "does not include model weights, fitted lens " + "weights",
        "does not include fitted lens " + "weights",
    ]
    for phrase in stale_notice_phrases:
        assert_ok(phrase not in notice, f"stale NOTICE weight phrase remains: {phrase}")
    assert_ok("includes the layer16 n50 fitted JLens matrix" in notice, "NOTICE must state included layer16 n50 fitted JLens matrix")
    assert_ok("weights/README.md" in notice, "NOTICE must point to weights/README.md")

    for html_path in [ROOT / "index.html", ROOT / "examples/fixture_demo/index.html"]:
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            for stale in ["Official J-lens", "Qwen3.5", "layer14", "layer 14"]:
                assert_ok(stale not in html, f"stale term {stale!r} in {html_path.relative_to(ROOT)}")

    print("PASS static staging checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
