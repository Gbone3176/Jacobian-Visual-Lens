#!/usr/bin/env python3
"""Evaluate a fitted HuatuoGPT-V JacobianLens on heldout rows.

Importing this module or running ``--help`` is static: it does not import
transformers, load Huatuo weights, touch CUDA state, or run a forward pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = STAGE_ROOT / "src"
FIT_ROOT = STAGE_ROOT / "scripts" / "fit"
EVAL_ROOT = STAGE_ROOT / "scripts" / "eval"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(FIT_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

from huatuo_jlens_core import load_jacobian_lens, load_jsonl_rows, parse_layers, write_json  # noqa: E402
from huatuo_single_sample_f1 import aggregate_outputs, evaluate_record, parse_rules  # noqa: E402


DEFAULT_RULES = "sec9_raw"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(message, flush=True)


def import_runtime_helpers():
    import torch  # noqa: WPS433
    from huatuo_fit_jlens import import_huatuo_runtime, prepare_fit_sample  # noqa: WPS433
    from huatuo_single_sample_readout import build_readout_rows, top_tokens  # noqa: WPS433

    return torch, import_huatuo_runtime, prepare_fit_sample, build_readout_rows, top_tokens


def lens_source_layers(lens: Any) -> list[int]:
    raw = getattr(lens, "source_layers", None)
    if raw is not None:
        return [int(layer) for layer in raw]
    jacobians = getattr(lens, "jacobians", None)
    if isinstance(jacobians, dict):
        return sorted(int(layer) for layer in jacobians)
    raise ValueError("cannot determine layers from lens: missing source_layers and jacobians")


def resolve_layers(spec: str | None, lens: Any) -> list[int]:
    layers = parse_layers(spec) if spec else lens_source_layers(lens)
    available = set(lens_source_layers(lens))
    missing = sorted(set(layers) - available)
    if missing:
        raise ValueError(f"requested layers {missing} not found in lens; available={sorted(available)}")
    return layers


def capture_source_residuals(
    *,
    model: Any,
    prepared: dict[str, Any],
    layers: list[int],
    device: str,
    torch: Any,
) -> dict[int, Any]:
    holders: dict[int, Any] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            holders[layer_idx] = inputs[0].detach()

        return hook

    for layer_idx in layers:
        hooks.append(model.model.layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx)))
    try:
        with torch.no_grad():
            model.model(
                input_ids=None,
                inputs_embeds=prepared["inputs_embeds"].to(device=device, dtype=torch.bfloat16),
                attention_mask=prepared["attention_mask"].to(device) if prepared["attention_mask"] is not None else None,
                position_ids=prepared["position_ids"].to(device) if prepared["position_ids"] is not None else None,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()
    missing = [layer for layer in layers if layer not in holders]
    if missing:
        raise RuntimeError(f"did not capture residuals for layers {missing}")
    return holders


def readout_layer(
    *,
    model: Any,
    tokenizer: Any,
    lens: Any,
    layer: int,
    source_residual: Any,
    source_span: list[int],
    top_k: int,
    torch: Any,
) -> tuple[list[dict[str, Any]], list[list[str]], list[list[str]]]:
    source_start, source_end = source_span
    residual = source_residual[0, source_start:source_end, :].detach()
    from huatuo_single_sample_readout import build_readout_rows, top_tokens  # noqa: WPS433

    readout_rows = build_readout_rows(int(residual.shape[0]))
    offsets = torch.tensor([row["source_offset"] for row in readout_rows], device=residual.device)
    selected_source = residual.index_select(0, offsets)
    jacobian = lens.jacobians[int(layer)].to(device=residual.device, dtype=torch.float32)
    transported = residual.float() @ jacobian.T
    selected_transported = transported.index_select(0, offsets).to(dtype=torch.bfloat16)
    selected_source = selected_source.to(dtype=torch.bfloat16)
    with torch.no_grad():
        j_logits = model.lm_head(model.model.norm(selected_transported))
        base_logits = model.lm_head(model.model.norm(selected_source))

    details: list[dict[str, Any]] = []
    j_texts: list[list[str]] = []
    base_texts: list[list[str]] = []
    for idx, row in enumerate(readout_rows):
        j_top = top_tokens(tokenizer, j_logits[idx], top_k)
        base_top = top_tokens(tokenizer, base_logits[idx], top_k)
        details.append({**row, "j_lens": j_top, "logit_lens": base_top})
        j_texts.append([item["text"] for item in j_top])
        base_texts.append([item["text"] for item in base_top])
    return details, j_texts, base_texts


def build_readout_record(
    *,
    prepared: dict[str, Any],
    layers: list[int],
    j_lens_token_texts: dict[str, list[list[str]]],
    logit_lens_token_texts: dict[str, list[list[str]]],
    readout_details: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    sample = prepared["sample"]
    sample_id = sample.get("id") or sample.get("sample_id") or f"row_{prepared['row_index']}"
    return {
        "schema": "huatuo_fitted_lens_eval_readout_v1",
        "id": sample_id,
        "sample_id": sample_id,
        "dataset": sample.get("dataset"),
        "image_path": prepared["image_path"],
        "prompt": sample.get("prompt"),
        "question": sample.get("prompt"),
        "answer": sample.get("answer"),
        "gold_text": sample.get("answer"),
        "top_k": int(top_k),
        "layers": [int(layer) for layer in layers],
        "j_lens_token_texts": j_lens_token_texts,
        "logit_lens_token_texts": logit_lens_token_texts,
        "readout_details": readout_details,
        "source_span": prepared["source_span"],
        "source_count": prepared["image_token_count"],
        "target_span": prepared["target_span"],
    }


def evaluate_sample(
    *,
    args: argparse.Namespace,
    row_index: int,
    lens: Any,
    layers: list[int],
    rules: list[str],
    runtime: dict[str, Any],
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_token_index: int,
    torch: Any,
    prepare_fit_sample: Any,
) -> dict[str, Any]:
    prepared = prepare_fit_sample(
        runtime=runtime,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        image_token_index=image_token_index,
        model_path=Path(args.model_path).resolve(),
        corpus_path=Path(args.corpus_path).resolve(),
        row_index=row_index,
        device=args.device,
        torch=torch,
    )
    captured = capture_source_residuals(
        model=model,
        prepared=prepared,
        layers=layers,
        device=args.device,
        torch=torch,
    )
    j_lens_token_texts: dict[str, list[list[str]]] = {}
    logit_lens_token_texts: dict[str, list[list[str]]] = {}
    readout_details: dict[str, list[dict[str, Any]]] = {}
    for layer in layers:
        details, j_texts, base_texts = readout_layer(
            model=model,
            tokenizer=tokenizer,
            lens=lens,
            layer=layer,
            source_residual=captured[layer],
            source_span=prepared["source_span"],
            top_k=args.top_k,
            torch=torch,
        )
        j_lens_token_texts[str(layer)] = j_texts
        logit_lens_token_texts[str(layer)] = base_texts
        readout_details[str(layer)] = details
    readout_record = build_readout_record(
        prepared=prepared,
        layers=layers,
        j_lens_token_texts=j_lens_token_texts,
        logit_lens_token_texts=logit_lens_token_texts,
        readout_details=readout_details,
        top_k=args.top_k,
    )
    f1 = evaluate_record(readout_record, rules=rules)
    return {
        "row_index": row_index,
        "sample_id": readout_record["sample_id"],
        "status": "success",
        "readout_record": readout_record,
        "f1": f1,
    }


def run_eval(args: argparse.Namespace) -> int:
    torch, import_huatuo_runtime, prepare_fit_sample, _build_readout_rows, _top_tokens = import_runtime_helpers()
    del _build_readout_rows, _top_tokens
    started_at = now_utc()
    rules = parse_rules(args.rules)
    lens = load_jacobian_lens(Path(args.lens_path))
    layers = resolve_layers(args.layers, lens)
    rows = load_jsonl_rows(Path(args.corpus_path), row_start=args.row_start, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("no corpus rows selected")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(int(args.device.split(":")[-1]))

    runtime = import_huatuo_runtime()
    model, tokenizer, processor, image_token_index, loading_info = runtime["load_model"](
        args.model_path, args.support_repo, args.device
    )
    output_json = Path(args.output_json)
    partial_jsonl = Path(args.partial_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    partial_jsonl.parent.mkdir(parents=True, exist_ok=True)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with partial_jsonl.open("w", encoding="utf-8") as partial:
        for offset, _row in enumerate(rows):
            row_index = int(args.row_start + offset)
            sample_started = now_utc()
            try:
                result = evaluate_sample(
                    args=args,
                    row_index=row_index,
                    lens=lens,
                    layers=layers,
                    rules=rules,
                    runtime=runtime,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    image_token_index=image_token_index,
                    torch=torch,
                    prepare_fit_sample=prepare_fit_sample,
                )
                successes.append(result)
                partial.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                partial.flush()
            except Exception as exc:  # noqa: BLE001 - failure isolation is required.
                failure = {
                    "row_index": row_index,
                    "status": "failed",
                    "started_at_utc": sample_started,
                    "ended_at_utc": now_utc(),
                    "fail_reason": str(exc),
                    "traceback_summary": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    "stage": "eval_sample",
                    "retryable": True,
                }
                failures.append(failure)
                partial.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
                partial.flush()
                log(f"sample row_index={row_index} failed: {exc}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not successes:
        write_json(
            output_json,
            {
                "schema": "huatuo_fitted_lens_eval_summary_v1",
                "status": "FAIL",
                "started_at_utc": started_at,
                "ended_at_utc": now_utc(),
                "success_count": 0,
                "failure_count": len(failures),
                "failures": failures,
                "lens_path": str(Path(args.lens_path).resolve()),
                "layers": layers,
                "top_k": args.top_k,
                "partial_jsonl": str(partial_jsonl),
            },
        )
        raise RuntimeError("0 successful eval samples")

    aggregate = aggregate_outputs([row["f1"] for row in successes], rules)
    summary = {
        "schema": "huatuo_fitted_lens_eval_summary_v1",
        "status": "PASS",
        "started_at_utc": started_at,
        "ended_at_utc": now_utc(),
        "success_count": len(successes),
        "failure_count": len(failures),
        "failures": failures,
        "lens_path": str(Path(args.lens_path).resolve()),
        "layers": layers,
        "top_k": args.top_k,
        "rules": rules,
        "row_start": args.row_start,
        "max_samples": args.max_samples,
        "partial_jsonl": str(partial_jsonl),
        "aggregate_f1": aggregate,
        "loading_info": {
            "missing_key_count": len(loading_info.get("missing_keys", [])),
            "unexpected_key_count": len(loading_info.get("unexpected_keys", [])),
        },
    }
    write_json(output_json, summary)
    log(json.dumps({"status": "PASS", "output_json": str(output_json), "success_count": len(successes)}, indent=2))
    del model, lens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--support-repo", required=True)
    parser.add_argument("--lens-path", required=True)
    parser.add_argument("--layers")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--partial-jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rules", default=DEFAULT_RULES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
