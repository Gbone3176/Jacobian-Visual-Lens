#!/usr/bin/env python3
"""Fit HuatuoGPT-V image-token Jacobian lenses.

This entrypoint is smoke-ready for small N runs.  Importing it or running
``--help`` does not load Huatuo, transformers, torch CUDA state, or model
weights; runtime imports happen only after CLI parsing succeeds.
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
sys.path.insert(0, str(SRC_ROOT))

from huatuo_jlens_core import (  # noqa: E402
    atomic_torch_save,
    build_fit_metadata,
    load_jsonl_rows,
    parse_layers,
    parse_prefix_save_at,
    save_jacobian_lens,
    sha256_file,
    write_json,
)


DEFAULT_SOURCE_LAYERS = "8,10,12,14,16,18,20,22,24,26,27"
TARGET_CROP_CHOICES = ("none", "head", "head_tail", "uniform")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(message, flush=True)


def select_target_positions(target_positions: Any, strategy: str, max_target_tokens: int | None, torch: Any) -> Any:
    count = int(target_positions.numel())
    if count == 0:
        raise RuntimeError("teacher-forced target positions are empty")
    if max_target_tokens is None or max_target_tokens <= 0 or count <= max_target_tokens:
        return target_positions
    keep = int(max_target_tokens)
    if strategy == "none":
        return target_positions
    if strategy == "head":
        return target_positions[:keep]
    if strategy == "uniform":
        idx = torch.linspace(0, count - 1, steps=keep, device=target_positions.device).round().long()
        return target_positions.index_select(0, torch.unique(idx, sorted=True))
    if strategy == "head_tail":
        head = (keep + 1) // 2
        tail = keep - head
        if tail == 0:
            return target_positions[:head]
        return torch.cat([target_positions[:head], target_positions[-tail:]])
    raise ValueError(f"unknown target crop strategy: {strategy}")


def import_huatuo_runtime():
    from huatuo_single_sample_readout import load_model  # noqa: WPS433
    from huatuo_single_sample_runtime import (  # noqa: WPS433
        IGNORE_INDEX,
        image_to_tensor,
        prepare_multimodal,
        read_jsonl_row,
        tokenizer_image_token,
    )

    return {
        "IGNORE_INDEX": IGNORE_INDEX,
        "image_to_tensor": image_to_tensor,
        "load_model": load_model,
        "prepare_multimodal": prepare_multimodal,
        "read_jsonl_row": read_jsonl_row,
        "tokenizer_image_token": tokenizer_image_token,
    }


def prepare_fit_sample(
    *,
    runtime: dict[str, Any],
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_token_index: int,
    model_path: Path,
    corpus_path: Path,
    row_index: int,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    sample = runtime["read_jsonl_row"](corpus_path, row_index)
    image_path = Path(sample["image"]).resolve()
    prefix = f"<|user|>\n<image>\n{sample['prompt']}\n<|assistant|>\n"
    answer_text = f"{sample['answer']} \n"
    full_prompt = prefix + answer_text

    prefix_ids_1d = runtime["tokenizer_image_token"](tokenizer, prefix, image_token_index, return_tensors="pt")
    full_ids_1d = runtime["tokenizer_image_token"](tokenizer, full_prompt, image_token_index, return_tensors="pt")
    placeholder_count = int((full_ids_1d == image_token_index).sum().item())
    if placeholder_count != 1:
        raise RuntimeError(f"expected one image placeholder, got {placeholder_count}")
    raw_image_pos = int(torch.where(full_ids_1d == image_token_index)[0][0].item())

    image_tensor, original_image_size, square_image_size = runtime["image_to_tensor"](image_path, processor, device)
    full_ids = full_ids_1d.unsqueeze(0).to(device)
    raw_labels_1d = torch.full_like(full_ids_1d, runtime["IGNORE_INDEX"])
    raw_answer_start = int(prefix_ids_1d.shape[0])
    raw_labels_1d[raw_answer_start:] = full_ids_1d[raw_answer_start:]
    raw_labels = raw_labels_1d.unsqueeze(0).to(device)

    _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = runtime["prepare_multimodal"](
        model, full_ids, raw_labels, [image_tensor.to(dtype=torch.bfloat16)]
    )
    image_token_count = int(inputs_embeds.shape[1] - (full_ids_1d.shape[0] - 1))
    source_span = [raw_image_pos, raw_image_pos + image_token_count]
    target_positions = torch.where(expanded_labels[0] != runtime["IGNORE_INDEX"])[0]
    if target_positions.numel() == 0:
        raise RuntimeError("expanded answer target span is empty")

    return {
        "sample": sample,
        "row_index": row_index,
        "image_path": str(image_path),
        "original_image_size": list(original_image_size),
        "square_image_size": list(square_image_size),
        "inputs_embeds": inputs_embeds.detach(),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "source_span": source_span,
        "target_positions": target_positions,
        "target_span": [int(target_positions[0].item()), int(target_positions[-1].item()) + 1],
        "placeholder_count": placeholder_count,
        "raw_prefix_len": int(prefix_ids_1d.shape[0]),
        "raw_full_len": int(full_ids_1d.shape[0]),
        "expanded_seq_len": int(inputs_embeds.shape[1]),
        "image_token_count": image_token_count,
        "model_path": str(model_path),
        "corpus_path": str(corpus_path),
    }


def compute_layer_jacobian(
    *,
    model: Any,
    prepared: dict[str, Any],
    source_layer_idx: int,
    target_layer_idx: int,
    target_positions: Any,
    dim_batch: int,
    device: str,
    torch: Any,
) -> tuple[Any, int]:
    d_model = int(model.config.hidden_size)
    source_start, source_end = prepared["source_span"]
    source_positions = torch.arange(source_start, source_end, device=device)
    n_passes = (d_model + dim_batch - 1) // dim_batch
    jacobian = torch.zeros(d_model, d_model, dtype=torch.float32)
    max_memory = 0

    inputs_base = prepared["inputs_embeds"].to(device=device, dtype=torch.bfloat16)
    attention_mask_base = prepared["attention_mask"].to(device) if prepared["attention_mask"] is not None else None
    position_ids_base = prepared["position_ids"].to(device) if prepared["position_ids"] is not None else None
    target_positions = target_positions.to(device)

    for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
        n_dims = min(dim_batch, d_model - dim_start)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        source_input: dict[str, Any] = {}
        target_output: dict[str, Any] = {}

        def source_pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
            hidden_states = inputs[0]
            hidden_states.retain_grad()
            source_input["tensor"] = hidden_states

        def target_fwd_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            target_output["tensor"] = output[0] if isinstance(output, tuple) else output

        hooks = [
            model.model.layers[source_layer_idx].register_forward_pre_hook(source_pre_hook),
            model.model.layers[target_layer_idx].register_forward_hook(target_fwd_hook),
        ]
        try:
            repeated_inputs = inputs_base.expand(n_dims, -1, -1).contiguous().detach().requires_grad_(True)
            repeated_attention = (
                attention_mask_base.expand(n_dims, -1).contiguous() if attention_mask_base is not None else None
            )
            repeated_position = (
                position_ids_base.expand(n_dims, -1).contiguous() if position_ids_base is not None else None
            )
            with torch.enable_grad():
                model.model(
                    input_ids=None,
                    inputs_embeds=repeated_inputs,
                    attention_mask=repeated_attention,
                    position_ids=repeated_position,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                cotangent = torch.zeros_like(target_output["tensor"])
                batch_idx = torch.arange(n_dims, device=device)
                dim_idx = dim_start + batch_idx
                cotangent[batch_idx[:, None], target_positions[None, :], dim_idx[:, None]] = 1.0
                grads = torch.autograd.grad(
                    outputs=target_output["tensor"],
                    inputs=source_input["tensor"],
                    grad_outputs=cotangent,
                    retain_graph=False,
                )[0]
                rows = grads[:n_dims, source_positions, :].float().mean(dim=1)
                jacobian[dim_start : dim_start + n_dims, :] = rows.detach().cpu()
        finally:
            for hook in hooks:
                hook.remove()
        if torch.cuda.is_available():
            max_memory = max(max_memory, int(torch.cuda.max_memory_allocated(device)))
            torch.cuda.empty_cache()
        if pass_idx == 0 or pass_idx == n_passes - 1 or (pass_idx + 1) % 64 == 0:
            log(
                f"layer={source_layer_idx} target_layer={target_layer_idx} "
                f"dims={dim_start + n_dims}/{d_model}"
            )
        del source_input, target_output
        if "repeated_inputs" in locals():
            del repeated_inputs
        if "grads" in locals():
            del grads
        if "cotangent" in locals():
            del cotangent
    return jacobian, max_memory


def mean_jacobians(jacobian_sums: dict[int, Any], success_count: int) -> dict[int, Any]:
    if success_count <= 0:
        raise RuntimeError("cannot average Jacobians with zero successful samples")
    return {layer: value / success_count for layer, value in jacobian_sums.items()}


def save_outputs(
    *,
    out_dir: Path,
    jacobian_sums: dict[int, Any],
    success_count: int,
    d_model: int,
    metadata: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    lens = mean_jacobians(jacobian_sums, success_count)
    save_jacobian_lens(lens, success_count, d_model, out_dir / "huatuo_jimage_lens.pt")
    write_json(out_dir / "huatuo_jimage_lens.json", metadata)
    atomic_torch_save(checkpoint, out_dir / "huatuo_jimage_lens.fit_checkpoint.pt")


def run_fit(args: argparse.Namespace) -> int:
    import torch

    runtime = import_huatuo_runtime()
    started_at = now_utc()
    source_layers = parse_layers(args.source_layers)
    prefix_save_at = set(parse_prefix_save_at(args.prefix_save_at))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_rows(Path(args.corpus_path), row_start=args.row_start, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("no corpus rows selected")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(int(args.device.split(":")[-1]))

    model, tokenizer, processor, image_token_index, loading_info = runtime["load_model"](
        args.model_path, args.support_repo, args.device
    )
    d_model = int(model.config.hidden_size)
    jacobian_sums: dict[int, Any] = {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers}
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    max_memory = 0

    for offset, _row in enumerate(rows):
        row_index = int(args.row_start + offset)
        sample_started = now_utc()
        try:
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
            target_crop = select_target_positions(
                prepared["target_positions"],
                args.target_crop_strategy,
                args.max_target_tokens,
                torch,
            )
            sample_jacobians: dict[int, Any] = {}
            for source_layer in source_layers:
                jacobian, peak = compute_layer_jacobian(
                    model=model,
                    prepared=prepared,
                    source_layer_idx=source_layer,
                    target_layer_idx=args.target_layer,
                    target_positions=target_crop,
                    dim_batch=args.dim_batch,
                    device=args.device,
                    torch=torch,
                )
                sample_jacobians[source_layer] = jacobian
                max_memory = max(max_memory, peak)
            for source_layer, jacobian in sample_jacobians.items():
                jacobian_sums[source_layer] += jacobian
            successes.append(
                {
                    "row_index": row_index,
                    "sample_id": prepared["sample"].get("id"),
                    "source_span": prepared["source_span"],
                    "source_count": prepared["image_token_count"],
                    "target_span": prepared["target_span"],
                    "target_crop": [int(x) for x in target_crop.detach().cpu().tolist()],
                    "started_at_utc": sample_started,
                    "ended_at_utc": now_utc(),
                }
            )
            success_count = len(successes)
            checkpoint = {
                "jacobian_sums": jacobian_sums,
                "success_count": success_count,
                "failures": failures,
                "source_layers": source_layers,
                "target_layer": int(args.target_layer),
                "d_model": d_model,
            }
            metadata = build_fit_metadata(
                source_layers=source_layers,
                target_layer=args.target_layer,
                n_prompts=success_count,
                d_model=d_model,
                model_path=str(Path(args.model_path).resolve()),
                corpus_path=str(Path(args.corpus_path).resolve()),
                target_crop_strategy=args.target_crop_strategy,
                max_target_tokens=args.max_target_tokens,
                dim_batch=args.dim_batch,
                top_k=args.top_k,
                extra={
                    "schema": "huatuo_jlens_fit_result_v1",
                    "status": "RUNNING",
                    "started_at_utc": started_at,
                    "updated_at_utc": now_utc(),
                    "row_start": args.row_start,
                    "max_samples": args.max_samples,
                    "success_count": success_count,
                    "failure_count": len(failures),
                    "successes": successes,
                    "failures": failures,
                    "peak_memory_gib": round(max_memory / (1024**3), 3),
                    "lens_path": str(out_dir / "huatuo_jimage_lens.pt"),
                    "checkpoint_path": str(out_dir / "huatuo_jimage_lens.fit_checkpoint.pt"),
                },
            )
            save_outputs(
                out_dir=out_dir,
                jacobian_sums=jacobian_sums,
                success_count=success_count,
                d_model=d_model,
                metadata=metadata,
                checkpoint=checkpoint,
            )
            if success_count in prefix_save_at:
                prefix_dir = out_dir / f"prefix_n{success_count}"
                prefix_metadata = dict(metadata, status="PREFIX", prefix_n=success_count)
                save_jacobian_lens(
                    mean_jacobians(jacobian_sums, success_count),
                    success_count,
                    d_model,
                    prefix_dir / "huatuo_jimage_lens.pt",
                )
                write_json(prefix_dir / "huatuo_jimage_lens.json", prefix_metadata)
        except Exception as exc:  # noqa: BLE001 - failure isolation is required.
            failures.append(
                {
                    "row_index": row_index,
                    "started_at_utc": sample_started,
                    "ended_at_utc": now_utc(),
                    "fail_reason": str(exc),
                    "traceback_summary": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    "stage": "fit_sample",
                    "retryable": True,
                }
            )
            log(f"sample row_index={row_index} failed: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not successes:
        write_json(
            out_dir / "huatuo_jimage_lens.json",
            {
                "schema": "huatuo_jlens_fit_result_v1",
                "status": "FAIL",
                "started_at_utc": started_at,
                "ended_at_utc": now_utc(),
                "success_count": 0,
                "failure_count": len(failures),
                "failures": failures,
            },
        )
        raise RuntimeError("0 successful samples; no lens saved")

    final_metadata = build_fit_metadata(
        source_layers=source_layers,
        target_layer=args.target_layer,
        n_prompts=len(successes),
        d_model=d_model,
        model_path=str(Path(args.model_path).resolve()),
        corpus_path=str(Path(args.corpus_path).resolve()),
        target_crop_strategy=args.target_crop_strategy,
        max_target_tokens=args.max_target_tokens,
        dim_batch=args.dim_batch,
        top_k=args.top_k,
        extra={
            "schema": "huatuo_jlens_fit_result_v1",
            "status": "PASS",
            "started_at_utc": started_at,
            "ended_at_utc": now_utc(),
            "row_start": args.row_start,
            "max_samples": args.max_samples,
            "success_count": len(successes),
            "failure_count": len(failures),
            "successes": successes,
            "failures": failures,
            "peak_memory_gib": round(max_memory / (1024**3), 3),
            "lens_path": str(out_dir / "huatuo_jimage_lens.pt"),
            "checkpoint_path": str(out_dir / "huatuo_jimage_lens.fit_checkpoint.pt"),
            "lens_sha256": sha256_file(out_dir / "huatuo_jimage_lens.pt"),
        },
    )
    save_outputs(
        out_dir=out_dir,
        jacobian_sums=jacobian_sums,
        success_count=len(successes),
        d_model=d_model,
        metadata=final_metadata,
        checkpoint={
            "jacobian_sums": jacobian_sums,
            "success_count": len(successes),
            "failures": failures,
            "source_layers": source_layers,
            "target_layer": int(args.target_layer),
            "d_model": d_model,
        },
    )
    log(json.dumps({"status": "PASS", "out_dir": str(out_dir), "success_count": len(successes)}, indent=2))
    del model, jacobian_sums
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--support-repo", required=True)
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output run directory. For this open-source package, use an explicit path such as runs/fit_layer16_n50.",
    )
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--source-layers", default=DEFAULT_SOURCE_LAYERS)
    parser.add_argument("--target-layer", type=int, default=27)
    parser.add_argument("--dim-batch", type=int, default=1)
    parser.add_argument("--target-crop-strategy", choices=TARGET_CROP_CHOICES, default="head_tail")
    parser.add_argument("--max-target-tokens", type=int, default=64)
    parser.add_argument("--prefix-save-at")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.dim_batch <= 0:
        raise ValueError("--dim-batch must be positive")
    if args.max_target_tokens is not None and args.max_target_tokens <= 0:
        raise ValueError("--max-target-tokens must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    return run_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
