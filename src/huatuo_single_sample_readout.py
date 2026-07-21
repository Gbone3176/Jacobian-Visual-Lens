#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))
from huatuo_single_sample_runtime import (  # noqa: E402
    IGNORE_INDEX,
    image_to_tensor,
    prepare_multimodal,
    read_jsonl_row,
    tokenizer_image_token,
)


def log(message):
    print(message, flush=True)


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def decode_token(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    text = text.replace("\ufffd", "").strip()
    if not text or "<|" in text:
        return None
    return text


def top_tokens(tokenizer, logits, top_k):
    k = min(max(top_k * 8, top_k), logits.shape[-1])
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    rows = []
    for score, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        text = decode_token(tokenizer, token_id)
        if text is None:
            continue
        rows.append({"token_id": int(token_id), "text": text, "score": float(score)})
        if len(rows) >= top_k:
            break
    return rows


def build_readout_rows(source_token_count):
    side = int(round(source_token_count ** 0.5))
    rows = []

    def add(name, y, x):
        y = max(0, min(side - 1, int(y)))
        x = max(0, min(side - 1, int(x)))
        idx = y * side + x
        if idx < source_token_count:
            rows.append({"name": name, "source_offset": idx, "grid_y": y, "grid_x": x})

    add("upper_center", side // 4, side // 2)
    add("center", side // 2, side // 2)
    add("lower_center", (side * 3) // 4, side // 2)
    coords = torch.linspace(0, side - 1, steps=5).round().long().tolist()
    for y in coords:
        for x in coords:
            add(f"grid_{y}_{x}", y, x)

    seen = set()
    unique = []
    for row in rows:
        key = row["source_offset"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def load_model(model_path, support_repo, device):
    support_repo = Path(support_repo).resolve()
    sys.path.insert(0, str(support_repo))
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
        str(model_path),
        init_vision_encoder_from_ckpt=False,
        output_loading_info=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.tokenizer = tokenizer
    model.config.tokenizer_padding_side = "left"
    model.eval()
    model.to(device)
    model.requires_grad_(False)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16, device=device)
    return model, tokenizer, vision_tower.image_processor, IMAGE_TOKEN_INDEX, loading_info


def prepare_sample(model, tokenizer, processor, image_token_index, model_path, corpus_path, row_index, device):
    sample = read_jsonl_row(corpus_path, row_index)
    image_path = Path(sample["image"]).resolve()
    prefix = f"<|user|>\n<image>\n{sample['prompt']}\n<|assistant|>\n"
    answer_text = f"{sample['answer']} \n"
    full_prompt = prefix + answer_text

    prefix_ids_1d = tokenizer_image_token(tokenizer, prefix, image_token_index, return_tensors="pt")
    full_ids_1d = tokenizer_image_token(tokenizer, full_prompt, image_token_index, return_tensors="pt")
    placeholder_count = int((full_ids_1d == image_token_index).sum().item())
    if placeholder_count != 1:
        raise RuntimeError(f"expected one image placeholder, got {placeholder_count}")
    raw_image_pos = int(torch.where(full_ids_1d == image_token_index)[0][0].item())

    image_tensor, original_image_size, square_image_size = image_to_tensor(image_path, processor, device)
    full_ids = full_ids_1d.unsqueeze(0).to(device)
    raw_labels_1d = torch.full_like(full_ids_1d, IGNORE_INDEX)
    raw_answer_start = int(prefix_ids_1d.shape[0])
    raw_labels_1d[raw_answer_start:] = full_ids_1d[raw_answer_start:]
    raw_labels = raw_labels_1d.unsqueeze(0).to(device)

    _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = prepare_multimodal(
        model, full_ids, raw_labels, [image_tensor.to(dtype=torch.bfloat16)]
    )
    image_token_count = int(inputs_embeds.shape[1] - (full_ids_1d.shape[0] - 1))
    source_span = [raw_image_pos, raw_image_pos + image_token_count]
    target_positions = torch.where(expanded_labels[0] != IGNORE_INDEX)[0]
    target_crop = target_positions[: min(4, int(target_positions.numel()))]
    if target_crop.numel() != 4:
        raise RuntimeError(f"expected 4 target crop positions, got {int(target_crop.numel())}")

    return {
        "sample": sample,
        "image_path": str(image_path),
        "original_image_size": list(original_image_size),
        "square_image_size": list(square_image_size),
        "inputs_embeds": inputs_embeds.detach(),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "source_span": source_span,
        "target_span": [int(target_positions[0].item()), int(target_positions[-1].item()) + 1],
        "target_crop_positions": target_crop,
        "placeholder_count": placeholder_count,
        "raw_prefix_len": int(prefix_ids_1d.shape[0]),
        "raw_full_len": int(full_ids_1d.shape[0]),
        "expanded_seq_len": int(inputs_embeds.shape[1]),
        "image_token_count": image_token_count,
        "model_path": str(model_path),
        "corpus_path": str(corpus_path),
        "row_index": row_index,
    }


def compute_jacobian(model, prepared, dim_batch, progress_path, device):
    d_model = int(model.config.hidden_size)
    source_start, source_end = prepared["source_span"]
    target_crop = prepared["target_crop_positions"].to(device)
    source_positions = torch.arange(source_start, source_end, device=device)
    n_passes = (d_model + dim_batch - 1) // dim_batch
    J = torch.zeros(d_model, d_model, dtype=torch.float32)
    started = time.time()
    max_memory = 0

    inputs_base = prepared["inputs_embeds"].to(device=device, dtype=torch.bfloat16)
    attention_mask_base = prepared["attention_mask"].to(device) if prepared["attention_mask"] is not None else None
    position_ids_base = prepared["position_ids"].to(device) if prepared["position_ids"] is not None else None

    for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
        n_dims = min(dim_batch, d_model - dim_start)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        layer14_input = {}
        layer14_output = {}

        def pre_hook(module, inputs):
            hidden_states = inputs[0]
            hidden_states.retain_grad()
            layer14_input["tensor"] = hidden_states

        def fwd_hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            layer14_output["tensor"] = hidden_states

        hooks = [
            model.model.layers[14].register_forward_pre_hook(pre_hook),
            model.model.layers[14].register_forward_hook(fwd_hook),
        ]
        try:
            repeated_inputs = inputs_base.expand(n_dims, -1, -1).contiguous().detach().requires_grad_(True)
            repeated_attention = (
                attention_mask_base.expand(n_dims, -1).contiguous()
                if attention_mask_base is not None
                else None
            )
            repeated_position = (
                position_ids_base.expand(n_dims, -1).contiguous()
                if position_ids_base is not None
                else None
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
                target_activation = layer14_output["tensor"]
                cotangent = torch.zeros_like(target_activation)
                batch_idx = torch.arange(n_dims, device=device)
                dim_idx = dim_start + batch_idx
                cotangent[batch_idx[:, None], target_crop[None, :], dim_idx[:, None]] = 1.0
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=layer14_input["tensor"],
                    grad_outputs=cotangent,
                    retain_graph=False,
                )[0]
                rows = grads[:n_dims, source_positions, :].float().mean(dim=1)
                J[dim_start : dim_start + n_dims, :] = rows.detach().cpu()
        finally:
            for hook in hooks:
                hook.remove()
        max_memory = max(
            max_memory,
            int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        )
        del layer14_input, layer14_output
        if "repeated_inputs" in locals():
            del repeated_inputs
        if "grads" in locals():
            del grads
        if "cotangent" in locals():
            del cotangent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pass_idx == 0 or pass_idx == n_passes - 1 or (pass_idx + 1) % 16 == 0:
            progress = {
                "status": "RUNNING",
                "updated_at_utc": now_utc(),
                "dim_batch": dim_batch,
                "completed_dims": dim_start + n_dims,
                "total_dims": d_model,
                "n_passes": n_passes,
                "pass_idx": pass_idx,
                "elapsed_s": round(time.time() - started, 3),
                "max_memory_allocated_bytes": max_memory,
            }
            write_json(progress_path, progress)
            log(f"progress dims={progress['completed_dims']}/{d_model} elapsed_s={progress['elapsed_s']}")

    write_json(
        progress_path,
        {
            "status": "PASS",
            "updated_at_utc": now_utc(),
            "dim_batch": dim_batch,
            "completed_dims": d_model,
            "total_dims": d_model,
            "elapsed_s": round(time.time() - started, 3),
            "max_memory_allocated_bytes": max_memory,
        },
    )
    return J, time.time() - started, max_memory


def readout(model, tokenizer, J, source_residual, source_span, top_k):
    source_start, source_end = source_span
    source_residual = source_residual[0, source_start:source_end, :].detach()
    rows = build_readout_rows(int(source_residual.shape[0]))
    offsets = torch.tensor([row["source_offset"] for row in rows], device=source_residual.device)
    selected_source = source_residual.index_select(0, offsets)

    J_gpu = J.to(device=selected_source.device, dtype=torch.float32)
    transported_all = source_residual.float() @ J_gpu.T
    selected_transported = transported_all.index_select(0, offsets).to(dtype=torch.bfloat16)
    selected_source = selected_source.to(dtype=torch.bfloat16)

    with torch.no_grad():
        j_logits = model.lm_head(model.model.norm(selected_transported))
        base_logits = model.lm_head(model.model.norm(selected_source))

    details = []
    j_texts = []
    base_texts = []
    for idx, row in enumerate(rows):
        j_top = top_tokens(tokenizer, j_logits[idx], top_k)
        base_top = top_tokens(tokenizer, base_logits[idx], top_k)
        details.append(
            {
                **row,
                "j_lens": j_top,
                "logit_lens": base_top,
            }
        )
        j_texts.append([item["text"] for item in j_top])
        base_texts.append([item["text"] for item in base_top])
    return rows, details, j_texts, base_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--support-repo", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--meta-json", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    started_at = now_utc()
    start_time = time.time()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(int(args.device.split(":")[-1]))
        torch.cuda.reset_peak_memory_stats(args.device)

    log(f"started_at={started_at}")
    log(f"python={sys.executable}")
    log(f"torch={torch.__version__}")
    log(f"device={args.device}")
    model, tokenizer, processor, image_token_index, loading_info = load_model(
        args.model_path, args.support_repo, args.device
    )
    prepared = prepare_sample(
        model,
        tokenizer,
        processor,
        image_token_index,
        Path(args.model_path).resolve(),
        Path(args.corpus_path).resolve(),
        args.row_index,
        args.device,
    )
    log(
        "prepared "
        f"seq_len={prepared['expanded_seq_len']} source={prepared['source_span']} "
        f"target_crop={[int(x) for x in prepared['target_crop_positions'].detach().cpu().tolist()]}"
    )

    used_dim_batch = int(args.dim_batch)
    try:
        J, jacobian_elapsed_s, jacobian_peak = compute_jacobian(
            model, prepared, used_dim_batch, args.progress_json, args.device
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or used_dim_batch == 1:
            raise
        log(f"OOM at dim_batch={used_dim_batch}; retrying with dim_batch=1")
        write_json(
            args.progress_json,
            {
                "status": "RETRY_AFTER_OOM",
                "updated_at_utc": now_utc(),
                "failed_dim_batch": used_dim_batch,
                "retry_dim_batch": 1,
                "error": str(exc),
            },
        )
        gc.collect()
        torch.cuda.empty_cache()
        used_dim_batch = 1
        J, jacobian_elapsed_s, jacobian_peak = compute_jacobian(
            model, prepared, used_dim_batch, args.progress_json, args.device
        )

    # One-row forward to capture the source residual used for the readout.
    source_holder = {}

    def capture_source(module, inputs):
        source_holder["tensor"] = inputs[0].detach()

    hook = model.model.layers[14].register_forward_pre_hook(capture_source)
    try:
        with torch.no_grad():
            model.model(
                input_ids=None,
                inputs_embeds=prepared["inputs_embeds"].to(args.device, dtype=torch.bfloat16),
                attention_mask=prepared["attention_mask"].to(args.device) if prepared["attention_mask"] is not None else None,
                position_ids=prepared["position_ids"].to(args.device) if prepared["position_ids"] is not None else None,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        hook.remove()

    rows, details, j_texts, base_texts = readout(
        model,
        tokenizer,
        J,
        source_holder["tensor"],
        prepared["source_span"],
        args.top_k,
    )
    ended_at = now_utc()
    total_elapsed_s = time.time() - start_time
    peak = int(torch.cuda.max_memory_allocated(args.device)) if torch.cuda.is_available() else 0

    sample = prepared["sample"]
    readout_payload = {
        "schema": "huatuo_single_sample_readout_for_f1_v1",
        "id": sample.get("id"),
        "sample_id": sample.get("id"),
        "dataset": sample.get("dataset"),
        "image_path": prepared["image_path"],
        "prompt": sample.get("prompt"),
        "question": sample.get("prompt"),
        "answer": sample.get("answer"),
        "gold_text": sample.get("answer"),
        "top_k": args.top_k,
        "layers": [14],
        "j_lens_token_texts": {"14": j_texts},
        "logit_lens_token_texts": {"14": base_texts},
        "readout_rows": rows,
        "readout_details": {"14": details},
        "notes": [
            "token_texts are ordered by readout_rows and contain filtered top-k decoded tokens",
            "this is a single-sample smoke artifact for downstream token-level F1 plumbing, not a fitting conclusion",
        ],
    }
    meta_payload = {
        "schema": "huatuo_single_sample_readout_meta_v1",
        "status": "PASS",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_s": round(total_elapsed_s, 3),
        "jacobian_elapsed_s": round(jacobian_elapsed_s, 3),
        "python": sys.executable,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": args.device,
        "model_path": str(Path(args.model_path).resolve()),
        "support_repo": str(Path(args.support_repo).resolve()),
        "corpus_path": str(Path(args.corpus_path).resolve()),
        "row_index": args.row_index,
        "sample_id": sample.get("id"),
        "image_path": prepared["image_path"],
        "prompt": sample.get("prompt"),
        "question": sample.get("prompt"),
        "answer": sample.get("answer"),
        "gold_text": sample.get("answer"),
        "layer": 14,
        "hidden_dim": int(model.config.hidden_size),
        "J_shape": list(J.shape),
        "J_dtype": str(J.dtype),
        "dim_batch": used_dim_batch,
        "completed_full_3584_dims": bool(J.shape == (3584, 3584)),
        "source_span": prepared["source_span"],
        "source_token_count": prepared["image_token_count"],
        "target_span": prepared["target_span"],
        "target_crop_positions": [int(x) for x in prepared["target_crop_positions"].detach().cpu().tolist()],
        "readout_rows": rows,
        "top_k": args.top_k,
        "placeholder_count": prepared["placeholder_count"],
        "expanded_seq_len": prepared["expanded_seq_len"],
        "raw_prefix_len": prepared["raw_prefix_len"],
        "raw_full_len": prepared["raw_full_len"],
        "max_memory_allocated_bytes": max(peak, jacobian_peak),
        "max_memory_allocated_gib": round(max(peak, jacobian_peak) / (1024 ** 3), 3),
        "loading_info": {
            "missing_key_count": len(loading_info.get("missing_keys", [])),
            "unexpected_key_count": len(loading_info.get("unexpected_keys", [])),
        },
        "forbidden_items_touched": False,
    }
    write_json(args.output_json, readout_payload)
    write_json(args.meta_json, meta_payload)
    log(json.dumps(meta_payload, indent=2, ensure_ascii=False))

    del model, J
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
