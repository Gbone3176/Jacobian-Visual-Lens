#!/usr/bin/env python3
"""Real Huatuo single-image bridge for JVLens.

This module is import-safe: it does not import torch, transformers, jlens, or
Huatuo support code until run_huatuo_single_image() is called.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_TOKEN_COUNT = 576
TOKEN_GRID = (24, 24)
IGNORE_INDEX = -100


def _top_tokens(tokenizer: Any, logits: Any, top_k: int) -> list[dict[str, Any]]:
    k = min(max(top_k * 8, top_k), int(logits.shape[-1]))
    values, indices = __import__("torch").topk(logits.float(), k=k, dim=-1)
    rows: list[dict[str, Any]] = []
    for rank, (score, token_id) in enumerate(zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()), start=1):
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        text = text.replace("\ufffd", "").strip()
        if not text or "<|" in text:
            continue
        rows.append({"rank": len(rows) + 1, "token_id": int(token_id), "text": text, "score": float(score)})
        if len(rows) >= top_k:
            break
    return rows


def _load_runtime_helpers(runtime_root: Path):
    runtime_root = Path(runtime_root).resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    from huatuo_jlens_core import load_jacobian_lens
    from huatuo_single_sample_runtime import image_to_tensor, prepare_multimodal, tokenizer_image_token

    return load_jacobian_lens, image_to_tensor, prepare_multimodal, tokenizer_image_token


def _load_model_bundle(
    *,
    model_path: Path,
    support_repo: Path,
    processor_path: Path | None,
    device: str,
):
    import torch
    from transformers import AutoTokenizer

    support_repo = Path(support_repo).resolve()
    if str(support_repo) not in sys.path:
        sys.path.insert(0, str(support_repo))
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is not available")
        torch.cuda.set_device(int(device.split(":")[-1]))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
        str(model_path),
        init_vision_encoder_from_ckpt=False,
        output_loading_info=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.config._attn_implementation = "eager"
    model.tokenizer = tokenizer
    model.config.tokenizer_padding_side = "left"
    model.eval()
    model.to(device)
    model.requires_grad_(False)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16, device=device)

    if processor_path is not None:
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor.from_pretrained(str(processor_path), local_files_only=True)
    else:
        processor = vision_tower.image_processor

    return model, tokenizer, processor, IMAGE_TOKEN_INDEX, loading_info


def _prepare_prompt_sample(
    *,
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_to_tensor: Any,
    prepare_multimodal: Any,
    tokenizer_image_token: Any,
    image_token_index: int,
    image_path: Path,
    prompt: str,
    device: str,
) -> dict[str, Any]:
    import torch

    prefix = f"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"
    input_ids_1d = tokenizer_image_token(tokenizer, prefix, image_token_index, return_tensors="pt")
    placeholder_count = int((input_ids_1d == image_token_index).sum().item())
    if placeholder_count != 1:
        raise RuntimeError(f"expected one image placeholder, got {placeholder_count}")
    raw_image_pos = int(torch.where(input_ids_1d == image_token_index)[0][0].item())
    image_tensor, original_image_size, square_image_size = image_to_tensor(image_path, processor, device)
    input_ids = input_ids_1d.unsqueeze(0).to(device)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = prepare_multimodal(
        model,
        input_ids,
        labels,
        [image_tensor.to(dtype=torch.bfloat16)],
    )
    del expanded_labels
    image_token_count = int(inputs_embeds.shape[1] - (input_ids_1d.shape[0] - 1))
    source_span = [raw_image_pos, raw_image_pos + image_token_count]
    if image_token_count != IMAGE_TOKEN_COUNT:
        raise RuntimeError(f"expected {IMAGE_TOKEN_COUNT} image tokens, got {image_token_count}")
    return {
        "inputs_embeds": inputs_embeds.detach(),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "original_image_size": list(original_image_size),
        "square_image_size": list(square_image_size),
        "source_span": source_span,
        "image_token_count": image_token_count,
        "placeholder_count": placeholder_count,
        "raw_prompt_len": int(input_ids_1d.shape[0]),
        "expanded_seq_len": int(inputs_embeds.shape[1]),
    }


def _get_attentions(model_output: Any) -> Any:
    attentions = getattr(model_output, "attentions", None)
    if attentions is None and isinstance(model_output, dict):
        attentions = model_output.get("attentions")
    if attentions is None:
        raise RuntimeError("Huatuo forward did not return attentions; output_attentions=True path must be verified")
    return attentions


def _capture_single_forward(*, model: Any, prepared: dict[str, Any], source_layer: int, device: str) -> dict[str, Any]:
    import torch

    holder: dict[str, Any] = {}

    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        holder["source_residual"] = inputs[0].detach()

    handle = model.model.layers[int(source_layer)].register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            output = model.model(
                input_ids=None,
                inputs_embeds=prepared["inputs_embeds"].to(device=device, dtype=torch.bfloat16),
                attention_mask=prepared["attention_mask"].to(device) if prepared["attention_mask"] is not None else None,
                position_ids=prepared["position_ids"].to(device) if prepared["position_ids"] is not None else None,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if "source_residual" not in holder:
        raise RuntimeError(f"did not capture layer {source_layer} source residual")

    attentions = _get_attentions(output)
    if len(attentions) <= int(source_layer):
        raise RuntimeError(
            f"Huatuo forward returned {len(attentions)} attention layers; "
            f"need layer {source_layer}. Confirm eager attention is active."
        )
    source_start, source_end = prepared["source_span"]
    image_slice = attentions[int(source_layer)][0, :, -1, source_start:source_end]
    if int(image_slice.shape[-1]) != IMAGE_TOKEN_COUNT:
        raise RuntimeError(f"attention image-token slice has {int(image_slice.shape[-1])} tokens, expected {IMAGE_TOKEN_COUNT}")
    raw_attention = image_slice.float().mean(dim=0).reshape(*TOKEN_GRID).detach().cpu().numpy().astype(np.float32)
    return {"source_residual": holder["source_residual"], "raw_attention": raw_attention}


def _readout_all_patches(
    *,
    model: Any,
    tokenizer: Any,
    lens: Any,
    source_residual: Any,
    source_span: list[int],
    top_k_tokens: int,
    source_layer: int,
) -> list[dict[str, Any]]:
    import torch

    source_start, source_end = source_span
    residual = source_residual[0, source_start:source_end, :].detach()
    if int(residual.shape[0]) != IMAGE_TOKEN_COUNT:
        raise RuntimeError(f"residual source span has {int(residual.shape[0])} tokens, expected {IMAGE_TOKEN_COUNT}")

    jacobian = lens.jacobians[int(source_layer)].to(device=residual.device, dtype=torch.float32)
    transported = (residual.float() @ jacobian.T).to(dtype=torch.bfloat16)
    baseline = residual.to(dtype=torch.bfloat16)
    with torch.no_grad():
        j_logits = model.lm_head(model.model.norm(transported))
        base_logits = model.lm_head(model.model.norm(baseline))

    rows: list[dict[str, Any]] = []
    for patch_id in range(IMAGE_TOKEN_COUNT):
        patch_row, patch_col = divmod(patch_id, TOKEN_GRID[1])
        rows.append(
            {
                "patch_id": patch_id,
                "patch_row": patch_row,
                "patch_col": patch_col,
                "jlens_top": _top_tokens(tokenizer, j_logits[patch_id], top_k_tokens),
                "logit_lens_top": _top_tokens(tokenizer, base_logits[patch_id], top_k_tokens),
            }
        )
    return rows


def run_huatuo_single_image(
    *,
    image_path: Path,
    prompt: str,
    model_path: Path,
    support_repo: Path,
    runtime_root: Path,
    lens_path: Path,
    processor_path: Path | None,
    source_layer: int,
    top_k_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Run one real Huatuo forward and return VG attention plus JLens readouts."""

    import torch
    import transformers

    load_jacobian_lens, image_to_tensor, prepare_multimodal, tokenizer_image_token = _load_runtime_helpers(runtime_root)
    model, tokenizer, processor, image_token_index, loading_info = _load_model_bundle(
        model_path=Path(model_path),
        support_repo=Path(support_repo),
        processor_path=Path(processor_path) if processor_path else None,
        device=device,
    )
    lens = load_jacobian_lens(Path(lens_path))
    prepared = _prepare_prompt_sample(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        image_to_tensor=image_to_tensor,
        prepare_multimodal=prepare_multimodal,
        tokenizer_image_token=tokenizer_image_token,
        image_token_index=int(image_token_index),
        image_path=Path(image_path),
        prompt=prompt,
        device=device,
    )
    captured = _capture_single_forward(model=model, prepared=prepared, source_layer=source_layer, device=device)
    readouts = _readout_all_patches(
        model=model,
        tokenizer=tokenizer,
        lens=lens,
        source_residual=captured["source_residual"],
        source_span=prepared["source_span"],
        top_k_tokens=top_k_tokens,
        source_layer=source_layer,
    )
    return {
        "raw_attention": captured["raw_attention"],
        "patch_readouts": readouts,
        "prepared": prepared,
        "loading_info": {
            "missing_key_count": len(loading_info.get("missing_keys", [])) if isinstance(loading_info, dict) else None,
            "unexpected_key_count": len(loading_info.get("unexpected_keys", [])) if isinstance(loading_info, dict) else None,
        },
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": device,
        },
        "runtime_contract": {
            "single_forward": True,
            "attention_formula": "attentions[source_layer][0, :, -1, image_start:image_start+576].mean(dim=0).reshape(24,24)",
            "residual_capture": "register_forward_pre_hook on model.model.layers[source_layer] in the same forward",
            "jlens_orientation": "residual @ J.T",
        },
    }
