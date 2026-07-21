#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer


IGNORE_INDEX = -100


def log(message):
    print(message, flush=True)


def read_jsonl_row(path, row_index):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row_index:
                return json.loads(line)
    raise IndexError(f"row_index {row_index} not found in {path}")


def tokenizer_image_token(tokenizer, prompt, image_token_index, return_tensors=None):
    prompt_chunks = [
        tokenizer(chunk, add_special_tokens=False).input_ids
        for chunk in prompt.split("<image>")
    ]

    def insert_separator(chunks, sep):
        return [ele for sublist in zip(chunks, [sep] * len(chunks)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(chunk[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    if return_tensors is not None:
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def image_to_tensor(image_path, processor, device):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
    image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image_tensor.to(device), original_size, image.size


def tensor_shape(tensor):
    return list(tensor.shape) if tensor is not None else None


def tensor_stats(tensor):
    if tensor is None:
        return None
    t = tensor.detach()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "nonzero": int((t != 0).sum().item()),
        "norm": float(t.float().norm().item()),
        "max_abs": float(t.float().abs().max().item()) if t.numel() else 0.0,
    }


def prepare_multimodal(model, input_ids, labels, images):
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return model.prepare_inputs_labels_for_multimodal_new(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels,
        images=images,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--support-repo", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    support_repo = Path(args.support_repo).resolve()
    model_path = Path(args.model_path).resolve()
    output_json = Path(args.output_json).resolve()

    sys.path.insert(0, str(support_repo))
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(int(args.device.split(":")[-1]))

    env_report = {
        "python": sys.executable,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    log(f"env={env_report}")
    log(f"model_path={model_path}")
    log(f"support_repo={support_repo}")

    sample = read_jsonl_row(args.corpus_path, args.row_index)
    image_path = Path(sample["image"]).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"sample image not found: {image_path}")

    prompt = sample["prompt"]
    answer = sample["answer"]
    prefix = f"<|user|>\n<image>\n{prompt}\n<|assistant|>\n"
    answer_text = f"{answer} \n"
    full_prompt = prefix + answer_text

    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    log("loading Huatuo LlavaQwen2ForCausalLM")
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
    model.to(args.device)
    model.requires_grad_(False)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=torch.bfloat16, device=args.device)
    processor = vision_tower.image_processor

    prefix_ids_1d = tokenizer_image_token(tokenizer, prefix, IMAGE_TOKEN_INDEX, return_tensors="pt")
    full_ids_1d = tokenizer_image_token(tokenizer, full_prompt, IMAGE_TOKEN_INDEX, return_tensors="pt")
    answer_ids_1d = tokenizer(answer_text, add_special_tokens=False, truncation=True, return_tensors="pt").input_ids[0]
    placeholder_count = int((full_ids_1d == IMAGE_TOKEN_INDEX).sum().item())
    if placeholder_count != 1:
        raise RuntimeError(f"expected exactly one IMAGE_TOKEN_INDEX, got {placeholder_count}")
    raw_image_token_index = int(torch.where(full_ids_1d == IMAGE_TOKEN_INDEX)[0][0].item())

    image_tensor, original_image_size, square_image_size = image_to_tensor(image_path, processor, args.device)
    images = [image_tensor.to(dtype=torch.bfloat16)]

    full_ids = full_ids_1d.unsqueeze(0).to(args.device)
    raw_labels_1d = torch.full_like(full_ids_1d, IGNORE_INDEX)
    answer_raw_start = int(prefix_ids_1d.shape[0])
    raw_labels_1d[answer_raw_start:] = full_ids_1d[answer_raw_start:]
    raw_labels = raw_labels_1d.unsqueeze(0).to(args.device)

    log("preparing multimodal embeddings")
    _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = prepare_multimodal(
        model, full_ids, raw_labels, images
    )
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)
    expanded_seq_len = int(inputs_embeds.shape[1])
    image_token_count = int(expanded_seq_len - (full_ids_1d.shape[0] - 1))
    source_start = raw_image_token_index
    source_end = source_start + image_token_count
    target_positions = torch.where(expanded_labels[0] != IGNORE_INDEX)[0]
    if target_positions.numel() == 0:
        raise RuntimeError("expanded answer target span is empty")
    target_crop = target_positions[: min(4, int(target_positions.numel()))]

    layer_shapes = {}
    layer14_input = {}
    layer14_output = {}
    hooks = []

    def make_pre_hook(layer_idx):
        def pre_hook(module, inputs):
            hidden_states = inputs[0]
            layer_shapes[f"{layer_idx}_input"] = tensor_shape(hidden_states)
            if layer_idx == 14:
                hidden_states.retain_grad()
                layer14_input["tensor"] = hidden_states
        return pre_hook

    def make_fwd_hook(layer_idx):
        def fwd_hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            layer_shapes[f"{layer_idx}_output"] = tensor_shape(hidden_states)
            if layer_idx == 14:
                hidden_states.retain_grad()
                layer14_output["tensor"] = hidden_states
        return fwd_hook

    for layer_idx in [0, 14, 27]:
        layer = model.model.layers[layer_idx]
        hooks.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_fwd_hook(layer_idx)))

    log("running decoder/LM forward with prepared multimodal embeddings")
    with torch.enable_grad():
        outputs = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits_shape = tensor_shape(outputs.logits)
        target_residual = layer14_output["tensor"][0, target_crop, :4].float()
        scalar = target_residual.sum()
        log(f"backward scalar target shape={list(target_residual.shape)}")
        scalar.backward()

    for hook in hooks:
        hook.remove()

    source_grad = layer14_input["tensor"].grad[0, source_start:source_end, :]
    source_grad_head4 = source_grad[:, :4]
    result = {
        "status": "PASS",
        "started_at_utc": started_at,
        "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": os.getcwd(),
        "env": env_report,
        "model_path": str(model_path),
        "support_repo": str(support_repo),
        "corpus_path": str(Path(args.corpus_path).resolve()),
        "row_index": args.row_index,
        "sample": {
            "id": sample.get("id"),
            "dataset": sample.get("dataset"),
            "image": str(image_path),
            "original_image_size": list(original_image_size),
            "square_image_size": list(square_image_size),
            "prompt": prompt,
            "answer_char_count": len(answer),
            "answer_word_count": sample.get("answer_word_count"),
        },
        "model_structure": {
            "n_layers": int(len(model.model.layers)),
            "d_model": int(model.config.hidden_size),
            "vocab_size": int(model.config.vocab_size),
            "norm_path": "model.norm",
            "lm_head_path": "lm_head",
            "layers_path": "model.layers",
            "vision_tower_path": "model.vision_tower",
            "mm_projector_path": "model.mm_projector",
        },
        "loading_info": {
            "missing_key_count": len(loading_info.get("missing_keys", [])),
            "unexpected_key_count": len(loading_info.get("unexpected_keys", [])),
            "unexpected_key_prefixes": sorted({k.split(".")[0] for k in loading_info.get("unexpected_keys", [])}),
        },
        "tokenization": {
            "image_token_index": IMAGE_TOKEN_INDEX,
            "placeholder_count": placeholder_count,
            "raw_prefix_len": int(prefix_ids_1d.shape[0]),
            "raw_full_len": int(full_ids_1d.shape[0]),
            "raw_answer_token_count": int(answer_ids_1d.shape[0]),
            "raw_image_placeholder_position": raw_image_token_index,
        },
        "multimodal_expansion": {
            "expanded_seq_len": expanded_seq_len,
            "image_source_span": [source_start, source_end],
            "image_source_token_count": image_token_count,
            "expected_clip_patch_tokens": int(getattr(vision_tower, "num_patches", -1)),
            "target_span": [int(target_positions[0].item()), int(target_positions[-1].item()) + 1],
            "target_token_count": int(target_positions.numel()),
            "target_crop_positions": [int(x) for x in target_crop.detach().cpu().tolist()],
        },
        "hook_layer_shapes": layer_shapes,
        "forward": {
            "logits_shape": logits_shape,
            "inputs_embeds_shape": tensor_shape(inputs_embeds),
            "attention_mask_shape": tensor_shape(attention_mask),
            "position_ids_shape": tensor_shape(position_ids),
        },
        "jacobian_smoke": {
            "layer": 14,
            "cotangent_target_positions": [int(x) for x in target_crop.detach().cpu().tolist()],
            "cotangent_residual_dims": [0, 1, 2, 3],
            "source_grad": tensor_stats(source_grad),
            "source_grad_head4": tensor_stats(source_grad_head4),
        },
        "forbidden_items_touched": False,
        "notes": [
            "source span is decoder residual stream after multimodal embedding replacement",
            "Jacobian smoke is a single scalar backward, not a full 3584-dimension Jacobian run",
            "token-level F1 is not computed in this runtime smoke; target positions/readout paths are recorded for later F1 integration",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log(json.dumps(result, indent=2, ensure_ascii=False))
    del outputs, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
