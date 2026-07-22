#!/usr/bin/env python3
"""Shared import-safe helpers for HuatuoGPT-V JLens fitting and evaluation."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


DEFAULT_RUNS_ROOT = Path("runs")


def parse_layers(spec: str) -> list[int]:
    """Parse comma-separated layer ids and inclusive ranges.

    Examples:
        ``"8,10,12"`` -> ``[8, 10, 12]``
        ``"8:14:2,27"`` -> ``[8, 10, 12, 14, 27]``
    """
    if spec is None or not str(spec).strip():
        raise ValueError("layer spec must be non-empty")
    layers: list[int] = []
    seen: set[int] = set()
    for chunk in str(spec).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid layer range: {item!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step == 0:
                raise ValueError(f"range step cannot be zero: {item!r}")
            if (stop - start) * step < 0:
                raise ValueError(f"range step moves away from stop: {item!r}")
            end = stop + (1 if step > 0 else -1)
            values = range(start, end, step)
        else:
            values = [int(item)]
        for layer in values:
            if layer not in seen:
                seen.add(layer)
                layers.append(layer)
    if not layers:
        raise ValueError(f"no layers parsed from {spec!r}")
    return layers


def parse_prefix_save_at(spec: str | None) -> list[int]:
    """Parse prefix checkpoint sample counts.

    Empty values disable prefix checkpointing.
    """
    if spec is None or not str(spec).strip():
        return []
    values = parse_layers(spec)
    bad = [value for value in values if value <= 0]
    if bad:
        raise ValueError(f"prefix-save-at values must be positive: {bad}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl_rows(path: Path, row_start: int = 0, max_samples: int | None = None) -> list[dict[str, Any]]:
    if row_start < 0:
        raise ValueError("row_start must be non-negative")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < row_start:
                continue
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_torch_save(obj: Any, path: Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_jacobian_lens(
    jacobians: dict[int, "torch.Tensor"],
    n_prompts: int,
    d_model: int,
    path: Path,
) -> None:
    from jlens.lens import JacobianLens

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        lens = JacobianLens(jacobians, n_prompts=n_prompts, d_model=d_model)
        lens.save(str(tmp))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_jacobian_lens(path: Path):
    from jlens.lens import JacobianLens

    return JacobianLens.load(str(Path(path)))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_fit_metadata(
    *,
    source_layers: list[int],
    target_layer: int,
    n_prompts: int,
    d_model: int,
    model_path: str | None = None,
    corpus_path: str | None = None,
    target_crop_strategy: str | None = None,
    max_target_tokens: int | None = None,
    dim_batch: int | None = None,
    top_k: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema": "huatuo_jlens_fit_metadata_v1",
        "orientation": "residual @ J.T",
        "source_semantics": "decoder residual stream at image-token positions after multimodal embedding replacement",
        "target_semantics": "teacher-forced answer-token residual positions at target_layer",
        "source_layers": [int(layer) for layer in source_layers],
        "target_layer": int(target_layer),
        "n_prompts": int(n_prompts),
        "d_model": int(d_model),
    }
    optional = {
        "model_path": model_path,
        "corpus_path": corpus_path,
        "target_crop_strategy": target_crop_strategy,
        "max_target_tokens": max_target_tokens,
        "dim_batch": dim_batch,
        "top_k": top_k,
    }
    metadata.update({key: value for key, value in optional.items() if value is not None})
    if extra:
        metadata.update(dict(extra))
    return metadata
