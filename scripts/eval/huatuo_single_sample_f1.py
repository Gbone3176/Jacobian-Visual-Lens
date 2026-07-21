#!/usr/bin/env python3
"""Author-style token word-bag F1 for Huatuo single-sample readouts.

This evaluator intentionally depends only on the Python standard library.  It
expects runtime code to provide already-decoded top-k token text rows for JLens
and logit-lens outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "huatuo_single_sample_readout_f1_v1"

SEC9_STOPWORDS = set(
    "this that with from have been were they their there which image shows also appears "
    "left right background foreground overall likely visible while near onto over under".split()
)

MEDICAL_EXTRA_STOPWORDS = set(
    "patient patients clinical clinically disease diseases medical medicine imaging scan scans "
    "figure panel panels view views axial coronal sagittal section sections slice slices obtained "
    "using shown demonstrates demonstrate reveals reveal seen showing based findings result results "
    "normal abnormal case study arrow arrows".split()
)

RULE_STOPWORDS = {
    "sec9_raw": SEC9_STOPWORDS,
    "medical_extended": SEC9_STOPWORDS | MEDICAL_EXTRA_STOPWORDS,
}


def words(text: str, stopwords: set[str]) -> set[str]:
    """Match the sec9 author-style word bag: lowercase regex words, length >= 4."""
    return {word for word in re.findall(r"[a-z]{4,}", text.lower()) if word not in stopwords}


def token_text_is_usable(token_text: str) -> bool:
    token = token_text.strip()
    return bool(token) and "<|" not in token and "\ufffd" not in token


def rows_for_layer(payload: Any, layer: int) -> list[list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("token text payload must be a dict keyed by layer")
    value = payload.get(str(layer), payload.get(layer))
    if value is None:
        raise ValueError(f"missing token text rows for layer {layer}")
    if not isinstance(value, list):
        raise ValueError(f"layer {layer} token text rows must be a list")
    rows: list[list[str]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise ValueError(f"layer {layer} row {row_index} must be a list of token text strings")
        rows.append([str(token) for token in row])
    return rows


def word_bag(token_text_rows: Iterable[Iterable[str]], stopwords: set[str], *, top_k: int) -> set[str]:
    bag: set[str] = set()
    for row in token_text_rows:
        kept = 0
        for token_text in row:
            if not token_text_is_usable(str(token_text)):
                continue
            bag |= words(str(token_text), stopwords)
            kept += 1
            if kept >= top_k:
                break
    return bag


def prf(predicted: set[str], gold: set[str]) -> dict[str, float | int]:
    hits = len(predicted & gold)
    recall = hits / len(gold) if gold else 0.0
    precision = hits / max(len(predicted), 1)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits": hits,
        "pred_words": len(predicted),
        "gold_words": len(gold),
    }


def parse_layers(record: dict[str, Any]) -> list[int]:
    raw_layers = record.get("layers")
    if raw_layers is None:
        raw_layers = sorted({int(layer) for layer in record["j_lens_token_texts"].keys()})
    layers = [int(layer) for layer in raw_layers]
    if not layers:
        raise ValueError("record must contain at least one layer")
    return layers


def evaluate_record(record: dict[str, Any], *, rules: list[str]) -> dict[str, Any]:
    unknown_rules = sorted(set(rules) - set(RULE_STOPWORDS))
    if unknown_rules:
        raise ValueError(f"unknown rules {unknown_rules}; choices={sorted(RULE_STOPWORDS)}")
    for field in ("answer", "j_lens_token_texts", "logit_lens_token_texts"):
        if field not in record:
            raise ValueError(f"missing required field: {field}")

    top_k = int(record.get("top_k", 8))
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    layers = parse_layers(record)
    j_texts = record["j_lens_token_texts"]
    logit_texts = record["logit_lens_token_texts"]

    per_rule: dict[str, Any] = {}
    for rule in rules:
        stopwords = RULE_STOPWORDS[rule]
        gold = words(str(record["answer"]), stopwords)
        per_layer: dict[str, Any] = {}
        all_j_rows: list[list[str]] = []
        all_logit_rows: list[list[str]] = []

        for layer in layers:
            layer_j_rows = rows_for_layer(j_texts, layer)
            layer_logit_rows = rows_for_layer(logit_texts, layer)
            all_j_rows.extend(layer_j_rows)
            all_logit_rows.extend(layer_logit_rows)
            j_words = word_bag(layer_j_rows, stopwords, top_k=top_k)
            logit_words = word_bag(layer_logit_rows, stopwords, top_k=top_k)
            per_layer[str(layer)] = {
                "layer": layer,
                "rule": rule,
                "top_k": top_k,
                "gold_words": sorted(gold),
                "gold_word_count": len(gold),
                "j_words": sorted(j_words),
                "logit_words": sorted(logit_words),
                "j_lens": prf(j_words, gold),
                "logit_lens": prf(logit_words, gold),
            }

        band_j_words = word_bag(all_j_rows, stopwords, top_k=top_k)
        band_logit_words = word_bag(all_logit_rows, stopwords, top_k=top_k)
        per_rule[rule] = {
            "rule": rule,
            "top_k": top_k,
            "stopword_count": len(stopwords),
            "gold_words": sorted(gold),
            "gold_word_count": len(gold),
            "per_layer": per_layer,
            "band": {
                "layers": layers,
                "rule": rule,
                "top_k": top_k,
                "gold_words": sorted(gold),
                "gold_word_count": len(gold),
                "j_words": sorted(band_j_words),
                "logit_words": sorted(band_logit_words),
                "j_lens": prf(band_j_words, gold),
                "logit_lens": prf(band_logit_words, gold),
            },
        }

    return {
        "status": "PASS",
        "schema": SCHEMA,
        "sample_id": record.get("sample_id", record.get("id")),
        "top_k": top_k,
        "layers": layers,
        "rules": rules,
        "results": per_rule,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"empty input: {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in stripped.splitlines() if line.strip()]
    parsed = json.loads(stripped)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("input must be a JSON object, JSON list, or JSONL records")


def mean_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        metric: sum(float(row[key][metric]) for row in rows) / len(rows)
        for metric in ("precision", "recall", "f1")
    }


def aggregate_outputs(outputs: list[dict[str, Any]], rules: list[str]) -> dict[str, Any]:
    if len(outputs) == 1:
        return outputs[0]
    aggregate: dict[str, Any] = {}
    for rule in rules:
        bands = [output["results"][rule]["band"] for output in outputs]
        aggregate[rule] = {
            "rule": rule,
            "record_count": len(outputs),
            "j_lens": mean_metric(bands, "j_lens"),
            "logit_lens": mean_metric(bands, "logit_lens"),
        }
    return {
        "status": "PASS",
        "schema": SCHEMA + "_batch",
        "record_count": len(outputs),
        "records": outputs,
        "aggregate_band_mean": aggregate,
    }


def parse_rules(spec: str) -> list[str]:
    rules = [rule.strip() for rule in spec.split(",") if rule.strip()]
    if not rules:
        raise ValueError("at least one rule is required")
    unknown = sorted(set(rules) - set(RULE_STOPWORDS))
    if unknown:
        raise ValueError(f"unknown rules {unknown}; choices={sorted(RULE_STOPWORDS)}")
    return rules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Readout JSON object/list or JSONL file.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path; stdout is always written.")
    parser.add_argument(
        "--rules",
        default="sec9_raw",
        help="Comma-separated F1 rules. sec9_raw is the default author-style rule; medical_extended is optional.",
    )
    args = parser.parse_args(argv)

    rules = parse_rules(args.rules)
    records = load_records(args.input)
    outputs = [evaluate_record(record, rules=rules) for record in records]
    result = aggregate_outputs(outputs, rules)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
