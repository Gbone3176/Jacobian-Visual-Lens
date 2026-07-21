# Huatuo JLens Fitting and Token-Level F1

This page documents the optional fitting and evaluation scripts included in this staging package. The scripts are for reproducibility review and local experiments; they do not include HuatuoGPT-Vision model weights, training data, or an automatic experiment launcher.

## Included Scripts

- `scripts/fit/huatuo_fit_jlens.py`: computes image-token Jacobian lenses and saves `huatuo_jimage_lens.pt`, metadata, checkpoint, and optional prefix checkpoints.
- `scripts/eval/huatuo_eval_fitted_lens.py`: evaluates a fitted lens on held-out rows and reports JLens and logit-lens token-level F1.
- `scripts/eval/huatuo_single_sample_f1.py`: standard-library word-bag F1 implementation. `sec9_raw` is the default author-style rule; `medical_extended` is available as an optional extension.
- `src/huatuo_jlens_core.py`: shared lens save/load, metadata, JSONL, layer parsing, and checksum helpers.
- `src/huatuo_single_sample_runtime.py` and `src/huatuo_single_sample_readout.py`: open-source runtime helpers used by the fit/eval scripts.

`huatuo_aggregate_pilot.py` is not included in this staging package because the current open-source fitting path does not need its pilot-schema dependency.

## Data Contract

The fitting corpus is a JSONL file. Each selected row must include at least:

- `image`: path to the image file.
- `prompt`: question or instruction text.
- `answer`: teacher-forced answer text.
- Optional `id`, `sample_id`, or `dataset` fields for metadata.

The fit path treats the image-token residual positions as source rows and teacher-forced answer-token residual positions at `target_layer` as targets. The saved lens keeps the orientation `residual @ J.T`.

## Layer16 n50 Fit Template

The example below uses placeholder paths. Replace them with local model, support repository, corpus, and run directories.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/fit/huatuo_fit_jlens.py \
  --model-path /path/to/HuatuoGPT-Vision-7B \
  --support-repo /path/to/HuatuoGPT-Vision \
  --corpus-path /path/to/medical_vqa_fit.jsonl \
  --out-dir runs/fit_layer16_n50 \
  --row-start 0 \
  --max-samples 50 \
  --source-layers 16 \
  --target-layer 27 \
  --dim-batch 4 \
  --target-crop-strategy head_tail \
  --max-target-tokens 4 \
  --prefix-save-at 1,2,4,8,16,32,50 \
  --top-k 8 \
  --device cuda:0
```

The script intentionally has no repository-external default output path. `--out-dir` is required.

## Held-Out F1 Evaluation Template

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/eval/huatuo_eval_fitted_lens.py \
  --model-path /path/to/HuatuoGPT-Vision-7B \
  --support-repo /path/to/HuatuoGPT-Vision \
  --corpus-path /path/to/medical_vqa_eval.jsonl \
  --lens-path runs/fit_layer16_n50/huatuo_jimage_lens.pt \
  --layers 16 \
  --row-start 50 \
  --max-samples 40 \
  --top-k 8 \
  --rules sec9_raw \
  --output-json runs/eval_layer16_n50/sec9_summary.json \
  --partial-jsonl runs/eval_layer16_n50/sec9_partial.jsonl \
  --device cuda:0
```

For comparability, keep the F1 contract fixed before comparing runs: tokenizer, decoded token text, top-k, target crop, row ranges, layers, and rule name.

## F1 Rule Summary

The `sec9_raw` rule builds a lowercased regex word bag using `[a-z]{4,}`, filters empty strings, model special tokens, replacement characters, and a small stopword list, then reports precision, recall, and F1 for both JLens and logit-lens outputs.

The optional `medical_extended` rule adds medical-display stopwords. It can be useful diagnostically, but it should not replace `sec9_raw` as the default author-style metric.
