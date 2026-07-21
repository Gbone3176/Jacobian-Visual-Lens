# JVLens Open-Source Staging Candidate

JVLens is a single-image visualization utility that joins two views in one static report:

- VG-style raw attention over a 24x24 image-token grid.
- JLens patch readout tokens for the top attention patches, with a logit-lens baseline.

This staging package is prepared for review. It contains source code, documentation, schemas, a synthetic fixture demo, and the layer16 n50 JLens fitted lens matrix. It does not include HuatuoGPT-Vision model weights, local cache paths, or real dataset images.

## Showcase Examples

These review images are sanitized static overviews copied from validated layer16 VG/JLens alignment outputs. They use attribute prompts and normalized_attention attribute maps, and embed only a small screenshot-style PNG per case, not standalone source images or datasets.

### COCO

![COCO natural image JVLens showcase](examples/showcase/coco_umbrella_layer16/overview.png)

COCO natural image example using an attribute prompt, layer16, prefix_n50 JLens, a VG-style 24x24 normalized attention map, and independent patch blocks with no interpolation.

### Dermoscopy

![Dermoscopy JVLens showcase](examples/showcase/dermoscopy_skin_lesion_layer16/overview.png)

Dermoscopy skin-lesion example using an attribute prompt, layer16, prefix_n50 JLens, a VG-style 24x24 normalized attention map, and independent patch blocks with no interpolation.

### Colorectal-Endoscopy

![Colorectal-Endoscopy JVLens showcase](examples/showcase/colorectal_endoscopy_polyp_layer16/overview.png)

Colorectal/endoscopy polyp example using an attribute prompt, layer16, prefix_n50 JLens, a VG-style 24x24 normalized attention map, and independent patch blocks with no interpolation.

## Status

This staging directory is prepared under the MIT License. Upstream HuatuoGPT-Vision model weights and support code remain external assets governed by their own licenses and model terms.

## Install

Use Python 3.11 or newer. Create an environment using your preferred tool, then install the minimal dependencies:

```bash
pip install -r requirements.txt
```

For real model execution, install the HuatuoGPT-Vision support code and any JLens runtime dependencies required by your local Huatuo adapter.

## CPU-Only Commands

Show CLI help:

```bash
PYTHONDONTWRITEBYTECODE=1 python run_jvlens.py --help
```

Validate the included synthetic fixture:

```bash
PYTHONDONTWRITEBYTECODE=1 python run_jvlens.py validate-output --out-dir examples/fixture_demo
```

Generate a new synthetic fixture without loading a model:

```bash
PYTHONDONTWRITEBYTECODE=1 python run_jvlens.py make-fixture-demo --out-dir experiment/fixture_demo_local --overwrite
```

## Real Single-Image Run Template

Real execution is guarded. Without `--allow-model-run`, `run-single` fails fast and does not load the model.

```bash
PYTHONDONTWRITEBYTECODE=1 python run_jvlens.py run-single \
  --image /path/to/image.png \
  --prompt "What visual evidence supports the answer?" \
  --q-type custom \
  --model-path /path/to/HuatuoGPT-Vision-7B \
  --support-repo /path/to/HuatuoGPT-Vision \
  --runtime-root /path/to/runtime-adapters \
  --lens-path weights/huatuo_jimage_lens_layer16_n50.pt \
  --out-dir experiment/my_single_image_run \
  --device cuda:0 \
  --allow-model-run
```

## Required External Assets

Users must provide:

- HuatuoGPT-Vision-7B model files from the upstream project.
- The Huatuo support repository used by the local adapter.
- The included fitted JLens weight file for layer16 n50, or a replacement with the same contract. See `weights/README.md`.

## Safety Gate

`run-single` requires `--allow-model-run` before importing the real runtime bridge and loading model/GPU dependencies. This is intentional: review commands and fixture validation remain CPU-only.

## Optional Lens Fitting and F1 Evaluation

This package also includes optional Huatuo JLens fitting and token-level F1 evaluation scripts:

- `scripts/fit/huatuo_fit_jlens.py`
- `scripts/eval/huatuo_eval_fitted_lens.py`
- `scripts/eval/huatuo_single_sample_f1.py`

These scripts are provided for reproducibility review. They do not include model weights, training data, or an automatic experiment launcher. Fit output requires an explicit `--out-dir`; use a local path such as `runs/fit_layer16_n50`. The default F1 rule is `sec9_raw`, matching the author-style word-bag metric; `medical_extended` remains available as an optional extension.

See `docs/fitting.md` and `configs/fit_huatuogpt_v7b_layer16_n50.yaml` for command templates and field contracts.

## Visualization Contract

The attention map is q_type-aligned and rendered as image-aspect patch blocks:

- logical grid: 24x24
- display mode: `patch_grid_image_aspect`
- colormap: viridis
- per-map display minmax only
- token interpolation: false
- independent rectangular patch blocks: true

See `docs/visualization_contract.md` for details.

## Limitations

This staging package does not claim official HuatuoGPT-Vision support. It is a research visualization utility and should be validated in the target environment before use on new models, prompts, or lens weights.

## Citation and References

- HuatuoGPT-Vision official: https://github.com/FreedomIntelligence/HuatuoGPT-Vision
- JLens: https://huggingface.co/kieraisverybored/jlens-qwen3.5-27b
- VG / Medical-MLLMs-Fail: https://github.com/Guimeng-Leo-Liu/Medical-MLLMs-Fail
