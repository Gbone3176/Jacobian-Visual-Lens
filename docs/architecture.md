# Architecture

JVLens has four layers:

1. CLI layer: `run_jvlens.py` and `src/jvlens_cli.py`.
2. Runtime bridge: `src/huatuo_runtime_bridge.py`, import-safe until a real run is authorized.
3. Static output contract: JSON, JSONL, PNG, and HTML under one run directory.
4. Review UI: source image overlay, q_type-aligned attention map, top patch table, JLens tokens, and logit-lens tokens.

For a real `run-single`, the intended execution path is one Huatuo forward pass:

- capture layer16 raw attention over image tokens;
- capture layer16 image-token residuals;
- read out the top attention patches through the fitted JacobianLens;
- compute logit-lens baseline tokens for comparison;
- write a static report and validate it.

The CPU fixture path generates synthetic attention and synthetic token rows without model loading.
