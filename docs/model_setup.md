# Model Setup

This staging package does not ship model weights.

Before real execution, prepare:

1. HuatuoGPT-Vision-7B model files.
2. HuatuoGPT-Vision support repository.
3. Runtime adapter helpers expected by `src/huatuo_runtime_bridge.py`.
4. Layer16 n50 JLens weights and metadata.

Then run `run-single` with explicit paths. Keep `--allow-model-run` absent for CPU-only checks and present only when model/GPU execution is authorized.

The runtime bridge is intentionally import-safe: heavy model libraries are imported only inside the real execution path.
