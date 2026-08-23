# Third-party acknowledgements

This repository is an independent research implementation. It does not vendor
code from the projects below, but its design and intended experiments build on
their work.

- The Qwen Team for Qwen3 and `Qwen/Qwen3-30B-A3B-Base` (Apache-2.0).
- Joshua Hill and NVIDIA Model Optimizer PR #2183 for Aumann-Shapley
  quantization sensitivity and the associated coverage/additivity analysis.
- turboderp and ExLlamaV3 contributors for EXL3/Trellis quantization and its
  encoder/runtime ecosystem.
- Local Inference Lab for the vLLM fork, B12X work, and local runtime research.
- NVIDIA Model Optimizer, vLLM, Hugging Face Transformers, safetensors, and
  huggingface_hub contributors.
- Google DeepMind for Gemma 4, used as the planned portability model.

Any future vendored adapter must retain the upstream license and notice in the
same commit that introduces the code.

