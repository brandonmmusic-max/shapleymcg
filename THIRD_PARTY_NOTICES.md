# Third-party acknowledgements

This repository is an independent research implementation. It does not vendor
code from the projects below, but its design and intended experiments build on
their work.

- Robert J. Aumann and Lloyd S. Shapley for the Aumann-Shapley value.
- Joshua Hill and NVIDIA Model Optimizer PR #2183 for Aumann-Shapley
  quantization sensitivity and the associated coverage/additivity analysis.
- Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa, and the QTIP/QuIP#
  authors for trellis quantization and incoherence-processing foundations.
- turboderp and ExLlamaV3 contributors for EXL3/Trellis quantization and its
  encoder/runtime ecosystem.
- The Qwen Team for Qwen3 and `Qwen/Qwen3-30B-A3B-Base` (Apache-2.0).
- The MC-MoE, HIGGS, PQI, GuidedQuant, MoEQuant, EAC-MoE, and VSRAQ authors
  for the mixed-precision, end-loss, router-aware, and route-shift research
  identified precisely in `docs/REFERENCES.md`.
- Luke Alonso and Local Inference Lab contributors for B12X, the vLLM fork,
  and local runtime research. The official BTX writer ports atom assembly from
  B12X `btx_synth.py` at the pinned commit; that upstream code is Apache-2.0.
- NVIDIA Model Optimizer, vLLM, Hugging Face Transformers, safetensors, and
  huggingface_hub contributors.
- Google DeepMind for Gemma 4, used as the planned portability model.

Any future vendored adapter must retain the upstream license and notice in the
same commit that introduces the code.

These acknowledgements identify intellectual and software lineage; they do not
imply endorsement of this repository by any named person or project. See
`docs/REFERENCES.md` for the method-to-component mapping and primary links.
