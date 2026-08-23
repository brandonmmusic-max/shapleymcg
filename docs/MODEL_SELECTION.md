# Pilot model decision

## Primary: Qwen3-30B-A3B-Base

`Qwen/Qwen3-30B-A3B-Base` is the primary experiment because one checkpoint can
exercise both levels of the proposed method:

1. model-level Aumann-Shapley path attribution across its 48 MoE layers; and
2. within-layer attribution across 128 routed experts, including gate/up/down
   joint candidates and cross-expert terms.

It also has three important controls that Gemma 4 does not provide together:

- the same architecture appears in the relevant published quantization work;
- public and personal quant baselines make KLD changes interpretable; and
- the Hugging Face checkpoint exposes expert gate/up/down weights separately,
  making actual-codec candidate construction and byte accounting direct.

Use the base model for next-token KLD. It avoids chat-template, thinking-mode,
sampling-temperature, and instruction-tuning policy effects. Use the instruct
checkpoint only after the quantization choice is frozen, for task behavior.

## Secondary: Gemma 4 26B-A4B

Gemma 4 26B-A4B is a useful portability test, not the initial adjudicator. It
can validate both attribution levels, but a negative result would be harder to
separate from a new model adapter or Gemma-specific routing behavior. The 31B
Gemma 4 checkpoint is dense and is therefore not the intended MoE pilot.

The current corrected EXL3/MCG numeric core requires matrix dimensions divisible
by 128. Gemma 4's 704-wide expert dimension fails that gate, so Gemma portability
requires a compatible codec or a separately validated padding/packing contract.
