import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from quant_pipeline.checkpoint.reference_pack import encode_reference_checkpoint
from quant_pipeline.codecs.uniform import quantize


def _write_qwen_fixture(root):
    config = {"num_hidden_layers": 1, "num_experts": 1}
    (root / "config.json").write_text(json.dumps(config))
    tensors = {
        f"model.layers.0.mlp.experts.0.{projection}.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    save_file(tensors, root / "model.safetensors")
    weight_map = {name: "model.safetensors" for name in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_reference_pack_rejects_false_allocation_bytes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    _write_qwen_fixture(model)
    allocation = tmp_path / "allocation.json"
    allocation.write_text(json.dumps({"choices": {"L000.E000.gate_proj": {"bits": 3, "stored_bytes": 3}}}))
    with pytest.raises(ValueError, match="byte claim"):
        encode_reference_checkpoint(model, "qwen3_moe", allocation, tmp_path / "packed", 8)


def test_gemma_reference_pack_slices_stacked_expert(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = {"text_config": {"num_hidden_layers": 1, "num_experts": 2}}
    (model / "config.json").write_text(json.dumps(config))
    gate_up = torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4)
    down = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    tensors = {
        "model.language_model.layers.0.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.experts.down_proj": down,
    }
    save_file(tensors, model / "model.safetensors")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}))
    expected = quantize(gate_up[1, :6].numpy(), 3, 8).stored_bytes
    allocation = tmp_path / "allocation.json"
    allocation.write_text(json.dumps({"stored_bytes": expected, "choices": {"L000.E001.gate": {"bits": 3, "stored_bytes": expected}}}))
    manifest = encode_reference_checkpoint(model, "gemma4", allocation, tmp_path / "packed", 8)
    assert manifest["units"]["L000.E001.gate"]["shape"] == [6, 4]
    assert manifest["allocation_sha256"]
    assert manifest["source"]["shards"]["model.safetensors"]["sha256"]

