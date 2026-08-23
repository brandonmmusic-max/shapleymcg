from quant_pipeline.models.inventory import gemma4_moe_units, qwen3_moe_units


def test_qwen_inventory_has_triplet_per_expert():
    units = qwen3_moe_units({"num_hidden_layers": 2, "num_experts": 3})
    assert len(units) == 18
    assert units[0].tensor_name == "model.layers.0.mlp.experts.0.gate_proj.weight"


def test_gemma_inventory_represents_stacked_slices():
    units = gemma4_moe_units({"text_config": {"num_hidden_layers": 2, "num_experts": 3}})
    assert len(units) == 18
    assert units[0].projection == "gate_slice"

