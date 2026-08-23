from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantUnit:
    unit_id: str
    tensor_name: str
    layer: int
    expert: int | None
    projection: str


def qwen3_moe_units(config: dict) -> list[QuantUnit]:
    layers = int(config["num_hidden_layers"])
    experts = int(config["num_experts"])
    units: list[QuantUnit] = []
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.experts"
        for expert in range(experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensor = f"{prefix}.{expert}.{projection}.weight"
                units.append(QuantUnit(f"L{layer:03d}.E{expert:03d}.{projection}", tensor, layer, expert, projection))
    return units


def gemma4_moe_units(config: dict) -> list[QuantUnit]:
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    units: list[QuantUnit] = []
    # Gemma4 stores experts stacked; slicing is recorded in unit_id rather than tensor name.
    for layer in range(layers):
        for expert in range(int(text["num_experts"])):
            base = f"model.language_model.layers.{layer}.experts"
            units.extend(
                [
                    QuantUnit(f"L{layer:03d}.E{expert:03d}.gate", f"{base}.gate_up_proj", layer, expert, "gate_slice"),
                    QuantUnit(f"L{layer:03d}.E{expert:03d}.up", f"{base}.gate_up_proj", layer, expert, "up_slice"),
                    QuantUnit(f"L{layer:03d}.E{expert:03d}.down", f"{base}.down_proj", layer, expert, "down_slice"),
                ]
            )
    return units


def load_inventory(config_path: str | Path, family: str) -> list[QuantUnit]:
    config = json.loads(Path(config_path).read_text())
    if family == "qwen3_moe":
        return qwen3_moe_units(config)
    if family == "gemma4":
        return gemma4_moe_units(config)
    raise ValueError(f"unsupported model family {family!r}")

