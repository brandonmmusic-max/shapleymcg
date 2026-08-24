import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "allocate_turboderp_v001_expert_exact_3p5",
    ROOT / "scripts" / "allocate_turboderp_v001_expert_exact_3p5.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v001_qwen_exact_3p5_reproduces_carried_surplus_pattern():
    document = MODULE.build(layers=48, experts=128, matrix_numel=2048 * 768, bpw=3.5)
    assert document["k3_count"] == document["k4_count"] == 9216
    assert document["average_weight_bits"] == 3.5
    assert document["layer_allocations"][0]["projection_bits"] == {
        "gate_proj": 3,
        "up_proj": 3,
        "down_proj": 4,
    }
    assert document["layer_allocations"][1]["projection_bits"] == {
        "gate_proj": 3,
        "up_proj": 4,
        "down_proj": 4,
    }
    assert len(document["choices"]) == 48 * 128 * 3
    assert len(document["allocation_sha256"]) == 64
