from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from quant_pipeline.normalization import (
    FitSamplePlan,
    MatrixInput,
    StreamingLayerFitter,
    StreamingTopologyLedger,
    decode_with_stored,
    fit_layer_absolute_normalization,
    tensor_sha256,
)


AUDIT = Path(
    "/home/brandonmusic/KLC_SANDBOXES/quant-research/"
    "prior-glm35-hf-audit"
)
pytestmark = pytest.mark.skipif(
    not AUDIT.is_dir(),
    reason="requires the separately sealed prior-GLM35 local audit oracle",
)
AUTHORITATIVE = AUDIT / (
    "reproducibility/local-corrected-v1/code/absolute_normalization_v1.py"
)
PINNED_CORE = AUDIT / "lineage/encode_tr3_v31.py"
CODEBOOK_SCALE = 1.24371088
BLOCK = 4


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cores():
    incumbent = sys.modules.get("exllamav3_ext")
    if incumbent is None:
        sys.modules["exllamav3_ext"] = types.ModuleType("exllamav3_ext")
    try:
        authoritative = _load(AUTHORITATIVE, "absolute_v31_frozen_oracle")
        core = _load(PINNED_CORE, "absolute_v31_pinned_numeric_core")
        core_sha = hashlib.sha256(PINNED_CORE.read_bytes()).hexdigest()
        yield authoritative, core, core_sha
    finally:
        sys.modules.pop("absolute_v31_frozen_oracle", None)
        sys.modules.pop("absolute_v31_pinned_numeric_core", None)
        if incumbent is None:
            sys.modules.pop("exllamav3_ext", None)


def _signs(length: int, offset: int) -> torch.Tensor:
    return torch.tensor(
        [1.0 if (index + offset) % 3 else -1.0 for index in range(length)],
        dtype=torch.float32,
    )


def _inputs(cls):
    """Three experts with Qwen-like rectangular GU/down geometry."""

    generator = torch.Generator(device="cpu").manual_seed(20260823)
    shared_gate_suh = _signs(8, 0)
    shared_down_svh = _signs(8, 1)
    items = []
    bits = ((3, 5, 4), (4, 3, 5), (5, 4, 3))
    masses = (0.25, 1.5, 7.25)
    for expert in range(3):
        gate = torch.randn((8, 12), generator=generator) * (0.009 + expert * 0.002)
        up = torch.randn((8, 12), generator=generator) * (0.015 + expert * 0.001)
        down = torch.randn((12, 8), generator=generator) * (0.012 + expert * 0.003)
        items.extend(
            (
                cls(
                    key=f"experts/{expert}/gate",
                    projection="gate_proj",
                    bits=bits[expert][0],
                    weight_kn=gate,
                    suh_sign=shared_gate_suh,
                    svh_sign=_signs(12, 2 + expert),
                    k_block_scales=(0.8, 1.25),
                    n_block_scales=(1.1, 0.9, 1.25),
                    mass=masses[expert],
                ),
                cls(
                    key=f"experts/{expert}/up",
                    projection="up_proj",
                    bits=bits[expert][1],
                    weight_kn=up,
                    suh_sign=shared_gate_suh,
                    svh_sign=_signs(12, 5 + expert),
                    k_block_scales=(0.8, 1.25),
                    n_block_scales=(1.25, 0.8, 1.0),
                    mass=masses[expert] * 0.75,
                ),
                cls(
                    key=f"experts/{expert}/down",
                    projection="down_proj",
                    bits=bits[expert][2],
                    weight_kn=down,
                    suh_sign=_signs(12, 8 + expert),
                    svh_sign=shared_down_svh,
                    k_block_scales=(0.9, 1.1, 1.25),
                    n_block_scales=(1.25, 0.8),
                    mass=masses[expert] * 1.25,
                ),
            )
        )
    return tuple(items)


def _gss(inputs) -> dict[str, float]:
    scales = (0.75, 1.5, 1.25, 1.1, 0.8, 1.35, 0.9, 1.6, 1.05)
    return {item.key: scales[index] for index, item in enumerate(inputs)}


def test_batch_and_streaming_are_byte_exact_to_frozen_source(cores):
    authoritative, core, core_sha = cores
    reference_inputs = _inputs(authoritative.MatrixInput)
    production_inputs = _inputs(MatrixInput)
    gss = _gss(production_inputs)

    reference_fit = authoritative.fit_layer_absolute_normalization(
        core,
        reference_inputs,
        codebook_scale=CODEBOOK_SCALE,
        block=BLOCK,
    )
    reference = reference_fit.finalize(gss)
    batch_fit = fit_layer_absolute_normalization(
        core,
        production_inputs,
        codebook_scale=CODEBOOK_SCALE,
        block=BLOCK,
    )
    batch = batch_fit.finalize(gss)

    assert torch.equal(batch_fit.shared_gate_up_suh, reference_fit.shared_gate_up_suh)
    assert torch.equal(batch_fit.shared_down_svh, reference_fit.shared_down_svh)

    plan = FitSamplePlan.from_inputs(production_inputs, block=BLOCK)
    fitter = StreamingLayerFitter(
        core,
        plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    for item in production_inputs:
        fitter.add_fit_matrix(item)
    streamed = fitter.finish()
    assert torch.equal(streamed.shared_gate_up_suh, reference_fit.shared_gate_up_suh)
    assert torch.equal(streamed.shared_down_svh, reference_fit.shared_down_svh)
    assert streamed.fit_plan_sha256 == plan.content_sha256
    assert streamed.numeric_core_sha256 == core_sha

    reference_targets = reference_fit.gss_targets()
    production_targets = batch_fit.gss_targets()
    ledger = StreamingTopologyLedger(streamed, (item.key for item in production_inputs))
    for item in production_inputs:
        prepared = streamed.prepare_matrix(item)
        actual = prepared.finalize(prepared.bind_gss(gss[item.key]))
        expected = reference.matrices[item.key]
        batch_value = batch.matrices[item.key]
        assert torch.equal(batch_fit.pre_gss_suh[item.key], reference_fit.pre_gss_suh[item.key])
        assert torch.equal(batch_fit.pre_gss_svh[item.key], reference_fit.pre_gss_svh[item.key])
        assert torch.equal(production_targets[item.key], reference_targets[item.key])
        assert torch.equal(prepared.pre_gss_suh, reference_fit.pre_gss_suh[item.key])
        assert torch.equal(prepared.pre_gss_svh, reference_fit.pre_gss_svh[item.key])
        assert torch.equal(prepared.gss_target(), reference_targets[item.key])
        assert torch.equal(actual.stored_suh, expected.stored_suh)
        assert torch.equal(actual.stored_svh, expected.stored_svh)
        assert torch.equal(actual.regularized, expected.regularized)
        assert torch.equal(batch_value.regularized, expected.regularized)
        assert actual.source_weight_identity_sha256 == batch_value.source_weight_identity_sha256
        decoded = decode_with_stored(
            core,
            actual.regularized,
            actual.stored_suh,
            actual.stored_svh,
            block=BLOCK,
        )
        assert torch.equal(decoded, authoritative.decode_with_stored(
            core,
            expected.regularized,
            expected.stored_suh,
            expected.stored_svh,
            block=BLOCK,
        ))
        ledger.add(actual)

    manifest = ledger.finish()
    assert manifest.numeric_core_sha256 == core_sha
    assert manifest.fit_plan_sha256 == plan.content_sha256
    assert {record.bits for record in manifest.matrices.values()} == {3, 4, 5}


def test_python_f64_division_boundary_is_preserved(cores):
    _, core, _ = cores
    inputs = _inputs(MatrixInput)
    fit = fit_layer_absolute_normalization(
        core, inputs, codebook_scale=CODEBOOK_SCALE, block=BLOCK
    )
    gate = fit.matrices["experts/0/gate"]
    expected = torch.tensor(
        [
            float(sign) / float(scale)
            for sign, scale in zip(
                _signs(8, 0).tolist(),
                [0.8] * 4 + [1.25] * 4,
                strict=True,
            )
        ],
        dtype=torch.float32,
    )
    assert torch.equal(gate.base_suh, expected)


def test_strict_subset_streaming_fit_prepares_non_sample_matrices_byte_exactly(cores):
    _, core, core_sha = cores
    inputs = _inputs(MatrixInput)
    fit_sample = inputs[:3]
    plan = FitSamplePlan.from_inputs(fit_sample, block=BLOCK)
    fitter = StreamingLayerFitter(
        core,
        plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    for item in fit_sample:
        fitter.add_fit_matrix(item)
    fit = fitter.finish()
    assert set(fit.fit_sample_keys) == {item.key for item in fit_sample}

    for index, sampled in enumerate(fit_sample):
        expected = fit.prepare_matrix(sampled)
        heldout = replace(
            sampled,
            key=f"heldout/{sampled.projection}",
            bits=(3, 4, 5)[(index + 1) % 3],
            mass=999.0 + index,
        )
        assert heldout.key not in fit.fit_sample_keys
        actual = fit.prepare_matrix(heldout)
        assert torch.equal(actual.pre_gss_suh, expected.pre_gss_suh)
        assert torch.equal(actual.pre_gss_svh, expected.pre_gss_svh)
        assert torch.equal(actual.gss_target(), expected.gss_target())
        actual_final = actual.finalize(actual.bind_gss(1.25))
        expected_final = expected.finalize(expected.bind_gss(1.25))
        assert torch.equal(actual_final.stored_suh, expected_final.stored_suh)
        assert torch.equal(actual_final.stored_svh, expected_final.stored_svh)
        assert torch.equal(actual_final.regularized, expected_final.regularized)


def test_boolean_gss_scale_is_rejected_by_batch_and_streaming_paths(cores):
    _, core, core_sha = cores
    inputs = _inputs(MatrixInput)
    batch = fit_layer_absolute_normalization(
        core, inputs, codebook_scale=CODEBOOK_SCALE, block=BLOCK
    )
    boolean_scales = {item.key: 1.0 for item in inputs}
    boolean_scales[inputs[0].key] = True
    with pytest.raises(ValueError, match="not boolean"):
        batch.finalize(boolean_scales)

    plan = FitSamplePlan.from_inputs(inputs[:3], block=BLOCK)
    fitter = StreamingLayerFitter(
        core,
        plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    for item in inputs[:3]:
        fitter.add_fit_matrix(item)
    prepared = fitter.finish().prepare_matrix(inputs[3])
    with pytest.raises(ValueError, match="not boolean"):
        prepared.bind_gss(True)


def test_shared_vectors_are_invariant_across_selected_bits(cores):
    _, core, core_sha = cores
    inputs = _inputs(MatrixInput)
    plan = FitSamplePlan.from_inputs(inputs, block=BLOCK)
    fitter = StreamingLayerFitter(
        core,
        plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    for item in inputs:
        fitter.add_fit_matrix(item)
    fit = fitter.finish()
    first = {
        item.key: fit.prepare_matrix(item).finalize(
            fit.prepare_matrix(item).bind_gss(1.0)
        )
        for item in inputs
    }
    second = {
        item.key: fit.prepare_matrix(item).finalize(
            fit.prepare_matrix(item).bind_gss(1.7)
        )
        for item in inputs
    }
    for item in inputs:
        if item.projection in ("gate_proj", "up_proj"):
            assert torch.equal(first[item.key].stored_suh, second[item.key].stored_suh)
            assert not torch.equal(first[item.key].stored_svh, second[item.key].stored_svh)
        else:
            assert torch.equal(first[item.key].stored_svh, second[item.key].stored_svh)
            assert not torch.equal(first[item.key].stored_suh, second[item.key].stored_suh)

    changed_bits = tuple(
        replace(item, bits={3: 5, 4: 3, 5: 4}[item.bits]) for item in inputs
    )
    changed_plan = FitSamplePlan.from_inputs(changed_bits, block=BLOCK)
    changed_fitter = StreamingLayerFitter(
        core,
        changed_plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    for item in changed_bits:
        changed_fitter.add_fit_matrix(item)
    changed_fit = changed_fitter.finish()
    assert torch.equal(fit.shared_gate_up_suh, changed_fit.shared_gate_up_suh)
    assert torch.equal(fit.shared_down_svh, changed_fit.shared_down_svh)


def test_streaming_plan_binds_source_bytes_and_core_identity(cores):
    _, core, core_sha = cores
    inputs = _inputs(MatrixInput)
    plan = FitSamplePlan.from_inputs(inputs, block=BLOCK)
    assert all("weight" not in field.name or field.name.endswith("sha256") for field in fields(plan.specs[0]))

    changed_weight = inputs[0].weight_kn.clone()
    changed_weight[0, 0] = torch.nextafter(
        changed_weight[0, 0], torch.tensor(float("inf"), dtype=torch.float32)
    )
    changed = (replace(inputs[0], weight_kn=changed_weight), *inputs[1:])
    fitter = StreamingLayerFitter(
        core,
        plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    with pytest.raises(ValueError, match="fit sample replay differs"):
        fitter.add_fit_matrix(changed[0])

    with pytest.raises(ValueError, match="numeric core identity"):
        StreamingLayerFitter(
            core,
            plan,
            codebook_scale=CODEBOOK_SCALE,
            numeric_core_sha256="unsealed",
        )


def test_no_glm_expert_count_or_square_geometry_assumption(cores):
    _, core, core_sha = cores
    inputs = _inputs(MatrixInput)
    assert len(inputs) == 9
    assert inputs[0].weight_kn.shape == (8, 12)
    assert inputs[2].weight_kn.shape == (12, 8)
    plan = FitSamplePlan.from_inputs(inputs, block=BLOCK)
    fitter = StreamingLayerFitter(
        core,
        plan,
        codebook_scale=CODEBOOK_SCALE,
        numeric_core_sha256=core_sha,
    )
    for item in inputs:
        fitter.add_fit_matrix(item)
    fit = fitter.finish()
    assert not hasattr(fit, "matrices")
    assert fit.shared_gate_up_suh.shape == (8,)
    assert fit.shared_down_svh.shape == (8,)


def test_local_corrected_layer3_frozen_payload_hash_oracle():
    safetensors = pytest.importorskip("safetensors")
    checkpoint = Path(
        "/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
    )
    manifest_path = checkpoint / "r7-experts-layer-003.json"
    shard_path = checkpoint / "r7-experts-layer-003.safetensors"
    if not manifest_path.is_file() or not shard_path.is_file():
        pytest.skip("the frozen corrected layer-3 checkpoint is not installed")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "36e623438c17bf8d143e3d81f712856421ab9fa8155c76ccb7482576c2eb410c"
    )
    assert manifest["shard_sha256"] == (
        "7ca24c32f2f71c034ff29f9471ecac5a86659b03b03db7b4d8fa732918731090"
    )
    assert manifest["allocation_bit_units"] == 2688
    assert manifest["provenance"]["normalization_schema"] == (
        "r10-topology-absolute-normalization-streaming-v1"
    )

    frozen = (
        manifest["shared_vectors"]["gate_up_suh"],
        manifest["shared_vectors"]["down_svh"],
        "model.layers.3.mlp.experts.0.gate_proj.svh",
        "model.layers.3.mlp.experts.0.gate_proj.trellis",
        "model.layers.3.mlp.experts.0.down_proj.suh",
        "model.layers.3.mlp.experts.0.down_proj.trellis",
    )
    with safetensors.safe_open(shard_path, framework="pt", device="cpu") as reader:
        for key in frozen:
            assert tensor_sha256(reader.get_tensor(key)) == manifest["payload_sha256"][key]

    gate_ref = manifest["vector_refs"]["model.layers.3.mlp.experts.0.gate_proj"]
    down_ref = manifest["vector_refs"]["model.layers.3.mlp.experts.0.down_proj"]
    assert gate_ref["suh"] == manifest["shared_vectors"]["gate_up_suh"]
    assert down_ref["svh"] == manifest["shared_vectors"]["down_svh"]
    assert manifest["roundtrip_hashes"]["model.layers.3.mlp.experts.0.gate_proj"][
        "packed_sha256"
    ] == manifest["payload_sha256"][
        "model.layers.3.mlp.experts.0.gate_proj.trellis"
    ]
