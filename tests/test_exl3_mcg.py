import sys
import shutil

import pytest

torch = pytest.importorskip("torch")

from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec


def _fake_codec_tree(tmp_path):
    package = tmp_path / "r7_encoder"
    package.mkdir()
    for name in ("__init__.py", "constants.py", "types.py", "determinism.py"):
        (package / name).write_text("\n")
    (package / "trellis.py").write_text(
        "class CodecConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
    )
    (package / "r10_codec.py").write_text(
        "import torch\n"
        "from types import SimpleNamespace\n"
        "class R10TrellisCodec:\n"
        "    def __init__(self, config):\n"
        "        self.config = config\n"
        "    def encode_bits(self, tensor_id, weight_hf, covariance, bits, suh, svh, sigma_reg, provenance):\n"
        "        return {bit: SimpleNamespace(\n"
        "            trellis=torch.zeros((bit,), dtype=torch.uint8),\n"
        "            reconstructed_kn=weight_hf.T.contiguous(),\n"
        "            suh=suh, svh=svh, packed_sha256=f'packed-{bit}',\n"
        "            reconstruction_sha256=f'recon-{bit}', provenance=provenance\n"
        "        ) for bit in bits}\n"
    )
    numeric_core = tmp_path / "numeric.py"
    extension = tmp_path / "extension.so"
    numeric_core.write_text("# sealed numeric core\n")
    extension.write_bytes(b"sealed-extension")
    return numeric_core, extension


def test_fake_pinned_interface_seal_and_geometry(tmp_path):
    numeric_core, extension = _fake_codec_tree(tmp_path)
    codec = Exl3MCGCodec(source_root=tmp_path, numeric_core=numeric_core, extension=extension, device="cpu")
    weight = torch.zeros((128, 128), dtype=torch.float32)
    vector = torch.ones(128, dtype=torch.float32)
    try:
        result = codec.encode_candidates(
            unit_id="L1.E2.gate_proj",
            weight_hf=weight,
            covariance=torch.eye(128),
            bits=(3, 4, 5),
            input_vector=vector,
            output_vector=vector,
        )
        assert tuple(result) == (3, 4, 5)
        assert result[3].reconstructed.shape == weight.shape
        assert result[3].stored_bytes == 3 + 2 * 128 * 4
        assert result[3].metadata["codec_identity"]["sigma_reg"] == 0.025
        assert result[3].metadata["codec_identity"]["backend_class"] == "r7_encoder.r10_codec.R10TrellisCodec"
        assert set(result[3].metadata["codec_identity"]["environment"]) == {"python", "machine", "torch", "torch_cuda", "compute_capability"}
        with pytest.raises(ValueError, match="divisible by 128"):
            codec._parse_unit("bad", (127, 128))
        (tmp_path / "r7_encoder" / "constants.py").write_text("DRIFT = True\n")
        with pytest.raises(RuntimeError, match="closure drifted"):
            codec._codec()
    finally:
        for name in list(sys.modules):
            if name == "r7_encoder" or name.startswith("r7_encoder."):
                sys.modules.pop(name)
        while str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_numeric_identity_binds_sigma_and_code_but_not_mount_path(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    numeric_one, extension_one = _fake_codec_tree(one)
    shutil.copytree(one / "r7_encoder", two / "r7_encoder")
    two.mkdir(exist_ok=True)
    numeric_two = two / "numeric.py"
    extension_two = two / "extension.so"
    shutil.copy2(numeric_one, numeric_two)
    shutil.copy2(extension_one, extension_two)
    first = Exl3MCGCodec(source_root=one, numeric_core=numeric_one, extension=extension_one, device="cpu")
    remounted = Exl3MCGCodec(source_root=two, numeric_core=numeric_two, extension=extension_two, device="cpu")
    assert first.identity == remounted.identity
    changed_sigma = Exl3MCGCodec(
        source_root=two, numeric_core=numeric_two, extension=extension_two, device="cuda:0", sigma_reg=0.05
    )
    assert changed_sigma.identity != first.identity
    with pytest.raises(ValueError, match="sigma_reg"):
        Exl3MCGCodec(source_root=one, numeric_core=numeric_one, extension=extension_one, sigma_reg=float("inf"))
