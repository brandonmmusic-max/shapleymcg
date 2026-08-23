from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.artifacts import sha256_file
from .protocols import CodecCandidate


@dataclass(frozen=True)
class GenericTensorId:
    """Geometry-bearing ID accepted by the pinned codec without GLM constants."""

    key: str
    k: int
    n: int
    layer: int = 0
    expert: int = 0
    projection: str = "projection"


class Exl3MCGCodec:
    """Adapter for Brandon's pinned corrected EXL3/MCG numeric implementation.

    The implementation is loaded from an explicitly supplied source tree. No
    machine-specific path is assumed and all executable inputs are hash-bound.
    The source tree must expose `r7_encoder.r10_codec` and `r7_encoder.trellis`.
    """

    name = "exl3-mcg-corrected"

    def __init__(
        self,
        *,
        source_root: str | Path,
        numeric_core: str | Path,
        extension: str | Path,
        device: str = "cuda:0",
        sigma_reg: float = 0.025,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.numeric_core = Path(numeric_core).resolve()
        self.extension = Path(extension).resolve()
        for path in (self.numeric_core, self.extension):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not (self.source_root / "r7_encoder" / "r10_codec.py").is_file():
            raise FileNotFoundError(self.source_root / "r7_encoder" / "r10_codec.py")
        self.device = device
        self.sigma_reg = sigma_reg
        self._codec_instance = None
        closure_files = ("__init__.py", "r10_codec.py", "trellis.py", "constants.py", "types.py", "determinism.py")
        closure = {
            name: sha256_file(self.source_root / "r7_encoder" / name)
            for name in closure_files
        }
        self.identity = {
            "source_root": str(self.source_root),
            "python_closure_sha256": closure,
            "numeric_core": str(self.numeric_core),
            "numeric_core_sha256": sha256_file(self.numeric_core),
            "extension": str(self.extension),
            "extension_sha256": sha256_file(self.extension),
        }

    def _codec(self):
        for filename, expected in self.identity["python_closure_sha256"].items():
            if sha256_file(self.source_root / "r7_encoder" / filename) != expected:
                raise RuntimeError(f"sealed EXL3/MCG Python closure drifted: {filename}")
        if self._codec_instance is not None:
            return self._codec_instance
        root = str(self.source_root)
        incumbents = sorted(name for name in sys.modules if name == "r7_encoder" or name.startswith("r7_encoder."))
        if incumbents:
            raise RuntimeError(f"refusing cached r7_encoder modules before sealed import: {incumbents}")
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
        r10 = importlib.import_module("r7_encoder.r10_codec")
        trellis = importlib.import_module("r7_encoder.trellis")
        expected_package = (self.source_root / "r7_encoder").resolve()
        for module in (r10, trellis):
            loaded = Path(str(getattr(module, "__file__", ""))).resolve()
            if expected_package not in loaded.parents:
                raise RuntimeError(f"sealed module resolved outside source_root: {loaded}")
        config = trellis.CodecConfig(
            device=self.device,
            sigma_reg=self.sigma_reg,
            numeric_core=self.numeric_core,
            numeric_core_sha256=self.identity["numeric_core_sha256"],
            extension=self.extension,
            extension_sha256=self.identity["extension_sha256"],
            verify_files=True,
        )
        self._codec_instance = r10.R10TrellisCodec(config)
        return self._codec_instance

    @staticmethod
    def _parse_unit(unit_id: str, shape: tuple[int, int]) -> GenericTensorId:
        match = re.fullmatch(r"L(\d+)\.E(\d+)\.(gate_proj|up_proj|down_proj)", unit_id)
        layer, expert, projection = (0, 0, "projection") if not match else (int(match[1]), int(match[2]), match[3])
        n, k = map(int, shape)
        if k % 128 or n % 128:
            raise ValueError(
                f"EXL3/MCG numeric core requires K and N divisible by 128; "
                f"{unit_id} has [N,K]={shape}"
            )
        return GenericTensorId(unit_id, k=k, n=n, layer=layer, expert=expert, projection=projection)

    def encode_candidates(
        self,
        *,
        unit_id: str,
        weight_hf: Any,
        covariance: Any,
        bits: Sequence[int],
        input_vector: Any,
        output_vector: Any,
        provenance: dict | None = None,
    ) -> dict[int, CodecCandidate]:
        if any(int(bit) not in (3, 4, 5) for bit in bits):
            raise ValueError("the pinned R10 adapter currently supports K3/K4/K5; K2 requires the declared extension")
        tensor_id = self._parse_unit(unit_id, tuple(weight_hf.shape))
        metadata = dict(provenance or {}) | {"codec_identity": self.identity}
        encoded = self._codec().encode_bits(
            tensor_id=tensor_id,
            weight_hf=weight_hf,
            covariance=covariance,
            bits=tuple(map(int, bits)),
            suh=input_vector,
            svh=output_vector,
            sigma_reg=self.sigma_reg,
            provenance=metadata,
        )
        return {
            bit: CodecCandidate(
                bits=bit,
                packed=value.trellis,
                reconstructed=value.reconstructed_kn.T.contiguous(),
                stored_bytes=int(value.trellis.numel() * value.trellis.element_size() + value.suh.numel() * value.suh.element_size() + value.svh.numel() * value.svh.element_size()),
                packed_sha256=value.packed_sha256,
                reconstruction_sha256=value.reconstruction_sha256,
                metadata=dict(value.provenance),
            )
            for bit, value in encoded.items()
        }
