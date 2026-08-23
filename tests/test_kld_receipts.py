import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from quant_pipeline.cli import command_kld
from quant_pipeline.core.artifacts import sha256_file


def _capture(root, token_sha):
    root.mkdir()
    path = root / "window-0000.safetensors"
    save_file({"logits": torch.tensor([[1.0, 0.0]])}, path, metadata={"token_sha256": token_sha})
    receipt = {
        "sealed_corpus_sha256": "corpus",
        "role": "final",
        "records": [{"file": path.name, "sha256": sha256_file(path), "token_sha256": token_sha}],
    }
    (root / "capture-receipt.json").write_text(json.dumps(receipt))


def test_kld_rejects_mismatched_token_identity(tmp_path):
    teacher = tmp_path / "teacher"
    student = tmp_path / "student"
    _capture(teacher, "teacher-token")
    _capture(student, "student-token")
    args = SimpleNamespace(teacher_dir=str(teacher), student_dir=str(student), output=str(tmp_path / "result.json"))
    with pytest.raises(ValueError, match="token identity"):
        command_kld(args)

