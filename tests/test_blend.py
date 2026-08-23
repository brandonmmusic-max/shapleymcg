import pytest

torch = pytest.importorskip("torch")

from quant_pipeline.scoring.blend import BlendedModule


def test_blend_casts_alpha_to_output_dtype_and_preserves_gradient():
    source = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)
    candidate = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)
    alpha = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    wrapped = BlendedModule.wrap(source, candidate, alpha)
    output = wrapped(torch.ones(1, 2, dtype=torch.float32))
    assert output.dtype == torch.float32
    output.sum().backward()
    assert alpha.grad is not None
