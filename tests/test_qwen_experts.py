import torch

from quant_pipeline.scoring.qwen_experts import project_qwen_expert_residuals


def test_projected_experts_accept_normal_batched_shapes():
    generator = torch.Generator().manual_seed(3)
    batch, tokens, hidden, intermediate, experts, top_k, rank = 2, 3, 4, 5, 2, 1, 3
    x = torch.randn(batch, tokens, hidden, generator=generator)
    indices = torch.randint(experts, (batch, tokens, top_k), generator=generator)
    weights = torch.ones(batch, tokens, top_k)
    gate = torch.randn(experts, intermediate, hidden, generator=generator)
    up = torch.randn(experts, intermediate, hidden, generator=generator)
    down = torch.randn(experts, hidden, intermediate, generator=generator)
    gradients = torch.randn(rank, batch, tokens, hidden, generator=generator)
    result = project_qwen_expert_residuals(
        x,
        indices,
        weights,
        gate,
        up,
        down,
        gate + 0.01,
        up - 0.01,
        down + 0.02,
        gradients,
    )
    assert result["projected_residuals"].shape == (experts, rank)
    assert torch.isfinite(result["projected_residuals"]).all()

