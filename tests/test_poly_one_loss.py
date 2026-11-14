import torch

from mile.constants import SEMANTIC_SEG_WEIGHTS
from mile.losses import SegmentationLoss


def test_poly_one_uses_probability_before_class_weighting():
    prediction = torch.zeros(1, 1, 8, 1, 1, requires_grad=True)
    target = torch.full((1, 1, 1, 1), 3, dtype=torch.long)
    coefficient = 2.0
    loss = SegmentationLoss(
        use_weights=True, poly_one=True, poly_one_coefficient=coefficient,
    )(prediction, target)
    weights = torch.tensor(SEMANTIC_SEG_WEIGHTS, dtype=prediction.dtype)
    # Poly-1 adds epsilon * (1 - p_target); uniform logits give p_target = 1/8.
    expected = weights[3] * torch.log(torch.tensor(8.0)) + coefficient * (1 - 1 / 8)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_poly_one_preserves_the_cross_entropy_ignore_index():
    prediction = torch.zeros(1, 1, 8, 1, 1, requires_grad=True)
    target = torch.full((1, 1, 1, 1), -100, dtype=torch.long)
    loss = SegmentationLoss(
        use_weights=True, poly_one=True, poly_one_coefficient=2.0,
    )(prediction, target)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
