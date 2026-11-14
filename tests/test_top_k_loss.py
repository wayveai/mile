import pytest
import torch

from mile.losses import SegmentationLoss


def test_small_top_k_ratio_keeps_the_hardest_pixel():
    prediction = torch.tensor([[[[[3.0, 0.0]], [[0.0, 3.0]]]]], requires_grad=True)
    target = torch.zeros(1, 1, 1, 2, dtype=torch.long)
    loss = SegmentationLoss(use_top_k=True, top_k_ratio=0.01)(prediction, target)
    expected = torch.nn.functional.cross_entropy(prediction[0, 0, :, :, 1:].unsqueeze(0), target[0, :, :, 1:])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize('ratio', [0.0, -0.1, 1.1])
def test_invalid_top_k_ratios_are_rejected(ratio):
    with pytest.raises(ValueError, match='top_k_ratio'):
        SegmentationLoss(use_top_k=True, top_k_ratio=ratio)
