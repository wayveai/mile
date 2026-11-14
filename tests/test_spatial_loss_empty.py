import pytest
import torch

from mile.losses import SpatialRegressionLoss


@pytest.mark.parametrize('norm', [1, 2])
def test_all_ignored_pixels_allow_backward(norm):
    prediction = torch.randn(2, 3, 2, 4, 5, requires_grad=True)
    target = torch.full_like(prediction, 255)
    loss = SpatialRegressionLoss(norm)(prediction, target)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_valid_pixels_keep_channel_sum_reduction():
    prediction = torch.tensor([[[[[2.0, 7.0]], [[3.0, 8.0]]]]], requires_grad=True)
    target = torch.tensor([[[[[0.0, 255.0]], [[0.0, 255.0]]]]])
    assert SpatialRegressionLoss(1)(prediction, target).item() == 5
