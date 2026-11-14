import pytest
import torch

from mile.models.frustum_pooling import FrustumPooling


@pytest.mark.parametrize('point', [
    [-2.1, 0.0, 0.0], [0.0, 2.1, 0.0], [0.0, 0.0, -10.1],
    [2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 10.0],
])
@pytest.mark.parametrize('quick', [True, False])
def test_outside_points_do_not_enter_the_boundary_voxel(point, quick):
    pool = FrustumPooling((4, 4), 1.0, 0, (1.0, 2.0, 1.0), 1, use_quickcumsum=quick)
    features = torch.ones(1, 1, 1, 1, 1, 1, requires_grad=True)
    geometry = torch.tensor(point).reshape(1, 1, 1, 1, 1, 3)
    result = pool.voxel_pooling(geometry, features, torch.zeros(0))
    assert result.count_nonzero() == 0
    result.sum().backward()
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


def test_point_on_lower_boundary_is_retained():
    pool = FrustumPooling((4, 4), 1.0, 0, (1.0, 2.0, 1.0), 1)
    features = torch.tensor([[[[[[7.0]]]]]])
    geometry = torch.tensor([-2.0, 2.0, -10.0]).reshape(1, 1, 1, 1, 1, 3)
    result = pool.voxel_pooling(geometry, features, torch.zeros(0))
    assert result.sum().item() == 7
    assert result[0, 0, 0, 0].item() == 7
