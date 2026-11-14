import torch

from mile.models.frustum_pooling import FrustumPooling


def test_frustum_rebuilds_for_new_feature_resolution():
    pool = FrustumPooling((4, 4), 1.0, 0, (1.0, 3.0, 1.0), 4)
    pool.initialize_frustum(torch.zeros(1, 1, 2, 2, 3, 1))
    assert pool.frustum.shape == (2, 2, 3, 3)
    pool.initialize_frustum(torch.zeros(1, 1, 2, 3, 2, 1))
    assert pool.frustum.shape == (2, 3, 2, 3)
    torch.testing.assert_close(pool.frustum[0, -1, -1], torch.tensor([7.0, 11.0, 1.0]))


def test_reused_pool_matches_new_pool_after_resolution_change():
    args = ((8, 8), 1.0, 0, (1.0, 2.0, 1.0), 1)
    pool = FrustumPooling(*args).eval()
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    pool(torch.ones(1, 1, 1, 2, 2, 1), intrinsics, pose)
    features = torch.ones(1, 1, 1, 3, 2, 1)
    actual = pool(features, intrinsics, pose)
    expected = FrustumPooling(*args).eval()(features, intrinsics, pose)
    torch.testing.assert_close(actual, expected)
