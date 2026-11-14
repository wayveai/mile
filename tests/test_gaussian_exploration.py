import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from mile.agents.rl_birdview.models.distributions import DiagGaussianDistribution


@pytest.mark.parametrize('suggestion,expected_mu', [('go', 0.66), ('stop', -0.66)])
def test_acceleration_exploration_uses_positive_standard_deviation(suggestion, expected_mu):
    distribution = DiagGaussianDistribution(action_dim=2)
    mu = torch.zeros(1, 2, requires_grad=True)
    distribution.proba_distribution(mu, torch.zeros_like(mu))
    actual = distribution.exploration_loss(np.array([(suggestion, '')]))
    target_mu = torch.tensor([[expected_mu, 0.0]])
    target_sigma = torch.tensor([[-3.0, 0.0]]).exp()
    expected = kl_divergence(Normal(target_mu, target_sigma), distribution.distribution).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(mu.grad).all()


@pytest.mark.parametrize('suggestion,expected_mu,log_std', [('turn', 0.0, -1.0), ('straight', 3.0, 3.0)])
def test_steering_exploration_uses_log_standard_deviation(suggestion, expected_mu, log_std):
    distribution = DiagGaussianDistribution(action_dim=2)
    mu = torch.zeros(1, 2, requires_grad=True)
    distribution.proba_distribution(mu, torch.zeros_like(mu))
    actual = distribution.exploration_loss(np.array([('', suggestion)]))
    target_mu = torch.tensor([[0.0, expected_mu]])
    target_sigma = torch.tensor([[0.0, log_std]]).exp()
    expected = kl_divergence(Normal(target_mu, target_sigma), distribution.distribution).mean()
    torch.testing.assert_close(actual, expected)
