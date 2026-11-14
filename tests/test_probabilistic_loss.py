import torch
from torch.distributions import Normal, kl_divergence

from mile.losses import ProbabilisticLoss


def test_first_posterior_is_unconstrained_by_default():
    prior_mu = torch.tensor([[[10.0], [1.0], [2.0]]], requires_grad=True)
    prior_sigma = torch.tensor([[[3.0], [2.0], [0.5]]], requires_grad=True)
    posterior_mu = torch.tensor([[[100.0], [4.0], [3.0]]], requires_grad=True)
    posterior_sigma = torch.tensor([[[8.0], [0.5], [2.0]]], requires_grad=True)
    expected = kl_divergence(
        Normal(posterior_mu[:, 1:], posterior_sigma[:, 1:]),
        Normal(prior_mu[:, 1:], prior_sigma[:, 1:]),
    ).sum(-1).mean()
    actual = ProbabilisticLoss()(prior_mu, prior_sigma, posterior_mu, posterior_sigma)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert posterior_mu.grad[:, 0].count_nonzero() == 0
    assert posterior_sigma.grad[:, 0].count_nonzero() == 0


def test_single_timestamp_has_differentiable_zero_loss():
    mu = torch.ones(2, 1, 3, requires_grad=True)
    sigma = torch.ones_like(mu)
    loss = ProbabilisticLoss()(mu, sigma, mu, sigma)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(mu.grad, torch.zeros_like(mu))


def test_first_timestamp_can_be_included_explicitly():
    prior_mu = torch.tensor([[[2.0], [1.0]]])
    posterior_mu = torch.tensor([[[4.0], [3.0]]])
    prior_sigma = torch.tensor([[[2.0], [0.5]]])
    posterior_sigma = torch.tensor([[[0.5], [2.0]]])
    expected = kl_divergence(
        Normal(posterior_mu, posterior_sigma), Normal(prior_mu, prior_sigma),
    ).sum(-1).mean()
    actual = ProbabilisticLoss(remove_first_timestamp=False)(
        prior_mu, prior_sigma, posterior_mu, posterior_sigma,
    )
    torch.testing.assert_close(actual, expected)
