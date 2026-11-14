import torch
import torch.nn as nn
import torch.nn.functional as F

from mile.constants import SEMANTIC_SEG_WEIGHTS


class SegmentationLoss(nn.Module):
    def __init__(self, use_top_k=False, top_k_ratio=1.0, use_weights=False, poly_one=False, poly_one_coefficient=0.0):
        super().__init__()
        if use_top_k and not 0 < top_k_ratio <= 1:
            raise ValueError('top_k_ratio must be in (0, 1]')
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.poly_one = poly_one
        self.poly_one_coefficient = poly_one_coefficient

        if self.use_weights:
            self.weights = SEMANTIC_SEG_WEIGHTS

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        if self.use_weights:
            weights = torch.tensor(self.weights, dtype=prediction.dtype, device=prediction.device)
        else:
            weights = None
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=weights,
        )

        if self.poly_one:
            prob = torch.exp(-F.cross_entropy(prediction, target, reduction='none'))
            loss_poly_one = self.poly_one_coefficient * (1-prob)
            loss = loss + loss_poly_one

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = max(1, int(self.top_k_ratio * loss.shape[2]))
            loss = loss.topk(k, dim=-1)[0]

        return torch.mean(loss)


class RegressionLoss(nn.Module):
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.sum() * 0.0

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        return loss[mask].mean()


class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""
    def __init__(self, remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        start = 1 if self.remove_first_timestamp else 0
        posterior_mu = posterior_mu[:, start:]
        posterior_sigma = posterior_sigma[:, start:]
        prior_mu = prior_mu[:, start:]
        prior_sigma = prior_sigma[:, start:]
        posterior_var = posterior_sigma ** 2
        prior_var = prior_sigma ** 2

        kl_div = (
                torch.log(prior_sigma) - torch.log(posterior_sigma) - 0.5
                + (posterior_var + (posterior_mu - prior_mu) ** 2) / (2 * prior_var)
        )
        if kl_div.numel() == 0:
            return kl_div.sum()

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss


class KLLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss(remove_first_timestamp=True)

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss
