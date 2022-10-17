import torch
import torch.nn as nn


class RepresentationModel(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1

        self.module = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(True),
            nn.Linear(in_channels, 2*self.latent_dim),
        )

    def forward(self, x):
        def sigmoid2(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
            return 2 * torch.sigmoid(tensor / 2) + min_value

        mu_log_sigma = self.module(x)
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)

        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class RSSM(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_state_dim, state_dim, action_latent_dim, receptive_field,
                 use_dropout=False,
                 dropout_probability=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim
        self.action_latent_dim = action_latent_dim
        self.receptive_field = receptive_field
        # Sometimes unroll the prior instead of always updating with the posterior
        # so that the model learns to unroll for more than 1 step in the future when imagining
        self.use_dropout = use_dropout
        self.dropout_probability = dropout_probability

        # Map input of the gru to a space with easier temporal dynamics
        self.pre_gru_net = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )

        self.recurrent_model = nn.GRUCell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
        )

        # Map action to a higher dimensional input
        self.posterior_action_module = nn.Sequential(
            nn.Linear(action_dim, self.action_latent_dim),
            nn.LeakyReLU(True),
        )

        self.posterior = RepresentationModel(
            in_channels=hidden_state_dim + embedding_dim + self.action_latent_dim,
            latent_dim=state_dim,
        )

        # Map action to a higher dimensional input
        self.prior_action_module = nn.Sequential(
            nn.Linear(action_dim, self.action_latent_dim),
            nn.LeakyReLU(True),
        )
        self.prior = RepresentationModel(in_channels=hidden_state_dim + self.action_latent_dim, latent_dim=state_dim)
        self.active_inference = False
        if self.active_inference:
            print('ACTIVE INFERENCE!!')

    def forward(self, input_embedding, action, use_sample=True, policy=None):
        """
        Inputs
        ------
            input_embedding: torch.Tensor size (B, S, C)
            action: torch.Tensor size (B, S, 2)
            use_sample: bool
                whether to use sample from the distributions, or taking the mean

        Returns
        -------
            output: dict
                prior: dict
                    hidden_state: torch.Tensor (B, S, C_h)
                    sample: torch.Tensor (B, S, C_s)
                    mu: torch.Tensor (B, S, C_s)
                    sigma: torch.Tensor (B, S, C_s)
                posterior: dict
                    hidden_state: torch.Tensor (B, S, C_h)
                    sample: torch.Tensor (B, S, C_s)
                    mu: torch.Tensor (B, S, C_s)
                    sigma: torch.Tensor (B, S, C_s)
        """
        output = {
            'prior': [],
            'posterior': [],
        }

        # Initialisation
        batch_size, sequence_length, _ = input_embedding.shape
        h_t = input_embedding.new_zeros((batch_size, self.hidden_state_dim))
        sample_t = input_embedding.new_zeros((batch_size, self.state_dim))
        for t in range(sequence_length):
            if t == 0:
                action_t = torch.zeros_like(action[:, 0])
            else:
                action_t = action[:, t-1]
            output_t = self.observe_step(
                h_t, sample_t, action_t, input_embedding[:, t], use_sample=use_sample, policy=policy,
            )
            # During training sample from the posterior, except when using dropout
            # always use posterior for the first frame
            use_prior = self.training and self.use_dropout and torch.rand(1).item() < self.dropout_probability and t > 0

            if use_prior:
                sample_t = output_t['prior']['sample']
            else:
                sample_t = output_t['posterior']['sample']
            h_t = output_t['prior']['hidden_state']

            for key, value in output_t.items():
                output[key].append(value)

        output = self.stack_list_of_dict_tensor(output, dim=1)
        return output

    def observe_step(self, h_t, sample_t, action_t, embedding_t, use_sample=True, policy=None):
        imagine_output = self.imagine_step(h_t, sample_t, action_t, use_sample, policy=policy)

        latent_action_t = self.posterior_action_module(action_t)
        posterior_mu_t, posterior_sigma_t = self.posterior(
            torch.cat([imagine_output['hidden_state'], embedding_t, latent_action_t], dim=-1)
        )

        sample_t = self.sample_from_distribution(posterior_mu_t, posterior_sigma_t, use_sample=use_sample)

        posterior_output = {
            'hidden_state': imagine_output['hidden_state'],
            'sample': sample_t,
            'mu': posterior_mu_t,
            'sigma': posterior_sigma_t,
        }

        output = {
            'prior': imagine_output,
            'posterior': posterior_output,
        }

        return output

    def imagine_step(self, h_t, sample_t, action_t, use_sample=True, policy=None):
        if self.active_inference:
            # Predict action with policy
            action_t = policy(torch.cat([h_t, sample_t], dim=-1))

        latent_action_t = self.prior_action_module(action_t)

        input_t = self.pre_gru_net(sample_t)
        h_t = self.recurrent_model(input_t, h_t)
        prior_mu_t, prior_sigma_t = self.prior(torch.cat([h_t, latent_action_t], dim=-1))
        sample_t = self.sample_from_distribution(prior_mu_t, prior_sigma_t, use_sample=use_sample)
        imagine_output = {
            'hidden_state': h_t,
            'sample': sample_t,
            'mu': prior_mu_t,
            'sigma': prior_sigma_t,
        }
        return imagine_output

    @staticmethod
    def sample_from_distribution(mu, sigma, use_sample):
        sample = mu
        if use_sample:
            noise = torch.randn_like(sample)
            sample = sample + sigma * noise
        return sample

    @staticmethod
    def stack_list_of_dict_tensor(output, dim=1):
        new_output = {}
        for outter_key, outter_value in output.items():
            if len(outter_value) > 0:
                new_output[outter_key] = dict()
                for inner_key in outter_value[0].keys():
                    new_output[outter_key][inner_key] = torch.stack([x[inner_key] for x in outter_value], dim=dim)
        return new_output
