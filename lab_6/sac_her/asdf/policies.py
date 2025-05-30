from typing import Iterable, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from numpy.typing import NDArray
from torch.distributions.normal import Normal

from .extractors import BaseExtractor, DictExtractor
from .utils import unsqueeze_observation

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Multi-layer perceptron (MLP) with ReLU activations.
    
    :param sizes: List of layer sizes.
    :param activation: Activation function to use for all layers except the last.
    :param output_activation: Activation function to use for the last layer.
    :return: A sequential model with the specified layers and activations.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):
    def __init__(
        self,
        extractor: BaseExtractor,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: Type[nn.Module],
        act_limit: float,
    ):
        super().__init__()
        self.extractor = extractor

        self.net = mlp(
            [extractor.n_features] + list(hidden_sizes), activation, activation
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        device = next(self.parameters()).device
        obs = self.extractor(obs, device=device)

        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(
        self,
        extractor: BaseExtractor,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.extractor = extractor

        self.q = mlp(
            [extractor.n_features + act_dim] + list(hidden_sizes) + [1], activation
        )

    def forward(self, obs, act):
        obs = self.extractor(obs)

        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MlpPolicy(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: nn.Module = nn.ReLU,
        extractor_type: BaseExtractor.type = DictExtractor,
        clip_action: bool = True,
    ):
        super().__init__()
        self.clip_action = clip_action
        self.extractor: BaseExtractor = extractor_type(observation_space)

        act_dim = action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            self.extractor,
            act_dim,
            hidden_sizes,
            activation,
            self.act_limit,
        )
        self.q1 = MLPQFunction(self.extractor, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(self.extractor, act_dim, hidden_sizes, activation)

    def act(self, obs: Union[NDArray, dict[str, NDArray]], deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(unsqueeze_observation(obs), deterministic, False)
            a = a.squeeze(dim=0).cpu().numpy()

        if self.clip_action:
            a = np.clip(a, -self.act_limit, self.act_limit)

        return a
