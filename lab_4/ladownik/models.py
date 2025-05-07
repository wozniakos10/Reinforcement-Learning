import torch
from torch import nn
from typing import Literal


class ActorCriticCommonBeginning(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, actor_output):
        super().__init__()
        self.flatten = nn.Flatten()

        self.common_beginning = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.ReLU(),
            nn.LayerNorm(h1_size),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.LayerNorm(h2_size)
        )

        self.actor_score = nn.Sequential(
            nn.Linear(h2_size, actor_output),
            nn.Softmax(dim=1)
        )

        self.critic_score = nn.Sequential(
            nn.Linear(h2_size, 1),
        )

    def forward(self, x, type: Literal["actor", "critic"]):
        cb = self.common_beginning(x)
        if type == "actor":
            return self.actor_score(cb)
        elif type == "critic":
            return self.critic_score(cb)


class Actor(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, actor_output):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.ReLU(),
            nn.LayerNorm(h1_size),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.LayerNorm(h2_size),
            nn.Linear(h2_size, actor_output),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.actor(x)

class Critic(nn.Module):
    def __init__(self, input_size, h1_size, h2_size):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.ReLU(),
            nn.LayerNorm(h1_size),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.LayerNorm(h2_size),
            nn.Linear(h2_size, 1),
        )

    def forward(self,x):
        return self.critic(x)