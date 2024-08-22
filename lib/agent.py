import numpy as np
import torch
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Agent, self).__init__()

        # Actor Network for mu
        actor_hid1_size = num_inputs * 10
        actor_hid3_size = num_outputs * 10
        actor_hid2_size = int(np.sqrt(actor_hid1_size * actor_hid3_size))
        self.actor_mu = nn.Sequential(
            layer_init(nn.Linear(num_inputs, actor_hid1_size)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hid1_size, actor_hid2_size)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hid2_size, actor_hid3_size)),
            nn.Tanh(),
            layer_init(nn.Linear(actor_hid3_size, num_outputs), std=0.01),
            nn.Tanh()  # [-1, 1]
        )

        # Diagonal covariance matrix variables are separately trained
        self.actor_logstd = nn.Parameter(torch.zeros(num_outputs))

        # Critic Network
        critic_hid1_size = num_inputs * 10
        critic_hid3_size = 5
        critic_hid2_size = int(np.sqrt(critic_hid1_size * critic_hid3_size))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_inputs, critic_hid1_size)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hid1_size, critic_hid2_size)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hid2_size, critic_hid3_size)),
            nn.Tanh(),
            layer_init(nn.Linear(critic_hid3_size, 1), std=1.0)
        )

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)
