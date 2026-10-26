"""MAAC: soft actor-critic with multihead, action-conditioned attentive critics."""
import copy
import math

import numpy as np
import torch
from torch import nn

from algorithms.benchmark_base import (BenchmarkLearner, Decision, tensor, mlp,
                                       masked_mean, soft_update)


class AttentionCritic(nn.Module):
    def __init__(self, observation_dim=42, hidden=64, heads=4, actions=5):
        super().__init__()
        if hidden % heads:
            raise ValueError('Hidden size must be divisible by attention heads')
        self.heads, self.width, self.actions = heads, hidden//heads, actions
        self.embedding = nn.Sequential(nn.Linear(observation_dim+actions, hidden), nn.LeakyReLU())
        self.query, self.key, self.value = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden)
        self.output = mlp(2*hidden, 1, hidden)

    def forward(self, observations, actions, present, return_attention=False):
        batch, agents, _ = observations.shape
        actual = self.embedding(torch.cat([observations, actions], -1))
        choices = torch.eye(self.actions, device=observations.device).view(1, 1, self.actions, self.actions).expand(batch, agents, -1, -1)
        ego = self.embedding(torch.cat([observations.unsqueeze(2).expand(-1, -1, self.actions, -1), choices], -1))
        q = self.query(ego).reshape(batch, agents, self.actions, self.heads, self.width)
        k = self.key(actual).reshape(batch, agents, self.heads, self.width)
        v = torch.nn.functional.leaky_relu(self.value(actual)).reshape(batch, agents, self.heads, self.width)
        scores = torch.einsum('bnahd,bmhd->bnham', q, k)/math.sqrt(self.width)
        eligible = present[:, None, None, None, :].bool() & ~torch.eye(agents, device=observations.device, dtype=torch.bool)[None, :, None, None, :]
        weights = torch.softmax(scores.masked_fill(~eligible, -1e9), -1)*eligible
        # A singleton has zero neighbors, not NaN attention on an empty set.
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-9)
        context = torch.einsum('bnham,bmhd->bnahd', weights, v).reshape(batch, agents, self.actions, -1)
        values = self.output(torch.cat([ego, context], -1)).squeeze(-1)
        return (values, weights) if return_attention else values


class MAAC(BenchmarkLearner):
    def __init__(self, method='maac_common', config=None, seed=42):
        super().__init__(method, config, seed)
        self.actor = mlp(self.obs_dim, 5, self.config.hidden)
        self.critic = AttentionCritic(self.obs_dim, self.config.hidden)
        self.target_actor, self.target_critic = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.optimizers['actor'] = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.optimizers['critic'] = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def act(self, frame, epsilon=0.0, evaluation=False):
        with torch.no_grad():
            distribution = torch.distributions.Categorical(logits=self.actor(tensor(frame.observations)))
            actions = distribution.probs.argmax(-1) if evaluation else distribution.sample()
        return Decision(actions.numpy(), distribution.probs.numpy())

    def learn(self):
        rows, states, following = self.batch()
        chosen = torch.tensor(np.stack([r['actions'] for r in rows]), dtype=torch.long)
        actions = nn.functional.one_hot(chosen, 5).float()
        valid, done = states['present'], tensor([r['done'] for r in rows])
        rewards = tensor([r['reward'] for r in rows]).unsqueeze(-1)
        with torch.no_grad():
            logits = self.target_actor(following['obs'])
            distribution = torch.distributions.Categorical(logits=logits)
            sampled = nn.functional.one_hot(distribution.sample(), 5).float()
            all_q = self.target_critic(following['obs'], sampled, following['present'])
            soft = (distribution.probs*(all_q-self.config.temperature*torch.log_softmax(logits, -1))).sum(-1)
            target = rewards+self.config.gamma*(1-done)*soft
        actual = self.critic(states['obs'], actions, valid).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        critic_loss = masked_mean((actual-target).square(), valid)
        self.optimize('critic', critic_loss, self.critic.parameters())
        logits = self.actor(states['obs'])
        probabilities, log_probabilities = torch.softmax(logits, -1), torch.log_softmax(logits, -1)
        with torch.no_grad():
            sampled = nn.functional.one_hot(torch.distributions.Categorical(probs=probabilities).sample(), 5).float()
            all_q = self.critic(states['obs'], sampled, valid)
            # Counterfactual baseline sums over the ego action, holding others'
            # choices fixed. Exact discrete expectation avoids score-function noise.
            baseline = (probabilities*all_q).sum(-1, keepdim=True)
        advantage = all_q-baseline
        actor_loss = masked_mean((probabilities*(self.config.temperature*log_probabilities-advantage)).sum(-1), valid)
        self.optimize('actor', actor_loss, self.actor.parameters())
        soft_update(self.actor, self.target_actor, self.config.tau)
        soft_update(self.critic, self.target_critic, self.config.tau)
        self.updates += 1
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
