"""Trust-MARL SUMO adaptation: temporal hybrid MASAC plus trust/game priors."""
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from algorithms.benchmark_base import (BenchmarkLearner, Decision, tensor, mlp,
                                       masked_mean, soft_update)
from algorithms.common_on_policy import TemporalNetwork


class HybridActor(TemporalNetwork):
    def __init__(self, inputs, hidden):
        super().__init__(inputs, 5, hidden)

    def sample(self, histories, prior, lane_masks, deterministic=False):
        outputs = self(histories)
        normal = Normal(outputs[..., 0], outputs[..., 1].clamp(-5, 2).exp())
        latent = normal.mean if deterministic else normal.rsample()
        bounded = torch.tanh(latent)
        acceleration = -.75+2.25*bounded
        log_continuous = normal.log_prob(latent)-torch.log(1-bounded.square()+1e-6)-np.log(2.25)
        logits = (outputs[..., 2:]+prior).masked_fill(~lane_masks.bool(), -1e9)
        log_lane = torch.log_softmax(logits, -1)
        probabilities = log_lane.exp()
        lane = probabilities.argmax(-1) if deterministic else torch.distributions.Categorical(probs=probabilities).sample()
        return acceleration, lane, log_continuous, probabilities, log_lane


class HybridCritic(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.encoder = TemporalNetwork(inputs, hidden, hidden)
        self.value = mlp(2*hidden+8, 1, hidden)

    def forward(self, histories, acceleration, lanes, present):
        encoded = self.encoder(histories)
        # Each agent's gradient uses its own continuous action, holding others'
        # sampled actions fixed. Critic encoders still learn jointly.
        own = torch.cat([encoded, acceleration.unsqueeze(-1), lanes], -1)
        others = torch.cat([encoded, acceleration.detach().unsqueeze(-1), lanes.detach()], -1)*present.unsqueeze(-1)
        pool = (others.sum(1, keepdim=True)-others)/(present.sum(1, keepdim=True)-present).clamp_min(1).unsqueeze(-1)
        candidates = torch.eye(3, device=encoded.device).view(1, 1, 3, 3).expand(*encoded.shape[:2], 3, 3)
        own_candidates = torch.cat([encoded.unsqueeze(2).expand(-1, -1, 3, -1),
                                    acceleration[..., None, None].expand(-1, -1, 3, 1), candidates], -1)
        return self.value(torch.cat([own_candidates, pool.unsqueeze(2).expand(-1, -1, 3, -1)], -1)).squeeze(-1)


class TrustMARL(BenchmarkLearner):
    def __init__(self, method='trust_marl_common', config=None, seed=42):
        super().__init__(method, config, seed)
        self.actor = HybridActor(self.obs_dim, self.config.hidden)
        self.q1, self.q2 = HybridCritic(self.obs_dim, self.config.hidden), HybridCritic(self.obs_dim, self.config.hidden)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1).requires_grad_(False), copy.deepcopy(self.q2).requires_grad_(False)
        self.optimizers['actor'] = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.optimizers['critic'] = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=self.config.critic_lr)

    def act(self, frame, epsilon=0.0, evaluation=False):
        with torch.no_grad():
            acceleration, lane, _, probabilities, _ = self.actor.sample(
                tensor(frame.histories), tensor(frame.game_prior), tensor(frame.lane_masks), evaluation)
        broadcast = np.zeros((5, 5), np.float32)
        broadcast[:, :3] = probabilities.numpy()
        return Decision(lane.numpy(), broadcast, acceleration.numpy())

    def learn(self):
        rows, states, following = self.batch()
        chosen = torch.tensor(np.stack([r['actions'] for r in rows]), dtype=torch.long)
        acceleration = tensor([r['parameters'] for r in rows])
        lanes = nn.functional.one_hot(chosen, 3).float()
        rewards = tensor([r['reward'] for r in rows]).unsqueeze(-1)
        done, valid = tensor([r['done'] for r in rows]), states['present']
        temperature = self.config.temperature
        with torch.no_grad():
            acc, lane, log_cont, probabilities, log_lane = self.target_actor.sample(
                following['history'], following['prior'], following['lanes'])
            one_hot = nn.functional.one_hot(lane, 3).float()
            q = torch.minimum(self.target_q1(following['history'], acc, one_hot, following['present']),
                              self.target_q2(following['history'], acc, one_hot, following['present']))
            soft_value = (probabilities*(q-temperature*(log_cont.unsqueeze(-1)+log_lane))).sum(-1)
            target = rewards+self.config.gamma*(1-done)*soft_value
        q1 = self.q1(states['history'], acceleration, lanes, valid).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        q2 = self.q2(states['history'], acceleration, lanes, valid).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        critic_loss = masked_mean((q1-target).square()+(q2-target).square(), valid)
        self.optimize('critic', critic_loss, list(self.q1.parameters())+list(self.q2.parameters()))
        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        try:
            acc, lane, log_cont, probabilities, log_lane = self.actor.sample(states['history'], states['prior'], states['lanes'])
            one_hot = nn.functional.one_hot(lane, 3).float()
            q = torch.minimum(self.q1(states['history'], acc, one_hot, valid), self.q2(states['history'], acc, one_hot, valid))
            actor_loss = masked_mean((probabilities*(temperature*(log_cont.unsqueeze(-1)+log_lane)-q)).sum(-1), valid)
            self.optimize('actor', actor_loss, self.actor.parameters())
        finally:
            self.q1.requires_grad_(True)
            self.q2.requires_grad_(True)
        for source, target_net in [(self.actor, self.target_actor), (self.q1, self.target_q1), (self.q2, self.target_q2)]:
            soft_update(source, target_net, self.config.tau)
        self.updates += 1
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
