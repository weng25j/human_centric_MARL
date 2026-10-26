"""Z-Merge Double PDQN on real 42-feature observations and the common reward."""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algorithms.benchmark_base import BenchmarkLearner, Decision, tensor, masked_mean
from algorithms.z_merge import ZMergeParameterNetwork, ZMergeQNetwork


def action_masks(lanes):
    masks = torch.ones((*lanes.shape[:-1], 5), dtype=torch.bool, device=lanes.device)
    masks[..., 0], masks[..., 1] = lanes[..., 1].bool(), lanes[..., 2].bool()
    return masks


class ZMergeCommon(BenchmarkLearner):
    def __init__(self, method='z_merge_common', config=None, seed=42):
        super().__init__(method, config, seed)
        self.q = ZMergeQNetwork(42, 5, 5)
        self.parameters_net = ZMergeParameterNetwork(42, 5, 1)
        self.target_q, self.target_parameters = copy.deepcopy(self.q), copy.deepcopy(self.parameters_net)
        self.target_q.requires_grad_(False)
        self.target_parameters.requires_grad_(False)
        self.optimizers['critic'] = torch.optim.Adam(self.q.parameters(), lr=self.config.critic_lr)
        self.optimizers['actor'] = torch.optim.Adam(self.parameters_net.parameters(), lr=self.config.actor_lr)

    def act(self, frame, epsilon=0.0, evaluation=False):
        with torch.no_grad():
            observations = tensor(frame.observations)
            parameters = self.parameters_net(observations)
            q = self.q(observations, parameters)
            mask = action_masks(tensor(frame.lane_masks))
            greedy = q.masked_fill(~mask, -torch.inf).argmax(-1).numpy()
        probabilities = np.zeros((5, 5), np.float32)
        actions = greedy.copy()
        for i in range(5):
            legal = np.flatnonzero(mask[i].numpy())
            probabilities[i, legal] = epsilon/len(legal) if not evaluation else 0
            probabilities[i, greedy[i]] += 1-epsilon if not evaluation else 1
            if not evaluation:
                actions[i] = np.random.choice(5, p=probabilities[i]/probabilities[i].sum())
        return Decision(actions, probabilities, parameters.numpy())

    def learn(self):
        rows, states, following = self.batch()
        actions = torch.tensor(np.stack([r['actions'] for r in rows]), dtype=torch.long)
        parameters = tensor([r['parameters'] for r in rows])
        valid, done = states['present'], tensor([r['done'] for r in rows])
        rewards = tensor([r['reward'] for r in rows]).unsqueeze(-1)
        target = self.td_targets(following, rewards, done)
        actual = self.q(states['obs'], parameters).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        critic_loss = masked_mean(F.huber_loss(actual, target, reduction='none'), valid)
        self.optimize('critic', critic_loss, self.q.parameters())
        self.q.requires_grad_(False)
        try:
            predicted = self.q(states['obs'], self.parameters_net(states['obs']))
            actor_loss = -masked_mean(predicted.gather(-1, actions.unsqueeze(-1)).squeeze(-1), valid)
            self.optimize('actor', actor_loss, self.parameters_net.parameters())
        finally:
            self.q.requires_grad_(True)
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            self.target_parameters.load_state_dict(self.parameters_net.state_dict())
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

    def td_targets(self, following, rewards, done):
        with torch.no_grad():
            selection = self.q(following['obs'], self.parameters_net(following['obs']))
            chosen = selection.masked_fill(~action_masks(following['lanes']), -torch.inf).argmax(-1)
            valuation = self.target_q(following['obs'], self.target_parameters(following['obs']))
            return rewards+self.config.gamma*(1-done)*valuation.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
