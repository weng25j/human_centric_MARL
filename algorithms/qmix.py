"""QMIX: recurrent utilities and state-conditioned nonnegative mixing weights."""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algorithms.benchmark_base import BenchmarkLearner, Decision, tensor


class RecurrentUtility(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.input = nn.Linear(inputs, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 5)

    def forward(self, history):
        leading = history.shape[:-2]
        flattened = history.reshape(-1, history.shape[-2], history.shape[-1])
        values, _ = self.gru(F.relu(self.input(flattened)))
        return self.output(values[:, -1]).reshape(*leading, 5)


class MonotonicMixer(nn.Module):
    def __init__(self, state_dim=17, agents=5, hidden=32):
        super().__init__()
        self.agents, self.hidden = agents, hidden
        self.w1 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, agents*hidden))
        self.b1 = nn.Linear(state_dim, hidden)
        self.w2 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, hidden))
        self.bias = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, utilities, state, present):
        features = torch.cat([state, present], -1)
        w1 = self.w1(features).abs().reshape(-1, self.agents, self.hidden)
        w2 = self.w2(features).abs().reshape(-1, self.hidden, 1)
        hidden = F.elu(torch.bmm((utilities*present).unsqueeze(1), w1)+self.b1(features).unsqueeze(1))
        return (torch.bmm(hidden, w2).flatten()+self.bias(features).flatten())


class QMIX(BenchmarkLearner):
    def __init__(self, method='qmix_common', config=None, seed=42):
        super().__init__(method, config, seed)
        self.utility = RecurrentUtility(self.obs_dim, self.config.hidden)
        self.mixer = MonotonicMixer()
        self.target_utility, self.target_mixer = copy.deepcopy(self.utility), copy.deepcopy(self.mixer)
        self.target_utility.requires_grad_(False)
        self.target_mixer.requires_grad_(False)
        self.optimizers['critic'] = torch.optim.Adam(list(self.utility.parameters())+list(self.mixer.parameters()), lr=self.config.critic_lr)

    def act(self, frame, epsilon=0.0, evaluation=False):
        with torch.no_grad():
            greedy = self.utility(tensor(frame.histories)).argmax(-1).numpy()
        probabilities = np.full((5, 5), 0 if evaluation else epsilon/5, np.float32)
        probabilities[np.arange(5), greedy] += 1 if evaluation else 1-epsilon
        actions = greedy if evaluation else np.asarray([np.random.choice(5, p=p/p.sum()) for p in probabilities])
        return Decision(actions, probabilities)

    def td_target(self, following, rewards):
        with torch.no_grad():
            # Paper QMIX uses max of delayed utilities; monotonicity turns these
            # independent maxima into the global maximum of the target mixer.
            utilities = self.target_utility(following['history']).max(-1).values
            q = self.target_mixer(utilities, following['global'], following['present'])
            alive = following['present'].any(-1).float()
            return rewards+self.config.gamma*alive*q

    def learn(self):
        rows, states, following = self.batch()
        actions = torch.tensor(np.stack([r['actions'] for r in rows]), dtype=torch.long)
        utilities = self.utility(states['history']).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        actual = self.mixer(utilities, states['global'], states['present'])
        target = self.td_target(following, tensor([r['reward'] for r in rows]))
        loss = (actual-target).square().mean()
        self.optimize('critic', loss, list(self.utility.parameters())+list(self.mixer.parameters()))
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target_utility.load_state_dict(self.utility.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return {'critic_loss': loss.item()}
