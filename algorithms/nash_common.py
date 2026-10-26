"""Existing Nash Double-Q mechanism, retrained on the common scalar reward."""
import copy
import random

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.benchmark_base import BenchmarkLearner, Decision, tensor
from algorithms.nash_double_q import JointQNetwork, NashConfig, equilibrium
from common.benchmark import REWARD_SCALE


def groups(frame):
    remaining = {i for i, n in enumerate(frame.ids) if n is not None}
    pairs = []
    while remaining:
        first = min(remaining, key=lambda i: frame.ids[i])
        remaining.remove(first)
        nearest = sorted(remaining, key=lambda i: (np.linalg.norm(frame.positions[first]-frame.positions[i]), frame.ids[i]))
        second = nearest[0] if nearest and np.linalg.norm(frame.positions[first]-frame.positions[nearest[0]]) <= 100 else None
        if second is not None:
            remaining.remove(second)
        pairs.append((first, second))
    return pairs


def encode(frame, pair):
    values, presence, masks = [], [], np.zeros((2, 3), bool)
    for role, index in enumerate(pair):
        if index is None or frame.ids[index] is None:
            values.extend([0]*6)
            presence.append(0)
            masks[role, 1] = True
        else:
            o = frame.observations[index]
            values.extend(o[[0, 2, 6, 7, 10, 11]])
            presence.append(1)
            s = frame.states[frame.ids[index]]
            masks[role] = [s['speed'] > .1, True, s['speed'] < s['max_speed']]
    return np.asarray(values+presence, np.float32), masks


class NashCommon(BenchmarkLearner):
    def __init__(self, method='nash_double_q_common', config=None, seed=42):
        super().__init__(method, config, seed)
        self.online = JointQNetwork(NashConfig(hidden_size=self.config.hidden))
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizers['critic'] = torch.optim.Adam(self.online.parameters(), lr=self.config.actor_lr)
        self.pending = []

    def act(self, frame, epsilon=0.0, evaluation=False):
        actions, probabilities, self.pending = np.ones(5, int), np.zeros((5, 5), np.float32), []
        for pair in groups(frame):
            state, masks = encode(frame, pair)
            with torch.no_grad():
                payoffs = self.online(tensor(state).unsqueeze(0))[0].numpy()
            strategies = equilibrium(payoffs, masks)
            selected = []
            for role, index in enumerate(pair):
                p = strategies[role] if evaluation else (1-epsilon)*strategies[role]+epsilon*masks[role]/masks[role].sum()
                choice = np.random.choice(3, p=p/p.sum())
                selected.append(choice)
                if index is not None:
                    actions[index], probabilities[index, :3] = choice, p
            self.pending.append((tuple(frame.ids[i] if i is not None else None for i in pair), state, selected))
        return Decision(actions, probabilities)

    def observe(self, frame, decision, reward, following, step, final=False):
        self.environment_steps += 1
        next_pairs = {tuple(following.ids[i] if i is not None else None for i in pair) for pair in groups(following)}
        for names, state, actions in self.pending:
            pair = tuple(following.ids.index(n) if n is not None and n in following.ids else None for n in names)
            next_state, masks = encode(following, pair)
            self.memory.append((state, actions, float(reward)/REWARD_SCALE, next_state, names not in next_pairs, masks))
            self.transitions += sum(n is not None for n in names)
        if len(self.memory) >= max(self.config.learning_starts, self.config.batch_size) and self.environment_steps % self.config.train_every == 0:
            return self.learn()
        return None

    def learn(self):
        rows = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, following, done, masks = zip(*rows)
        state, next_state = tensor(states), tensor(following)
        with torch.no_grad():
            online, target = self.online(next_state).numpy(), self.target(next_state)
            strategies = tensor([equilibrium(game, mask) for game, mask in zip(online, masks)])
            joint = strategies[:, 0, :, None]*strategies[:, 1, None, :]
            expected = (target*joint.unsqueeze(1)).sum((-1, -2))
            returns = tensor(rewards).unsqueeze(-1)+self.config.gamma*(1-tensor(done).unsqueeze(-1))*expected
        predicted = self.online(state)
        choices = torch.tensor(actions, dtype=torch.long)
        actual = predicted[torch.arange(len(rows)), :, choices[:, 0], choices[:, 1]]
        valid = state[:, -2:]
        loss = (F.huber_loss(actual, returns, reduction='none')*valid).sum()/valid.sum().clamp_min(1)
        self.optimize('critic', loss, self.online.parameters())
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {'critic_loss': loss.item()}
