"""Common-reward CTDE PPO and recurrent, fingerprint-conditioned MA2C."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from algorithms.benchmark_base import BenchmarkLearner, Decision, tensor
from algorithms.traffic_shepherd_v2 import generalized_advantages, network
from common.benchmark import REWARD_SCALE


class TemporalNetwork(nn.Module):
    def __init__(self, inputs, outputs, hidden):
        super().__init__()
        self.embedding = nn.Linear(inputs, hidden)
        self.memory = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, outputs)

    def forward(self, histories):
        shape = histories.shape[:-2]
        flat = histories.reshape(-1, histories.shape[-2], histories.shape[-1])
        encoded, _ = self.memory(torch.tanh(self.embedding(flat)))
        return self.output(encoded[:, -1]).reshape(*shape, -1)


class OnPolicyComparison(BenchmarkLearner):
    def __init__(self, method, config=None, seed=42):
        super().__init__(method, config, seed)
        self.recurrent = method == 'ma2c_common'
        if self.recurrent:
            self.actor = TemporalNetwork(self.obs_dim, 5, self.config.hidden)
            self.critic = TemporalNetwork(self.obs_dim, 1, self.config.hidden)
        else:
            # Preserve Shepherd 2.0's 256-wide tanh PPO architecture/settings.
            self.actor = network(self.obs_dim, 5, self.config.hidden)
            self.critic = network(self.obs_dim+12, 1, self.config.hidden)
        self.optimizers['actor'] = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.optimizers['critic'] = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.rollout = []

    def _inputs(self, frame):
        actor = frame.histories if self.recurrent else frame.observations
        critic = frame.histories if self.recurrent else np.concatenate(
            [frame.observations, np.tile(frame.global_state, (len(frame.ids), 1))], axis=-1)
        return actor, critic

    def act(self, frame, epsilon=0.0, evaluation=False):
        actor, critic = self._inputs(frame)
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(tensor(actor)))
            action = distribution.probs.argmax(-1) if evaluation else distribution.sample()
            log = distribution.log_prob(action)
            value = self.critic(tensor(critic)).squeeze(-1)
        return Decision(action.numpy(), distribution.probs.numpy(), log_probs=log.numpy(), values=value.numpy())

    def observe(self, frame, decision, reward, following, step, final=False):
        self.environment_steps += 1
        actor, critic = self._inputs(frame)
        _, next_critic = self._inputs(following)
        with torch.no_grad():
            next_values = self.critic(tensor(next_critic)).flatten().numpy()
        for i, name in enumerate(frame.ids):
            if name is None:
                continue
            terminal = following.ids[i] != name
            self.rollout.append({'vehicle': name, 'step': step, 'observation': actor[i].copy(),
                'critic_state': critic[i].copy(), 'action': int(decision.actions[i]),
                'value': float(decision.values[i]), 'log_prob': float(decision.log_probs[i]),
                'reward': float(reward)/REWARD_SCALE, 'terminal': terminal,
                'next_value': float(next_values[i]) if not terminal else 0.0})
            self.transitions += 1
        interval = self.config.rollout_steps
        if final or (step+1) % interval == 0:
            return self.learn()
        return None

    def learn(self):
        if not self.rollout:
            return None
        rows, self.rollout = self.rollout, []
        # lambda=1 is the n-step A2C return in MA2C, not PPO/GAE smoothing.
        advantage, returns = generalized_advantages(rows, self.config.gamma, 1.0 if self.recurrent else .95)
        obs = tensor([r['observation'] for r in rows])
        state = tensor([r['critic_state'] for r in rows])
        actions = torch.tensor([r['action'] for r in rows], dtype=torch.long)
        old_log = tensor([r['log_prob'] for r in rows])
        if not self.recurrent and len(rows) > 1 and advantage.std() > 1e-8:
            advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        advantage, returns = tensor(advantage), tensor(returns)
        losses = []
        epochs, batch_size = (1, len(rows)) if self.recurrent else (4, self.config.batch_size)
        for _ in range(epochs):
            order = torch.arange(len(rows)) if self.recurrent else torch.randperm(len(rows))
            for start in range(0, len(rows), batch_size):
                batch = order[start:start+batch_size]
                distribution = Categorical(logits=self.actor(obs[batch]))
                log = distribution.log_prob(actions[batch])
                if self.recurrent:
                    actor_loss = -(log*advantage[batch]).mean()-.01*distribution.entropy().mean()
                else:
                    ratio = (log-old_log[batch]).exp()
                    actor_loss = -torch.minimum(ratio*advantage[batch], ratio.clamp(.8, 1.2)*advantage[batch]).mean()-.01*distribution.entropy().mean()
                critic_loss = .5*(self.critic(state[batch]).flatten()-returns[batch]).square().mean()
                self.optimize('actor', actor_loss, self.actor.parameters())
                self.optimize('critic', critic_loss, self.critic.parameters())
                self.updates += 1
                losses.append([actor_loss.item(), critic_loss.item()])
        return dict(zip(('actor_loss', 'critic_loss'), np.mean(losses, axis=0).tolist()))
