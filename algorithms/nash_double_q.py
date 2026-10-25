"""Bimatrix Nash-equilibrium Double DQN for two-vehicle interaction games.

Independent implementation; benchmark adaptation, not the original paper's code.
"""
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import nashpy
import numpy as np
import torch
from torch import nn


@dataclass
class NashConfig:
    state_dim: int = 14
    action_dim: int = 3
    hidden_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 0.0003
    batch_size: int = 64
    replay_capacity: int = 50000
    learning_starts: int = 256
    train_every: int = 4
    target_interval: int = 500
    reward_scale: float = 20.0
    max_grad_norm: float = 0.5


def exploitability(payoffs, row, column):
    """Largest unilateral payoff improvement (zero at a Nash equilibrium)."""
    joint = np.outer(row, column)
    return max(float((payoffs[0] @ column).max() - (joint * payoffs[0]).sum()),
               float((row @ payoffs[1]).max() - (joint * payoffs[1]).sum()))


def equilibrium(payoffs, masks=None):
    """Solve a masked 3x3 bimatrix game, checking unilateral deviations.

    Choose a welfare-maximizing pure equilibrium when one exists. Otherwise use
    finite support enumeration, not an unverified independent/greedy fallback.
    """
    payoffs = np.asarray(payoffs, dtype=np.float64)
    if payoffs.ndim != 3 or payoffs.shape[0] != 2 or payoffs.shape[1] != payoffs.shape[2] or not np.isfinite(payoffs).all():
        raise ValueError('Expected two finite payoff matrices')
    masks = np.ones((2, payoffs.shape[-1]), bool) if masks is None else np.asarray(masks, bool)
    if masks.shape != (2, payoffs.shape[-1]) or not masks.any(axis=1).all():
        raise ValueError('Each player must have at least one legal action')
    indices = [np.flatnonzero(m) for m in masks]
    game = payoffs[:, indices[0]][:, :, indices[1]]
    scale = max(1.0, float(np.abs(game).max()))
    game = game / scale
    a, b = game
    pure = np.argwhere((a >= a.max(axis=0, keepdims=True)-1e-10)
                       & (b >= b.max(axis=1, keepdims=True)-1e-10))
    if len(pure):
        i, j = max(pure.tolist(), key=lambda ij: (a[tuple(ij)]+b[tuple(ij)], -ij[0], -ij[1]))
        row, column = np.eye(len(indices[0]))[i], np.eye(len(indices[1]))[j]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for row, column in nashpy.Game(a, b).support_enumeration():
                if (np.isfinite(row).all() and np.isfinite(column).all()
                        and np.min(row) >= -1e-9 and np.min(column) >= -1e-9
                        and abs(row.sum()-1) < 1e-6 and abs(column.sum()-1) < 1e-6
                        and exploitability(game, row, column) < 1e-6):
                    break
            else:
                raise RuntimeError('Could not verify a Nash equilibrium for the interaction game')
    strategies = np.zeros(masks.shape, dtype=np.float64)
    for player, probabilities in enumerate([row, column]):
        probabilities = np.maximum(probabilities, 0)
        strategies[player, indices[player]] = probabilities / probabilities.sum()
    return strategies


class JointQNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.actions = config.action_dim
        self.layers = nn.Sequential(nn.Linear(config.state_dim, config.hidden_size), nn.ReLU(),
                                    nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                                    nn.Linear(config.hidden_size, 2 * config.action_dim**2))

    def forward(self, states):
        return self.layers(states).reshape(-1, 2, self.actions, self.actions)


class NashDoubleQAgent:
    def __init__(self, config=None, seed=42):
        self.config = config or NashConfig()
        self.seed = seed
        torch.manual_seed(seed)
        self.online = JointQNetwork(self.config)
        self.target = JointQNetwork(self.config)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.memory = deque(maxlen=self.config.replay_capacity)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.environment_steps = 0

    @torch.no_grad()
    def select(self, state, masks, epsilon=0.0, deterministic=False):
        payoff = self.online(torch.as_tensor(state, dtype=torch.float32)[None])[0].numpy()
        strategies = equilibrium(payoff, masks)
        if self.rng.random() < epsilon:
            actions = [int(self.rng.choice(np.flatnonzero(mask))) for mask in masks]
        elif deterministic:
            # Largest-probability actions need not themselves form an equilibrium.
            # Sampling with a seeded RNG preserves the mixed equilibrium instead.
            if np.all(strategies.max(axis=1) >= 1-1e-9):
                actions = strategies.argmax(axis=1).tolist()
            else:
                actions = [int(self.rng.choice(self.config.action_dim, p=p)) for p in strategies]
        else:
            actions = [int(self.rng.choice(self.config.action_dim, p=p)) for p in strategies]
        return actions, strategies

    def remember(self, state, actions, rewards, next_state, done, next_masks):
        self.memory.append((np.array(state, np.float32), np.array(actions, np.int64),
                            np.array(rewards, np.float32), np.array(next_state, np.float32),
                            bool(done), np.array(next_masks, bool)))

    def learn(self):
        c = self.config
        if len(self.memory) < max(c.batch_size, c.learning_starts):
            return None
        batch = [self.memory[i] for i in self.rng.choice(len(self.memory), c.batch_size, replace=False)]
        states, actions, rewards, next_states, dones, masks = map(np.stack, zip(*batch))
        states, next_states = map(lambda x: torch.as_tensor(x, dtype=torch.float32), [states, next_states])
        with torch.no_grad():
            online_next = self.online(next_states).numpy()
            target_next = self.target(next_states)
            continuation = torch.zeros(c.batch_size, 2)
            for i in range(c.batch_size):
                if dones[i]:
                    continue
                row, column = equilibrium(online_next[i], masks[i])
                probabilities = torch.as_tensor(np.outer(row, column), dtype=torch.float32)
                continuation[i] = (target_next[i] * probabilities).sum(dim=(-1, -2)) * next_states[i, -2:]
            targets = torch.as_tensor(rewards) / c.reward_scale + c.gamma * continuation
        values = self.online(states)
        chosen = values[torch.arange(c.batch_size), :, actions[:, 0], actions[:, 1]]
        loss = nn.functional.smooth_l1_loss(chosen, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.max_grad_norm)
        self.optimizer.step()
        self.updates += 1
        if self.updates % c.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self, path, episode, epsilon):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'method': 'nash_double_q', 'version': 1, 'config': asdict(self.config),
                    'online': self.online.state_dict(), 'target': self.target.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'episode': episode, 'epsilon': epsilon,
                    'updates': self.updates, 'environment_steps': self.environment_steps, 'seed': self.seed,
                    'rng': self.rng.bit_generator.state}, path)

    @classmethod
    def load(cls, path, resume=False, seed=42):
        path = Path(path)
        if path.is_dir():
            files = [p for p in path.glob('checkpoint-*.pt') if p.stem[11:].isdigit()]
            if not files:
                raise FileNotFoundError(f'No Nash checkpoint in {path}')
            path = max(files, key=lambda p: int(p.stem[11:]))
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('method') != 'nash_double_q' or data.get('version') != 1:
            raise ValueError('Not a compatible Nash Double-Q checkpoint')
        agent = cls(NashConfig(**data['config']), data['seed'] if resume else seed)
        agent.online.load_state_dict(data['online'])
        agent.target.load_state_dict(data['target'])
        if resume:
            agent.optimizer.load_state_dict(data['optimizer'])
            agent.updates = data['updates']
            agent.environment_steps = data['environment_steps']
            agent.rng.bit_generator.state = data['rng']
        return agent, data
