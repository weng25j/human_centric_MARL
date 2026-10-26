"""Shared lifecycle only: each paper learner supplies its own update equations."""
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from common.benchmark import OBS_DIMS, REWARD_SCALE, reward_contract


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden: int = 64
    gamma: float = 0.99
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 64
    capacity: int = 50000
    learning_starts: int = 256
    train_every: int = 4
    target_interval: int = 500
    tau: float = 0.01
    temperature: float = 0.1
    max_grad_norm: float = 0.5
    rollout_steps: int = 256


@dataclass
class Decision:
    actions: np.ndarray
    probabilities: np.ndarray
    parameters: np.ndarray = None
    log_probs: np.ndarray = None
    values: np.ndarray = None


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def mlp(inputs, outputs, hidden=64):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, outputs))


def masked_mean(values, mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def soft_update(source, target, tau):
    with torch.no_grad():
        for online, delayed in zip(source.parameters(), target.parameters()):
            delayed.lerp_(online, tau)


def pack(frame):
    return {'obs': frame.observations.copy(), 'history': frame.histories.copy(),
            'present': frame.present.copy(), 'global': frame.global_state.copy(),
            'lanes': frame.lane_masks.copy(), 'prior': frame.game_prior.copy()}


class BenchmarkLearner(nn.Module):
    def __init__(self, method, config=None, seed=42):
        super().__init__()
        self.method, self.config, self.seed = method, config or BenchmarkConfig(), seed
        self.obs_dim = OBS_DIMS[method]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.optimizers = {}
        self.memory = deque(maxlen=self.config.capacity)
        self.updates = self.environment_steps = self.transitions = 0

    def observe(self, frame, decision, reward, following, step, final=False):
        self.environment_steps += 1
        if frame.present.any():
            self.memory.append({'state': pack(frame), 'next': pack(following),
                'actions': decision.actions.copy(),
                'parameters': None if decision.parameters is None else decision.parameters.copy(),
                'reward': float(reward)/REWARD_SCALE,
                'done': np.asarray([n is None or n != following.ids[i] for i, n in enumerate(frame.ids)], np.float32)})
            self.transitions += int(frame.present.sum())
        if (len(self.memory) >= max(self.config.learning_starts, self.config.batch_size)
                and self.environment_steps % self.config.train_every == 0):
            return self.learn()
        return None

    def batch(self):
        rows = random.sample(self.memory, self.config.batch_size)
        states = {key: tensor([r['state'][key] for r in rows]) for key in rows[0]['state']}
        following = {key: tensor([r['next'][key] for r in rows]) for key in rows[0]['next']}
        return rows, states, following

    def optimize(self, name, loss, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite {self.method} {name} loss')
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        optimizer.step()

    def save(self, path, episode, experiment):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'method': self.method, 'format': 'common_reward_checkpoint_v1',
            'contract': reward_contract(), 'config': asdict(self.config), 'seed': self.seed,
            'episode': episode, 'experiment': experiment, 'networks': self.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'updates': self.updates, 'environment_steps': self.environment_steps, 'transitions': self.transitions,
            'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @staticmethod
    def load(location, method=None, resume=False):
        path = Path(location).expanduser().resolve()
        if path.is_dir():
            candidates = [p for p in path.glob('checkpoint-*.pt') if p.stem[11:].isdigit()]
            if not candidates:
                raise FileNotFoundError(f'No numbered checkpoint under {path}')
            path = max(candidates, key=lambda p: int(p.stem[11:]))
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format') != 'common_reward_checkpoint_v1':
            raise ValueError('Expected a common-reward benchmark checkpoint, not legacy weights')
        if method is not None and payload['method'] != method:
            raise ValueError(f'Wrong method: expected {method}, got {payload["method"]}')
        if payload['contract'] != reward_contract():
            raise ValueError('Checkpoint reward contract differs from the current benchmark')
        from algorithms.common_reward_agents import make_agent
        agent = make_agent(payload['method'], BenchmarkConfig(**payload['config']), payload['seed'])
        agent.load_state_dict(payload['networks'], strict=True)
        agent.updates, agent.environment_steps, agent.transitions = payload['updates'], payload['environment_steps'], payload['transitions']
        if resume:
            for name, optimizer in agent.optimizers.items():
                optimizer.load_state_dict(payload['optimizers'][name])
            torch.set_rng_state(payload['torch_rng'])
            np.random.set_state(payload['numpy_rng'])
            random.setstate(payload['python_rng'])
        return agent, payload
