"""Traffic Shepherd 2.0: shared local PPO actor and centralized per-CAV critic."""
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

OBS_DIM = 42
GLOBAL_DIM = 12
ACTION_NAMES = ('non_intervention', 'gap_creation', 'flow_stabilization',
                'conservative_following', 'release')


@dataclass(frozen=True)
class ShepherdConfig:
    hidden_size: int = 256
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    batch_size: int = 256
    rollout_steps: int = 256
    reward_scale: float = 20.0


def network(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))


def generalized_advantages(rows, gamma, lam):
    """GAE follows vehicle identity, never another CAV's slot or next episode.

    Each row contains its own next value. Leaving the controlled set is terminal;
    a rollout/time-limit boundary bootstraps but does not join a later episode.
    """
    advantages = np.zeros(len(rows), np.float32)
    following = {}
    for index in range(len(rows)-1, -1, -1):
        row = rows[index]
        alive = 1.0-float(row['terminal'])
        delta = row['reward']+gamma*alive*row['next_value']-row['value']
        next_step, next_adv = following.get(row['vehicle'], (-1, 0.0))
        continuation = next_adv if next_step == row['step']+1 else 0.0
        advantages[index] = delta+gamma*lam*alive*continuation
        following[row['vehicle']] = (row['step'], advantages[index])
    returns = advantages+np.asarray([row['value'] for row in rows], np.float32)
    return advantages, returns


class TrafficShepherdV2:
    def __init__(self, config=None, seed=42):
        self.config, self.seed = config or ShepherdConfig(), seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.actor = network(OBS_DIM, len(ACTION_NAMES), self.config.hidden_size)
        self.critic = network(OBS_DIM+GLOBAL_DIM, 1, self.config.hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.updates = self.environment_steps = 0

    def act(self, observations, deterministic=False):
        """Execution takes LOCAL observations only; no centralized state argument."""
        if not observations:
            return {}, {}, {}
        names = list(observations)
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(torch.as_tensor(
                np.stack([observations[n] for n in names]), dtype=torch.float32)))
            actions = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(actions)
        return (dict(zip(names, actions.tolist())), dict(zip(names, log_probs.tolist())),
                {n: p for n, p in zip(names, distribution.probs.numpy())})

    def values(self, observations, global_state):
        if not observations:
            return {}
        names = list(observations)
        inputs = np.stack([np.concatenate([observations[n], global_state]) for n in names])
        with torch.no_grad():
            values = self.critic(torch.as_tensor(inputs, dtype=torch.float32)).flatten().tolist()
        return dict(zip(names, values))

    def update(self, rows):
        if not rows:
            return None
        cfg = self.config
        advantages, returns = generalized_advantages(rows, cfg.gamma, cfg.gae_lambda)
        # Keep the unnormalized advantage when the batch has one sample/zero variance.
        if len(rows) > 1 and advantages.std() > 1e-8:
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
        obs = torch.tensor(np.stack([r['observation'] for r in rows]), dtype=torch.float32)
        state = torch.tensor(np.stack([r['critic_state'] for r in rows]), dtype=torch.float32)
        actions = torch.tensor([r['action'] for r in rows], dtype=torch.long)
        old_log = torch.tensor([r['log_prob'] for r in rows], dtype=torch.float32)
        advantage, target = torch.tensor(advantages), torch.tensor(returns)
        losses = []
        for _ in range(cfg.epochs):
            order = torch.randperm(len(rows))
            for start in range(0, len(rows), cfg.batch_size):
                batch = order[start:start+cfg.batch_size]
                distribution = Categorical(logits=self.actor(obs[batch]))
                ratio = (distribution.log_prob(actions[batch])-old_log[batch]).exp()
                surrogate = torch.minimum(ratio*advantage[batch],
                    ratio.clamp(1-cfg.clip_ratio, 1+cfg.clip_ratio)*advantage[batch])
                actor_loss = -surrogate.mean()-cfg.entropy_coef*distribution.entropy().mean()
                critic_loss = 0.5*(self.critic(state[batch]).flatten()-target[batch]).square().mean()
                if not torch.isfinite(actor_loss+critic_loss):
                    raise FloatingPointError('Non-finite Shepherd 2.0 PPO loss')
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()
                losses.append((actor_loss.item(), critic_loss.item()))
                self.updates += 1
        return {'actor_loss': float(np.mean(losses, axis=0)[0]),
                'critic_loss': float(np.mean(losses, axis=0)[1]), 'samples': len(rows)}

    def save(self, path, episode, experiment):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'method': 'traffic_shepherd_v2', 'version': 1, 'config': asdict(self.config),
                   'seed': self.seed, 'episode': episode, 'experiment': experiment,
                   'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                   'actor_optimizer': self.actor_optimizer.state_dict(),
                   'critic_optimizer': self.critic_optimizer.state_dict(),
                   'updates': self.updates, 'environment_steps': self.environment_steps,
                   'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                   'python_rng': random.getstate()}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, location, resume=False):
        path = Path(location).expanduser().resolve()
        if path.is_dir():
            candidates = [p for p in path.glob('checkpoint-*.pt') if p.stem[11:].isdigit()]
            if not candidates:
                raise FileNotFoundError(f'No numbered checkpoint in {path}')
            path = max(candidates, key=lambda p: int(p.stem[11:]))
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}')
        # Optimizer/RNG checkpoints use pickle: load only your own trusted files.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('method') != 'traffic_shepherd_v2' or payload.get('version') != 1:
            raise ValueError('Expected a Traffic Shepherd 2.0 checkpoint (version 1)')
        agent = cls(ShepherdConfig(**payload['config']), payload['seed'])
        agent.actor.load_state_dict(payload['actor'])
        agent.critic.load_state_dict(payload['critic'])
        agent.updates, agent.environment_steps = payload['updates'], payload['environment_steps']
        if resume:
            agent.actor_optimizer.load_state_dict(payload['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(payload['critic_optimizer'])
            torch.set_rng_state(payload['torch_rng'])
            np.random.set_state(payload['numpy_rng'])
            random.setstate(payload['python_rng'])
        return agent, payload
