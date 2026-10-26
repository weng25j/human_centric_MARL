"""Numerical, dynamic-agent, control and optional real SUMO regression checks."""
from collections import defaultdict, deque
import builtins
from dataclasses import replace
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from algorithms.traffic_shepherd_v2 import (OBS_DIM, GLOBAL_DIM, ShepherdConfig,
    TrafficShepherdV2, generalized_advantages)
from environments.traffic_shepherd_v2 import (RewardConfig, TrafficShepherdV2Env,
    reward_components, weighted_reward)
from environments.shepherd_v2_scenarios import prepare_scenario
from training.train_traffic_shepherd_v2 import experiment_settings


def state(speed=10.0, position=(0.0, 0.0), **extra):
    result = {'speed': speed, 'acceleration': 0.0, 'position': np.asarray(position),
        'lane_index': 0, 'remaining': 100.0, 'road': 'E0', 'route': 'route_main',
        'max_speed': 25.0, 'loss': 0.0, 'leader': None, 'gap': 150.0,
        'follower': None, 'back_gap': 150.0, 'ttc': float('inf'),
        'speed_variance': 0.0, 'uncertainty': 0.0}
    result.update(extra)
    return result


def environment(states, budget=5):
    env = TrafficShepherdV2Env.__new__(TrafficShepherdV2Env)
    env.states, env.max_controlled, env.action_mode = states, budget, 'structured'
    env.bottleneck_x, env.previous_policies = 100.0, {}
    env.history, env.dt, env.steps = defaultdict(lambda: deque(maxlen=5)), 0.1, 0
    env.reward_config, env.controlled = RewardConfig(), set()
    env.arrivals, env.collisions = [], set()
    env.connection = MagicMock()
    return env


class PPONumericsTests(unittest.TestCase):
    def test_gae_tracks_identity_terminals_and_truncation(self):
        def row(vehicle, step, reward, value=0, next_value=0, terminal=False):
            return dict(vehicle=vehicle, step=step, reward=reward, value=value,
                        next_value=next_value, terminal=terminal)
        rows = [row('a', 0, 1, next_value=2), row('b', 0, 10, terminal=True),
                row('a', 1, 3, value=2, next_value=4), row('b', 1, 100, terminal=True)]
        advantages, returns = generalized_advantages(rows, .5, .5)
        # a1: 3+.5*4-2 = 3; a0: 1+.5*2+.5*.5*3 = 2.75
        np.testing.assert_allclose(advantages, [2.75, 10, 3, 100])
        np.testing.assert_allclose(returns, [2.75, 10, 5, 100])
        gap = [row('a', 0, 1), row('a', 2, 100)]
        self.assertEqual(generalized_advantages(gap, .9, .9)[0][0], 1)

    def test_updates_change_actor_and_critic_and_checkpoint_restores(self):
        cfg = ShepherdConfig(hidden_size=16, batch_size=4, epochs=2)
        agent = TrafficShepherdV2(cfg, seed=7)
        rows = []
        rng = np.random.default_rng(8)
        for step in range(8):
            obs, central = {'cav': rng.normal(size=OBS_DIM).astype(np.float32)}, rng.normal(size=GLOBAL_DIM).astype(np.float32)
            actions, logs, _ = agent.act(obs)
            value = agent.values(obs, central)['cav']
            rows.append(dict(vehicle='cav', step=step, observation=obs['cav'],
                critic_state=np.concatenate([obs['cav'], central]), action=actions['cav'],
                log_prob=logs['cav'], value=value, reward=float(step), next_value=.3, terminal=step == 7))
        actor = [p.detach().clone() for p in agent.actor.parameters()]
        critic = [p.detach().clone() for p in agent.critic.parameters()]
        losses = agent.update(rows)
        self.assertTrue(np.isfinite(losses['actor_loss']+losses['critic_loss']))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor, agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critic, agent.critic.parameters())))
        with tempfile.TemporaryDirectory() as td:
            agent.save(Path(td)/'checkpoint-2.pt', 2, {'ablation': 'full'})
            expected = agent.act(obs, deterministic=True)[0]
            sampled = agent.act(obs)[0]
            loaded, payload = TrafficShepherdV2.load(td, resume=True)
            self.assertEqual(loaded.act(obs)[0], sampled)
            self.assertEqual(loaded.act(obs, deterministic=True)[0], expected)
            self.assertEqual(payload['episode'], 2)
            self.assertEqual(loaded.updates, agent.updates)
            self.assertEqual(loaded.actor_optimizer.state_dict()['state'].keys(),
                             agent.actor_optimizer.state_dict()['state'].keys())
            torch.save({'method': 'shepherd_v1'}, Path(td)/'wrong.pt')
            with self.assertRaises(ValueError):
                TrafficShepherdV2.load(Path(td)/'wrong.pt')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                TrafficShepherdV2.load(td)

    def test_actor_has_no_centralized_input(self):
        agent = TrafficShepherdV2(ShepherdConfig(hidden_size=8))
        obs = {'cav': np.zeros(OBS_DIM, np.float32)}
        before = agent.act(obs, deterministic=True)
        self.assertNotEqual(agent.values(obs, np.zeros(GLOBAL_DIM)), agent.values(obs, np.ones(GLOBAL_DIM)))
        after = agent.act(obs, deterministic=True)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[2]['cav'], after[2]['cav'])
        self.assertEqual(agent.act({}), ({}, {}, {}))
        self.assertIsNone(agent.update([]))


class ObservationControlTests(unittest.TestCase):
    def test_locality_fingerprints_and_budget(self):
        states = {'cav0': state(), 'cav1': state(position=(30, 0)),
                  'human0': state(8, (10, 0)), 'human_far': state(1, (1000, 0))}
        env = environment(states, budget=1)
        env.previous_policies = {'cav1': np.asarray([0, 1, 0, 0, 0])}
        obs = env.observe()
        self.assertEqual(list(obs), ['cav0'])
        self.assertEqual(obs['cav0'].shape, (42,))
        np.testing.assert_array_equal(obs['cav0'][27:32], [0, 1, 0, 0, 0])
        states['human_far']['speed'] = 29
        np.testing.assert_array_equal(env.observe()['cav0'], obs['cav0'])
        self.assertEqual(len(env.history), 0)  # observe never advances histories
        self.assertEqual(env.global_state().shape, (12,))
        env.max_controlled = 0
        self.assertEqual(len(env.observe()), 2)

    def test_all_five_roles_primitive_ablation_and_native_release(self):
        env = environment({'cav0': state(), 'human': state(7, (20, 0))})
        commands = [env.command('cav0', k) for k in range(5)]
        self.assertIsNone(commands[0])
        self.assertLess(commands[1], 0)
        self.assertLess(commands[2], 0)
        self.assertGreater(commands[4], 0)
        self.assertTrue(all(-3 <= a <= 1.5 for a in commands[1:]))
        with self.assertRaises(ValueError):
            env.command('cav0', 5)
        env.action_mode = 'primitive'
        self.assertEqual([env.command('cav0', k) for k in range(5)], [-1.5, -.75, 0, .75, 1.5])
        env.action_mode, env.controlled = 'structured', {'cav0', 'cav_old'}
        env.states['cav_old'] = state()
        env.max_controlled = 1
        env._snapshot = lambda: {'cav0': state()}
        env.connection.simulation.getCollidingVehiclesIDList.return_value = []
        env.connection.simulation.getArrivedIDList.return_value = []
        env.step({'cav0': 0})
        env.connection.vehicle.setSpeedMode.assert_any_call('cav_old', 31)
        env.connection.vehicle.setSpeed.assert_any_call('cav_old', -1)
        env.connection.vehicle.setSpeed.assert_any_call('cav0', -1)
        env.connection.vehicle.setAcceleration.assert_not_called()

    def test_removed_collision_gets_reward_even_without_next_observation(self):
        env = environment({'cav0': state()})
        env.connection.simulation.getCollidingVehiclesIDList.return_value = ['cav0', 'human']
        env.connection.simulation.getArrivedIDList.return_value = []
        env._snapshot = lambda: {}
        obs, rewards = env.step({'cav0': 0})
        self.assertEqual(obs, {})
        self.assertEqual(env.collisions, {'cav0', 'human'})
        self.assertLessEqual(rewards['cav0'], -10)
        self.assertGreater(env.last_parts['safety'], 0)


class RewardScenarioTests(unittest.TestCase):
    def test_route_preparation_isolates_legacy_import_rng_and_source_files(self):
        import xml.etree.ElementTree as ET
        root = Path(__file__).resolve().parents[1]
        source = {p: p.read_bytes() for p in (root/'SUMO_network').rglob('*.xml')}
        original_import = builtins.__import__
        def legacy_import(name, *args, **kwargs):
            module = original_import(name, *args, **kwargs)
            if name == 'training.train_traffic_shepherd':
                torch.manual_seed(123)
                np.random.seed(123)
                random.seed(123)
            return module
        torch.manual_seed(9)
        np.random.seed(9)
        random.seed(9)
        torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        with tempfile.TemporaryDirectory() as td, patch('builtins.__import__', side_effect=legacy_import):
            network, routes = prepare_scenario(td, 800, .2, mixture='aggressive')
            self.assertEqual(network.read_bytes(), (root/'SUMO_network/test.net.xml').read_bytes())
            mixture = ET.parse(routes).getroot().find('vTypeDistribution')
            self.assertEqual([float(v.get('probability')) for v in mixture], [.1, .3, .6])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng)
        self.assertEqual(np.random.get_state()[0], numpy_rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
        self.assertEqual(random.getstate(), python_rng)
        for path, data in source.items():
            self.assertEqual(path.read_bytes(), data)

    def test_all_reward_terms_and_ablations(self):
        before = {'human': state(loss=1), 'cav0': state()}
        after = {'human': state(acceleration=-4, loss=2, ttc=1, speed_variance=3),
                 'cav0': state(acceleration=-1)}
        parts = reward_components(before, after, [], {'human'}, {'cav0': -1})
        self.assertTrue(all(parts[k] > 0 for k in ('safety', 'stability', 'fairness', 'control')))
        full = weighted_reward(parts, RewardConfig())
        for key in ('stability', 'fairness', 'control'):
            self.assertGreater(weighted_reward(parts, replace(RewardConfig(), **{key: 0})), full)
        self.assertEqual(reward_components({}, {}, [], set(), {})['fairness'], 0)

    def test_resume_cannot_silently_change_ablation(self):
        args = MagicMock(scenario=None, hdv_mixture=None, max_controlled_cavs=None, ablation=None)
        full = experiment_settings(args)
        self.assertEqual(full['max_controlled_cavs'], 5)
        args.ablation = 'no_fairness'
        self.assertEqual(experiment_settings(args)['reward']['fairness'], 0)
        with self.assertRaises(ValueError):
            experiment_settings(args, full)


@unittest.skipUnless(os.environ.get('SHEPHERD_V2_SUMO_TESTS') == '1', 'Set SHEPHERD_V2_SUMO_TESTS=1 for real SUMO checks')
class RealSUMOTests(unittest.TestCase):
    def test_both_scenarios_execute_every_role_with_real_traci(self):
        for scenario in ('ramp', 'lane_drop'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as td:
                network, routes = prepare_scenario(td, 800, .2, scenario)
                env = TrafficShepherdV2Env(network, routes, seed=42, scenario=scenario)
                count = np.zeros(5, int)
                try:
                    for step in range(120):
                        obs = env.observe()
                        self.assertTrue(all(x.shape == (42,) and np.isfinite(x).all() for x in obs.values()))
                        role = step % 5
                        _, rewards = env.step(dict.fromkeys(obs, role))
                        count[role] += len(obs)
                        self.assertTrue(all(np.isfinite(r) for r in rewards.values()))
                        self.assertTrue(all(n.startswith('cav') for n in env.last_commands))
                    self.assertTrue((count > 0).all(), count)
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
