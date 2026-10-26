"""Numerical paper-mechanism tests, checkpoint isolation, and real TraCI checks."""
from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn

from algorithms.benchmark_base import BenchmarkConfig, BenchmarkLearner, pack, tensor
from algorithms.common_reward_agents import make_agent
from algorithms.maac import AttentionCritic
from algorithms.qmix import MonotonicMixer
from algorithms.trust_game import rectangular_equilibrium
from common.benchmark import LEARNED, OBS_DIMS
from environments.common_reward_merge import CommonRewardMergeEnv, Frame, Observer, controls_for
from environments.shepherd_v2_scenarios import prepare_scenario


def frame(method, empty=False):
    dim = OBS_DIMS[method]
    generator = np.random.default_rng(8)
    observations = generator.normal(size=(5, dim)).astype(np.float32)
    history = generator.normal(size=(5, 5, dim)).astype(np.float32)
    states = {n: {'speed': 10., 'max_speed': 25., 'acceleration': 0., 'lane_index': 0,
                  'route': 'route_main', 'position': np.asarray([i*20., 0.])} for i, n in enumerate(('cav0', 'cav1'))}
    names = (None,)*5 if empty else ('cav0', 'cav1', None, None, None)
    return Frame(names, observations, history, np.asarray([n is not None for n in names]),
                 np.zeros(12, np.float32), np.asarray([[0, 0], [20, 0], [0, 0], [0, 0], [0, 0]], np.float32),
                 np.tile([True, True, False], (5, 1)), np.zeros((5, 3), np.float32), states)


class PaperMechanismTests(unittest.TestCase):
    def test_qmix_monotonicity_inactive_mask_and_global_greedy_consistency(self):
        torch.manual_seed(2)
        mixer = MonotonicMixer()
        utilities = torch.randn(1, 5, requires_grad=True)
        central, present = torch.randn(1, 12), tensor([[1, 1, 0, 0, 0]])
        value = mixer(utilities, central, present)
        gradient = torch.autograd.grad(value.sum(), utilities)[0]
        self.assertTrue((gradient >= -1e-8).all())
        torch.testing.assert_close(gradient[:, 2:], torch.zeros(1, 3))
        table = torch.randn(2, 5)
        greedy = table.max(-1).values
        independent = mixer(torch.cat([greedy, torch.zeros(3)]).unsqueeze(0), central, present)
        candidates = [mixer(tensor([[table[0, i], table[1, j], 0, 0, 0]]), central, present).item()
                      for i in range(5) for j in range(5)]
        self.assertAlmostEqual(independent.item(), max(candidates), places=5)

    def test_qmix_bootstrap_and_empty_next_team_have_closed_form_targets(self):
        agent = make_agent('qmix_common', BenchmarkConfig(hidden=16, gamma=.5))
        class Utility(nn.Module):
            def forward(self, histories):
                return torch.arange(1., 6.).expand(*histories.shape[:2], 5)
        class Mixer(nn.Module):
            def forward(self, values, state, mask):
                return (values*mask).sum(-1)
        agent.target_utility, agent.target_mixer = Utility(), Mixer()
        following = {'history': torch.zeros(2, 5, 5, 47), 'global': torch.zeros(2, 12),
                     'present': tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])}
        torch.testing.assert_close(agent.td_target(following, tensor([1, 1])), tensor([6, 1]))

    def test_maac_attention_excludes_self_and_absent_agents_but_uses_other_actions(self):
        torch.manual_seed(5)
        critic = AttentionCritic(hidden=16)
        observations = torch.randn(1, 5, 42)
        choices = nn.functional.one_hot(torch.zeros(1, 5, dtype=torch.long), 5).float()
        present = tensor([[1, 1, 0, 0, 0]])
        before, attention = critic(observations, choices, present, True)
        changed = choices.clone()
        changed[0, 0] = tensor([0, 1, 0, 0, 0])
        after = critic(observations, changed, present)
        torch.testing.assert_close(before[:, 0], after[:, 0])
        self.assertFalse(torch.allclose(before[:, 1], after[:, 1]))
        self.assertTrue((attention[..., 2:] == 0).all())
        self.assertTrue((attention[:, 0, :, :, 0] == 0).all())
        values, attention = critic(observations, choices, tensor([[1, 0, 0, 0, 0]]), True)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((attention[:, 0] == 0).all())

    def test_hybrid_sac_density_jacobian_bounds_and_lane_mask(self):
        agent = make_agent('trust_marl_common', BenchmarkConfig(hidden=16))
        with torch.no_grad():
            agent.actor.output.weight.zero_()
            agent.actor.output.bias.zero_()
        f = frame('trust_marl_common')
        f.lane_masks[:, 1:] = False
        acc, lane, log_cont, probabilities, _ = agent.actor.sample(tensor(f.histories), tensor(f.game_prior), tensor(f.lane_masks), True)
        torch.testing.assert_close(acc, torch.full((5,), -.75))
        self.assertTrue((lane == 0).all())
        torch.testing.assert_close(probabilities[:, 0], torch.ones(5))
        self.assertAlmostEqual(log_cont[0].item(), -.5*math.log(2*math.pi)-math.log(2.25), places=5)
        for _ in range(20):
            decision = agent.act(f)
            self.assertTrue(((decision.parameters >= -3)&(decision.parameters <= 1.5)).all())
            self.assertTrue((decision.actions == 0).all())

    def test_trust_game_mixed_equilibrium_is_not_independent_greedy(self):
        a = np.asarray([[1, -1, 1, -1], [-1, 1, -1, 1]], float)
        row, column = rectangular_equilibrium([a, -a])
        np.testing.assert_allclose(row, [.5, .5])
        self.assertAlmostEqual(column[[0, 2]].sum(), .5)
        self.assertLess((a@column).max()-row@a@column, 1e-6)
        self.assertLess((row@-a).max()-row@-a@column, 1e-6)

    def test_z_merge_target_masks_illegal_online_actions_then_evaluates_target(self):
        agent = make_agent('z_merge_common', BenchmarkConfig(gamma=.5))
        class ConstantQ(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer('values', tensor(values))
            def forward(self, states, parameters):
                return self.values.expand(*states.shape[:-1], -1)
        agent.q, agent.target_q = ConstantQ([10, 8, 7, 6, 5]), ConstantQ([99, 2, 3, 4, 5])
        following = {'obs': torch.zeros(1, 5, 42), 'lanes': tensor([[[1, 0, 1]]*5])}
        targets = agent.td_targets(following, torch.ones(1, 1), tensor([[0, 1, 0, 1, 0]]))
        torch.testing.assert_close(targets, tensor([[2, 1, 2, 1, 2]]))

    def test_ma2c_uses_local_temporal_values_and_fingerprints(self):
        agent = make_agent('ma2c_common', BenchmarkConfig(hidden=16))
        f = frame('ma2c_common')
        before = agent.act(f, evaluation=True)
        f.global_state[:] = 9
        after = agent.act(f, evaluation=True)
        np.testing.assert_array_equal(before.values, after.values)
        f.histories[:, -1, 27:42] += 1
        self.assertFalse(np.allclose(agent.act(f, evaluation=True).values, before.values))


class LearnerLifecycleTests(unittest.TestCase):
    def test_every_method_updates_real_parameters_and_round_trips(self):
        config = BenchmarkConfig(hidden=16, batch_size=2, learning_starts=2, train_every=1, rollout_steps=2, target_interval=2)
        for method in LEARNED:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as td:
                agent = make_agent(method, config, seed=7)
                f = frame(method)
                before = {k: v.clone() for k, v in agent.state_dict().items()}
                for step in range(4):
                    decision = agent.act(f, epsilon=.1)
                    loss = agent.observe(f, decision, 20., f, step, final=step == 3)
                self.assertGreater(agent.updates, 0)
                self.assertTrue(all(np.isfinite(v) for v in loss.values()))
                self.assertTrue(any(not torch.equal(value, agent.state_dict()[key]) for key, value in before.items()))
                self.assertTrue(all(p.grad is None for name, p in agent.named_parameters() if name.startswith('target')))
                agent.save(Path(td)/'checkpoint-1.pt', 1, {'scenario': 'ramp'})
                loaded, payload = BenchmarkLearner.load(td, method, resume=True)
                self.assertEqual(payload['method'], method)
                for key, value in agent.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key])
                self.assertEqual(loaded.updates, agent.updates)
                self.assertEqual(set(loaded.optimizers), set(agent.optimizers))
                self.assertEqual(len(loaded.memory), 0)
                np.random.seed(9)
                first = agent.act(f, evaluation=True)
                np.random.seed(9)
                second = loaded.act(f, evaluation=True)
                np.testing.assert_array_equal(first.actions, second.actions)
                if first.parameters is not None:
                    np.testing.assert_allclose(first.parameters, second.parameters)
                with self.assertRaises(ValueError):
                    BenchmarkLearner.load(td, 'wrong_method')
                damaged = torch.load(Path(td)/'checkpoint-1.pt', weights_only=False)
                damaged['contract']['reward_scale'] = 1
                torch.save(damaged, Path(td)/'bad.pt')
                with self.assertRaisesRegex(ValueError, 'reward contract'):
                    BenchmarkLearner.load(Path(td)/'bad.pt', method)

    def test_no_cav_steps_do_not_create_padded_training_samples(self):
        for method in LEARNED:
            agent = make_agent(method, BenchmarkConfig(hidden=16, batch_size=2, learning_starts=2, train_every=1))
            f = frame(method, empty=True)
            for step in range(2):
                decision = agent.act(f)
                self.assertIsNone(agent.observe(f, decision, 1, f, step, final=True))
            self.assertEqual(agent.transitions, 0)
            self.assertEqual(agent.updates, 0)


@unittest.skipUnless(os.environ.get('COMMON_REWARD_SUMO_TESTS') == '1', 'Set COMMON_REWARD_SUMO_TESTS=1 for live paper comparisons')
class RealPaperSUMOTests(unittest.TestCase):
    def test_five_paper_methods_act_learn_and_release_real_connections(self):
        for method in ('z_merge_common', 'trust_marl_common', 'ma2c_common', 'qmix_common', 'maac_common'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as td:
                network, routes = prepare_scenario(td, 800, .2)
                env = CommonRewardMergeEnv(network, routes)
                observer = Observer(method)
                agent = make_agent(method, BenchmarkConfig(hidden=16, batch_size=4, learning_starts=4, train_every=4, rollout_steps=20))
                try:
                    f = observer.capture(env)
                    for step in range(80):
                        d = agent.act(f, epsilon=.1)
                        controls = controls_for(method, env, f, d.actions, d.parameters)
                        observer.record_actions(f, d.actions)
                        _, rewards = env.step(controls, {n: d.probabilities[i] for i, n in enumerate(f.ids) if n})
                        self.assertLessEqual(len(set(rewards.values())), 1)
                        following = observer.capture(env)
                        agent.observe(f, d, env.team_reward, following, step, final=step == 79)
                        self.assertTrue(np.isfinite(following.observations).all())
                        f = following
                    self.assertGreater(agent.updates, 0)
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
