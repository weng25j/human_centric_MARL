"""Regression checks for the paper's Double PDQN update and runner lifecycle."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch import nn

from algorithms.z_merge import ZMergeAgent, ZMergeQNetwork
from training import train_z_merge as runner
from evaluation import evaluate_z_merge as evaluator


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class ConstantParameters(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.values = nn.Parameter(torch.full((5,), value))

    def forward(self, states):
        return self.values.expand(len(states), -1)


class FixedRankingCritic(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states, parameters):
        return self.values.expand(len(states), -1) + parameters


def deterministic_agent():
    agent = ZMergeAgent(22)
    agent.q_net = FixedRankingCritic([9, 1, 2, 3, 0])
    agent.q_target = FixedRankingCritic([5, 99, 3, 2, 1])
    agent.param_net = ConstantParameters(0.25)
    agent.param_target = ConstantParameters(0.75)
    # Zero learning rates let the expected losses be computed exactly.
    agent.q_optimizer = torch.optim.SGD(agent.q_net.parameters(), lr=0)
    agent.param_optimizer = torch.optim.SGD(agent.param_net.parameters(), lr=0)
    agent.memory.push(np.zeros(22), 2, np.zeros(5), 1, np.ones(22), False)
    agent.memory.push(np.zeros(22), 3, np.zeros(5), 4, np.ones(22), True)
    return agent


class DoublePDQNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_each_q_uses_only_its_own_effective_parameter(self):
        critic = ZMergeQNetwork(22, 5, 5)
        states = torch.randn(3, 22)
        parameters = torch.randn(3, 5, requires_grad=True)
        original = critic(states, parameters)
        changed = parameters.detach().clone()
        changed[:, [0, 1, 3, 4]] += 20
        torch.testing.assert_close(original[:, 2], critic(states, changed)[:, 2])
        original[:, 2].sum().backward()
        self.assertGreater(parameters.grad[:, 2].abs().sum().item(), 0)
        torch.testing.assert_close(parameters.grad[:, [0, 1, 3, 4]], torch.zeros(3, 4))
        # Lane change and maintain-state arguments are unused by the environment.
        arbitrary = torch.randn(3, 5) * 50
        torch.testing.assert_close(original[:, [0, 1, 4]], critic(states, arbitrary)[:, [0, 1, 4]])

    def test_double_target_and_terminal_mask_have_expected_losses(self):
        agent = deterministic_agent()
        q_loss, actor_loss = agent.train_step(batch_size=2, gamma=0.5)
        # Online argmax is 0, target argmax is 1. Use target action 0 (5+.75),
        # not target max (99+.75), and do not bootstrap the terminal transition.
        self.assertAlmostEqual(q_loss, 0.9375)
        self.assertAlmostEqual(actor_loss, -2.75)
        self.assertTrue(all(p.grad is None for p in agent.q_net.parameters()))
        self.assertTrue(all(p.requires_grad for p in agent.q_net.parameters()))
        torch.testing.assert_close(agent.param_net.values.grad, torch.tensor([0., 0., -.5, -.5, 0.]))
        self.assertTrue(all(p.grad is None for p in agent.q_target.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.param_target.parameters()))

    def test_hard_sync_occurs_at_existing_35000_step_interval(self):
        agent = deterministic_agent()
        agent.gradient_steps = 34998
        agent.train_step(2)
        self.assertEqual(agent.q_target.values[1].item(), 99)
        agent.train_step(2)
        torch.testing.assert_close(agent.q_net.values, agent.q_target.values)
        torch.testing.assert_close(agent.param_net.values, agent.param_target.values)

    def test_real_update_changes_both_online_networks(self):
        agent = ZMergeAgent(22)
        rng = np.random.default_rng(7)
        for i in range(8):
            agent.memory.push(rng.normal(size=22), 2+i % 2, rng.uniform(-1, 1, 5),
                              float(i), rng.normal(size=22), bool(i % 2))
        before = [p.detach().clone() for p in agent.q_net.parameters()]
        actor_before = [p.detach().clone() for p in agent.param_net.parameters()]
        target_before = [p.detach().clone() for p in agent.q_target.parameters()]
        losses = agent.train_step(8)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.q_net.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.param_net.parameters())))
        for a, b in zip(target_before, agent.q_target.parameters()):
            torch.testing.assert_close(a, b)
        self.assertEqual(agent.gradient_steps, 1)

    def test_checkpoint_roundtrip_explicit_and_legacy_prefix(self):
        agent = ZMergeAgent(22)
        state = np.arange(22, dtype=np.float32)
        expected = agent.act(state, epsilon=0)
        with tempfile.TemporaryDirectory() as td, working_directory(td):
            prefix = Path('results_zmerge/run/models/final')
            agent.save(prefix)
            for load_prefix in [prefix, 'run/models/final', Path(td)/prefix]:
                loaded = ZMergeAgent(22)
                loaded.load(load_prefix)
                actual = loaded.act(state, epsilon=0)
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                for online, target in zip(loaded.q_net.parameters(), loaded.q_target.parameters()):
                    torch.testing.assert_close(online, target)
            with self.assertRaises(FileNotFoundError):
                agent.load('missing')


class TrainingLifecycleTests(unittest.TestCase):
    def test_wrapper_reset_reuses_existing_sumo_connection(self):
        with patch.object(runner, 'traci') as traci, patch.object(runner, 'ZMergeEnv') as env:
            expected = {'cav': np.zeros(22)}
            env.return_value._get_observations.return_value = expected
            wrapper = runner.TraCIEnvWrapper('network', 'route', False)
            self.assertIs(wrapper.reset(), expected)
            traci.start.assert_not_called()
            env.return_value.reset.assert_not_called()

    def run_short_trainer(self, *, failure=False):
        agent = ZMergeAgent(22)
        wrapper = MagicMock()
        wrapper.reset.return_value = {'cav': np.ones(22)}
        wrapper.step.return_value = ({}, {'cav': -20.0}, {'cav': True})
        if failure:
            wrapper.step.side_effect = RuntimeError('algorithm failure')
        config = dict(runner.CONFIG, MAX_EPISODES=1, ROLL_OUT_N_STEPS=1)
        with tempfile.TemporaryDirectory() as td, working_directory(td), \
                patch.object(runner, 'traci') as traci, \
                patch.object(runner, 'TraCIEnvWrapper', return_value=wrapper), \
                patch.object(runner, 'ZMergeAgent', return_value=agent), \
                patch.object(runner, 'CONFIG', config), patch.object(runner, 'plt'):
            args = argparse.Namespace(seed=42, base_dir=str(Path(td)/'new/base'),
                                      gui=False, model_dir='', start_episode=0)
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'algorithm failure'):
                    runner.train(args)
            else:
                runner.train(args)
                self.assertEqual(len(agent.memory), 1)
                transition = agent.memory.buffer[0]
                self.assertEqual(transition[3], -1.0)  # unchanged reward scale 20
                self.assertTrue(transition[-1])
                np.testing.assert_array_equal(transition[-2], np.zeros(22))
            traci.start.assert_called_once()
            traci.close.assert_called_once()
        return agent

    def test_terminal_transitions_are_retained(self):
        self.run_short_trainer()

    def test_training_failures_are_not_reported_as_success(self):
        self.run_short_trainer(failure=True)

    def test_evaluate_option_uses_the_evaluation_program(self):
        args = argparse.Namespace(option='evaluate', model_dir='run/model', gui=True)
        with patch.object(runner, 'parse_args', return_value=args), \
                patch.object(runner, 'train') as train, \
                patch.object(evaluator, 'main') as evaluate:
            runner.main()
            train.assert_not_called()
            evaluate.assert_called_once_with(['--model-prefix', 'run/model', '--gui'])

    def test_evaluation_closes_connection_on_failure(self):
        with patch.object(evaluator, 'ZMergeAgent'), \
                patch.object(evaluator, 'ZMergeEnv') as env, \
                patch.object(evaluator, 'traci') as traci:
            env.return_value.reset.side_effect = RuntimeError('simulation failure')
            traci.isLoaded.return_value = True
            with self.assertRaisesRegex(RuntimeError, 'simulation failure'):
                evaluator.main(['--model-prefix', 'model'])
            traci.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
