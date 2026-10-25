"""Mathematical and lifecycle regression checks for the Nash comparison."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from algorithms.nash_double_q import NashConfig, NashDoubleQAgent, equilibrium, exploitability
from environments.nash_merge import NashMergeEnv, encode_game, interaction_groups
from common.merging_metrics import MergingMetrics


class FixedPayoffs(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.payoffs = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.payoffs.expand(len(states), -1, -1, -1)


class EquilibriumTests(unittest.TestCase):
    def test_coordination_equilibrium_changes_when_other_players_payoffs_change(self):
        game = np.array([[[3, 0], [0, 2]], [[3, 0], [0, 2]]], float)
        p, q = equilibrium(game)
        np.testing.assert_array_equal(p, [1, 0])
        np.testing.assert_array_equal(q, [1, 0])
        game[1] = [[0, 4], [0, 5]]
        p, q = equilibrium(game)
        np.testing.assert_array_equal(p, [0, 1])
        np.testing.assert_array_equal(q, [0, 1])
        self.assertLess(exploitability(game, p, q), 1e-6)

    def test_matching_pennies_requires_mixed_equilibrium(self):
        game = np.array([[[1,-1],[-1,1]],[[-1,1],[1,-1]]], float)
        p, q = equilibrium(game)
        np.testing.assert_allclose(p, [.5,.5])
        np.testing.assert_allclose(q, [.5,.5])
        self.assertLess(exploitability(game, p, q), 1e-6)

    def test_masks_and_singleton_game(self):
        game = np.zeros((2,3,3)); game[0,2,1] = 10
        strategies = equilibrium(game, np.array([[True, True, False],[False,True,False]]))
        self.assertEqual(strategies[0,2], 0)
        np.testing.assert_array_equal(strategies[1], [0,1,0])
        with self.assertRaises(ValueError):
            equilibrium(game, np.zeros((2,3), bool))


class LearningTests(unittest.TestCase):
    def test_double_q_uses_online_equilibrium_and_terminal_mask(self):
        cfg = NashConfig(batch_size=2, learning_starts=2, gamma=.5, reward_scale=1)
        agent = NashDoubleQAgent(cfg)
        online, target = np.zeros((2,3,3)), np.zeros((2,3,3))
        online[:,0,0] = 2
        target[:,0,0] = [4,6]
        target[:,2,2] = 99
        agent.online, agent.target = FixedPayoffs(online), FixedPayoffs(target)
        agent.optimizer = torch.optim.SGD(agent.online.parameters(), lr=0)
        state = np.zeros(14); state[-2:] = 1
        for rewards, done in [([1,2],False),([7,9],True)]:
            agent.remember(state, [1,1], rewards, state, done, np.ones((2,3),bool))
        self.assertAlmostEqual(agent.learn(), 5.5)
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))

    def test_real_update_sync_and_checkpoint(self):
        cfg = NashConfig(batch_size=4, learning_starts=4, target_interval=1)
        agent = NashDoubleQAgent(cfg, seed=7)
        rng = np.random.default_rng(9)
        for i in range(8):
            agent.remember(rng.normal(size=14), [i%3,(i+1)%3], [1,2],
                           rng.normal(size=14), i%2 == 0, np.ones((2,3),bool))
        before = [p.detach().clone() for p in agent.online.parameters()]
        self.assertTrue(np.isfinite(agent.learn()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.online.parameters())))
        for a,b in zip(agent.online.parameters(),agent.target.parameters()):
            torch.testing.assert_close(a,b)
        with tempfile.TemporaryDirectory() as td:
            file = Path(td)/'checkpoint-2.pt';agent.save(file,2,.5)
            loaded, record = NashDoubleQAgent.load(file, resume=True)
            self.assertEqual(record['episode'],2)
            self.assertEqual(loaded.updates,1)
            for a,b in zip(agent.online.parameters(),loaded.online.parameters()):
                torch.testing.assert_close(a,b)
            loaded, _ = NashDoubleQAgent.load(td)
            self.assertEqual(loaded.config, cfg)

    def test_groups_and_dummy_observation_do_not_drop_vehicles(self):
        snapshot = {name: {'position': np.array([x,0]), 'observation': [0]*6,
                           'speed': 0, 'max_speed': 30} for name,x in [('a',0),('b',3),('c',500)]}
        groups = interaction_groups(snapshot)
        self.assertEqual(groups, [('a','b'),('c',None)])
        state, masks = encode_game(groups[1],snapshot)
        self.assertEqual(state.shape,(14,))
        np.testing.assert_array_equal(state[-2:],[1,0])
        np.testing.assert_array_equal(masks[1],[False,True,False])
        next_state, _ = encode_game(('a','b'),{'a':snapshot['a']})
        np.testing.assert_array_equal(next_state[-2:],[1,0])


class SimulationBookkeepingTests(unittest.TestCase):
    def test_snapshot_keeps_the_five_cav_comparison_budget(self):
        env = object.__new__(NashMergeEnv)
        env.max_controlled = 5
        env.connection = MagicMock()
        v = env.connection.vehicle
        v.getIDList.return_value = [f'cav_{i}' for i in range(8)]
        v.getSpeed.return_value = 10.0
        v.getAllowedSpeed.return_value = 30.0
        v.getLeader.return_value = None
        v.getFollower.return_value = ('', -1)
        v.getLanePosition.return_value = 50.0
        v.getPosition.return_value = (50.0, 0.0)
        self.assertEqual(len(env.snapshot()), 5)

    def test_previous_control_is_released_and_removed_collision_reward_retained(self):
        env = object.__new__(NashMergeEnv)
        env.connection, env.controlled, env.step_length = MagicMock(), {'cav_old'}, 0.1
        env.connection.vehicle.getIDList.return_value = ['cav_old', 'cav_new']
        env.connection.simulation.getCollidingVehiclesIDList.return_value = ['cav_new']
        with patch.object(env, 'snapshot', return_value={}):
            _, rewards = env.step({'cav_new': 2})
        self.assertEqual(rewards['cav_new'], -10)
        env.connection.vehicle.setSpeed.assert_called_once_with('cav_old', -1)
        env.connection.vehicle.setAcceleration.assert_called_once_with('cav_new', 1.5, duration=0.1)

    def test_removed_collision_is_in_the_evaluation_denominator(self):
        connection = MagicMock()
        connection.simulation.getArrivedIDList.return_value = []
        connection.vehicle.getIDList.return_value = []
        connection.lane.getIDList.return_value = []
        collector = MergingMetrics(connection)
        collector.step({'cav_removed'})
        self.assertEqual(collector.result()['collision_rate'], 100)


if __name__ == '__main__':
    unittest.main()
