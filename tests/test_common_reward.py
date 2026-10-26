"""Common reward identity, actuator validation and dynamic vehicle bookkeeping."""
from dataclasses import asdict
import unittest
from unittest.mock import MagicMock

import numpy as np

from common.benchmark import METHODS, reward_contract, shared_reward
from environments.common_reward_merge import CommonRewardMergeEnv, Control, Observer
from environments.traffic_shepherd_v2 import RewardConfig, reward_components, weighted_reward


def sample(speed=10, x=0):
    return {'speed': speed, 'acceleration': 0., 'position': np.asarray([x, 0.]),
            'lane_index': 0, 'remaining': 100., 'road': 'E0', 'route': 'route_main',
            'max_speed': 30., 'loss': 1., 'leader': None, 'follower': None,
            'gap': 150., 'back_gap': 150., 'ttc': float('inf'), 'speed_variance': 0., 'uncertainty': 0.}


class RewardContractTests(unittest.TestCase):
    def test_scalar_is_mean_of_v2_rewards_including_removed_collisions(self):
        before = {'cav0': sample(), 'human': sample()}
        after = {'human': sample(5)}
        commands = {'cav0': -1., 'cav1': None}
        parts = reward_components(before, after, [], {'cav0'}, commands)
        v2_rewards = [weighted_reward(parts, RewardConfig())-10, weighted_reward(parts, RewardConfig())]
        self.assertAlmostEqual(shared_reward(parts, {'cav0'}, commands), np.mean(v2_rewards))
        self.assertEqual(shared_reward(parts, {'cav0'}, {}), weighted_reward(parts, RewardConfig()))

    def test_contract_is_method_independent_and_records_exact_weights(self):
        contract = reward_contract()
        self.assertEqual(contract['weights'], asdict(RewardConfig()))
        self.assertEqual(contract['reward_scale'], 20)
        self.assertEqual(len(contract['reward_fingerprint']), 64)
        parts = dict(efficiency=1, safety=.2, stability=.3, fairness=.4, control=.5)
        scores = [shared_reward(parts, set(), {'cav': 1}) for _ in METHODS]
        self.assertEqual(len(set(scores)), 1)

    def test_commands_reject_invalid_data_and_keep_native_action_distinct(self):
        env = CommonRewardMergeEnv.__new__(CommonRewardMergeEnv)
        env.states, env.dt = {'cav': sample()}, .1
        self.assertIsNone(env.command('cav', Control()))
        self.assertEqual(env.command('cav', Control(lane_delta=1)), 0)
        self.assertEqual(env.command('cav', Control(target_speed=30)), 1.5)
        for control in [Control(float('nan')), Control(8), Control(lane_delta=2)]:
            with self.assertRaises(ValueError):
                env.command('cav', control)

    def test_stable_slots_histories_and_replacements(self):
        env = CommonRewardMergeEnv.__new__(CommonRewardMergeEnv)
        env.states = {'cav0': sample(), 'cav1': sample(x=20)}
        env.max_controlled, env.bottleneck_x = 5, 100
        env.steps, env.previous_policies = 0, {}
        env.arrivals, env.collisions = [], set()
        env.connection = MagicMock()
        env.connection.edge.getLaneNumber.return_value = 2
        observer = Observer('ma2c_common')
        first = observer.capture(env)
        self.assertEqual(first.histories.shape, (5, 5, 60))
        with self.assertRaises(RuntimeError):
            observer.capture(env)
        env.steps += 1
        env.states = {'cav1': sample(x=20), 'cav2': sample(x=50)}
        second = observer.capture(env)
        self.assertEqual(first.ids[1], second.ids[1])
        self.assertEqual(second.ids[0], 'cav2')
        self.assertTrue((second.histories[0, :-1] == 0).all())
        self.assertEqual(len(observer.histories['cav1']), 2)


if __name__ == '__main__':
    unittest.main()
