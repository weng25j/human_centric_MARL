"""Versioned common-reward contract and method registry for fair SUMO comparisons."""
from dataclasses import asdict
import hashlib
import inspect
import json

from environments.traffic_shepherd_v2 import RewardConfig, reward_components, weighted_reward

PROTOCOL = 'shepherd_common_reward_v1'
REWARD_SCALE = 20.0
LEARNED = ('shepherd_v2_common', 'flat_mappo_common', 'z_merge_common', 'trust_marl_common',
           'ma2c_common', 'qmix_common', 'maac_common', 'nash_double_q_common')
FIXED = ('all_hdv_common', 'rule_cav_common', 'selfish_cav_common')
METHODS = LEARNED + FIXED
LABELS = {'shepherd_v2_common': 'TRAFFIC SHEPHERD 2.0', 'flat_mappo_common': 'FLAT MAPPO',
          'z_merge_common': 'Z-MERGE', 'trust_marl_common': 'TRUST-MARL', 'ma2c_common': 'MA2C',
          'qmix_common': 'QMIX', 'maac_common': 'MAAC', 'nash_double_q_common': 'NASH DOUBLE-Q',
          'all_hdv_common': 'ALL-HDV', 'rule_cav_common': 'RULE-CAV', 'selfish_cav_common': 'SELFISH-CAV'}
OBS_DIMS = dict.fromkeys(METHODS, 42)
OBS_DIMS.update(flat_mappo_common=6, trust_marl_common=45, ma2c_common=60)


def shared_reward(parts, collisions, commands):
    """Exactly the mean of Shepherd 2.0's per-acted-CAV rewards.

    Every learning agent receives THIS scalar. QMIX consumes it once, not summed
    over padded slots. No trust reward blending or MA2C reward redistribution.
    """
    collision_cost = 10.0*len(set(commands) & set(collisions))/max(1, len(commands))
    return float(weighted_reward(parts, RewardConfig())-collision_cost)


def reward_contract():
    source = '\n'.join(inspect.getsource(f) for f in (reward_components, weighted_reward, shared_reward))
    settings = {'protocol': PROTOCOL, 'weights': asdict(RewardConfig()), 'reward_scale': REWARD_SCALE,
                'collision_aggregation': 'mean_over_acted_cavs', 'step_seconds': 0.1}
    settings['reward_fingerprint'] = hashlib.sha256((source+json.dumps(settings, sort_keys=True)).encode()).hexdigest()
    return settings
