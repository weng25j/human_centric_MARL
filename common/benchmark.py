"""Versioned common-reward contract and method registry for fair SUMO comparisons."""
from dataclasses import asdict
import hashlib
import inspect
import json

from environments.traffic_shepherd_v2 import RewardConfig, reward_components, weighted_reward
from common.benchmark_methods import FIXED, LABELS, LEARNED, METHODS, OBS_DIMS

PROTOCOL = 'shepherd_common_reward_v1'
REWARD_SCALE = 20.0


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
