"""Explicit registry; algorithms share the experiment contract, not one optimizer."""
import numpy as np

from algorithms.benchmark_base import BenchmarkConfig, BenchmarkLearner, Decision
from common.benchmark import FIXED
from common.benchmark_methods import CONFIG_OVERRIDES


class FixedComparison(BenchmarkLearner):
    def act(self, frame, epsilon=0.0, evaluation=False):
        return Decision(np.zeros(5, int), np.tile([1., 0, 0, 0, 0], (5, 1)).astype(np.float32))


def make_agent(method, config=None, seed=42):
    if config is None:
        config = BenchmarkConfig(**CONFIG_OVERRIDES.get(method, {}))
    if method in FIXED:
        return FixedComparison(method, config, seed)
    if method in ('shepherd_v2_common', 'flat_mappo_common', 'ma2c_common'):
        from algorithms.common_on_policy import OnPolicyComparison
        return OnPolicyComparison(method, config, seed)
    if method == 'z_merge_common':
        from algorithms.z_merge_common import ZMergeCommon
        return ZMergeCommon(method, config, seed)
    if method == 'trust_marl_common':
        from algorithms.trust_marl import TrustMARL
        return TrustMARL(method, config, seed)
    if method == 'qmix_common':
        from algorithms.qmix import QMIX
        return QMIX(method, config, seed)
    if method == 'maac_common':
        from algorithms.maac import MAAC
        return MAAC(method, config, seed)
    if method == 'nash_double_q_common':
        from algorithms.nash_common import NashCommon
        return NashCommon(method, config, seed)
    raise ValueError(f'Unknown common-reward method: {method}')
