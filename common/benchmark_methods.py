"""Dependency-free comparison registry, also usable by command dry runs."""
LEARNED = ('shepherd_v2_common', 'flat_mappo_common', 'z_merge_common', 'trust_marl_common',
           'ma2c_common', 'qmix_common', 'maac_common', 'nash_double_q_common')
FIXED = ('all_hdv_common', 'rule_cav_common', 'selfish_cav_common')
METHODS = LEARNED + FIXED
LABELS = {'shepherd_v2_common': 'TRAFFIC SHEPHERD 2.0', 'flat_mappo_common': 'FLAT MAPPO',
          'z_merge_common': 'Z-MERGE', 'trust_marl_common': 'TRUST-MARL', 'ma2c_common': 'MA2C',
          'qmix_common': 'QMIX', 'maac_common': 'MAAC', 'nash_double_q_common': 'NASH DOUBLE-Q',
          'all_hdv_common': 'ALL-HDV', 'rule_cav_common': 'RULE-CAV', 'selfish_cav_common': 'SELFISH-CAV'}
OBS_DIMS = dict.fromkeys(METHODS, 42)
OBS_DIMS.update(flat_mappo_common=6, trust_marl_common=45, ma2c_common=60, qmix_common=47)
CONFIG_OVERRIDES = {
    'shepherd_v2_common': {'hidden': 256, 'batch_size': 256},
    'flat_mappo_common': {'hidden': 256, 'batch_size': 256},
    'ma2c_common': {'rollout_steps': 20},
    'z_merge_common': {'hidden': 256, 'actor_lr': 0.0001, 'critic_lr': 0.0001, 'target_interval': 35000},
    'nash_double_q_common': {'hidden': 128},
}
