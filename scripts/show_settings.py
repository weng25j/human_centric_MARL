"""Print experiment defaults and only the project's non-secret environment settings."""
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = {
    'MARL_VENV': str(ROOT / '.venv'), 'MPLBACKEND': 'Agg',
    'MPLCONFIGDIR': str(ROOT / '.cache/matplotlib'),
    'XDG_CACHE_HOME': str(ROOT / '.cache'), 'OMP_NUM_THREADS': '2',
    'PYTHONDONTWRITEBYTECODE': '1', 'SUMO_HOME': '(optional; simulator on PATH)',
    'MARL_RUN_DIR': '(new runs/<timestamp> directory)', 'MARL_SEED': '42',
    'MARL_EVAL_DEMANDS': '800,1700', 'MARL_EVAL_PRS': '0.05,0.10,0.20',
    'MARL_ZMERGE_EPISODES': '5',
}


def get_settings():
    configs = {}
    for path in sorted((ROOT/'training').glob('train_*.py')):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'CONFIG' for target in node.targets
            ):
                configs[path.stem] = ast.literal_eval(node.value)
    nash_tree = ast.parse((ROOT/'algorithms/nash_double_q.py').read_text())
    nash_class = next(node for node in nash_tree.body if isinstance(node, ast.ClassDef) and node.name == 'NashConfig')
    nash_config = {node.target.id: ast.literal_eval(node.value) for node in nash_class.body if isinstance(node, ast.AnnAssign)}
    v2_tree = ast.parse((ROOT/'algorithms/traffic_shepherd_v2.py').read_text(encoding='utf-8'))
    v2_class = next(node for node in v2_tree.body if isinstance(node, ast.ClassDef) and node.name == 'ShepherdConfig')
    v2_config = {node.target.id: ast.literal_eval(node.value) for node in v2_class.body if isinstance(node, ast.AnnAssign)}
    versions = {}
    for name in ['torch', 'numpy', 'gymnasium', 'pettingzoo', 'sumolib', 'traci',
                 'eclipse-sumo', 'pandas', 'openpyxl', 'nashpy']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version.split()[0], 'python_executable': sys.executable,
            'package_versions': versions, 'training_configs': configs,
            'environment': {name: {'set': name in os.environ, 'value': os.environ.get(name),
                                   'script_default': default} for name, default in VARIABLES.items()},
            'evaluation': {'mappo_steps': 4000, 'mappo_seeds': [0,25,50,75,100,125,150,175,200,
                           325,350,375,400,425,450,475,500,525,550,575],
                           'z_merge_steps': 1000, 'z_merge_episodes': 5, 'step_seconds': 0.1},
            'effective_nash_double_q_learner': nash_config,
            'effective_traffic_shepherd_v2_learner': v2_config,
            'effective_z_merge_learner': {'replay_capacity': 100000, 'gamma': 0.995,
                'q_lr': 0.0001, 'parameter_lr': 0.0001, 'target_update_steps': 35000,
                'hidden_layers': [256,512,512,128]},
            'notes': ['MAPPO learner max_steps defaults to 1000 despite CONFIG rollout_n_steps=4000.',
                      'Z-Merge CONFIG MEMORY_CAPACITY=50000 is not wired to its 100000-capacity buffer.',
                      'No API keys, credentials or external services are required.']}


if __name__ == '__main__':
    print(json.dumps(get_settings(), indent=2))
