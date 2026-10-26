"""Train Traffic Shepherd 2.0 in isolated SUMO workspaces with on-policy CTDE PPO."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from algorithms.traffic_shepherd_v2 import TrafficShepherdV2
from environments.traffic_shepherd_v2 import RewardConfig, TrafficShepherdV2Env
from environments.shepherd_v2_scenarios import MIXTURES, prepare_scenario
from recording.live_sumo import VideoPlan, add_video_arguments, validate_video_arguments

CONFIG = {'MAX_EPISODES': 3000, 'MAX_STEPS': 1000, 'checkpoint_interval': 50}
ABLATIONS = ('full', 'no_fairness', 'no_stability', 'no_control_cost', 'primitive_actions')


def experiment_settings(args, saved=None):
    defaults = {'scenario': 'ramp', 'hdv_mixture': 'default', 'max_controlled_cavs': 5, 'ablation': 'full'}
    settings = {}
    for key, default in defaults.items():
        supplied = getattr(args, key)
        settings[key] = supplied if supplied is not None else (saved or {}).get(key, default)
        if saved and settings[key] != saved[key]:
            raise ValueError(f'Resume cannot change {key}: checkpoint uses {saved[key]}')
    ablation = settings['ablation']
    settings['reward'] = asdict(RewardConfig(
        fairness=0.0 if ablation == 'no_fairness' else 0.5,
        stability=0.0 if ablation == 'no_stability' else 5.0,
        control=0.0 if ablation == 'no_control_cost' else 0.1))
    settings['action_mode'] = 'primitive' if ablation == 'primitive_actions' else 'structured'
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path('results_shepherd_v2'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=CONFIG['MAX_EPISODES'], help='Total target episode count')
    parser.add_argument('--max-steps', type=int, default=CONFIG['MAX_STEPS'])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--scenario', choices=('ramp', 'lane_drop'))
    parser.add_argument('--hdv-mixture', choices=MIXTURES)
    parser.add_argument('--ablation', choices=ABLATIONS)
    parser.add_argument('--max-controlled-cavs', type=int, help='Default 5; 0 controls all CAVs')
    parser.add_argument('--gui', action='store_true')
    add_video_arguments(parser)
    args = parser.parse_args(argv)
    validate_video_arguments(parser, args)
    if args.episodes < 1 or args.max_steps < 1 or (args.max_controlled_cavs is not None and args.max_controlled_cavs < 0):
        parser.error('Episodes/steps must be positive and control budget non-negative')
    base = args.base_dir.expanduser().resolve()
    models = base/'models'
    if not args.resume and models.exists() and any(models.glob('checkpoint-*.pt')):
        parser.error('Existing checkpoints: use --resume or a new --base-dir')
    if args.resume:
        agent, saved = TrafficShepherdV2.load(args.resume, resume=True)
        if args.seed != saved['seed']:
            parser.error(f'Resume with --seed {saved["seed"]}')
        first, previous = saved['episode'], saved['experiment']
        if args.max_steps != previous['max_steps']:
            parser.error(f'Resume with --max-steps {previous["max_steps"]}')
    else:
        agent, first, previous = TrafficShepherdV2(seed=args.seed), 0, None
    try:
        experiment = experiment_settings(args, previous)
    except ValueError as error:
        parser.error(str(error))
    experiment['max_steps'] = args.max_steps
    if first >= args.episodes:
        parser.error('--episodes must exceed the checkpoint episode count')
    base.mkdir(parents=True, exist_ok=True)
    metadata = {'method': 'traffic_shepherd_v2', 'arguments': vars(args), 'experiment': experiment,
                'learner': asdict(agent.config), 'training': CONFIG, 'step_seconds': 0.1}
    (base/'training_settings.json').write_text(json.dumps(metadata, default=str, indent=2), encoding='utf-8')
    videos = VideoPlan(args, base/'recordings', 'traffic_shepherd_v2', 'training')
    for episode in range(first, args.episodes):
        rng = random.Random(args.seed+episode)
        demand = 800 if episode < 1000 else rng.choice([800, 1200] if episode < 2000 else [800, 1200, 1700])
        pr = 0.20 if episode < 1000 else 0.10 if episode < 2000 else rng.choice([0.05, 0.10, 0.20])
        network, routes = prepare_scenario(base/'workspace', demand, pr, experiment['scenario'], experiment['hdv_mixture'])
        recording = videos.next(f'training_episode{episode+1}_seed{args.seed+episode}', args.max_steps,
            episode=episode+1, seed=args.seed+episode, demand=demand, pr=pr,
            scenario=experiment['scenario'], hdv_mixture=experiment['hdv_mixture'])
        env = TrafficShepherdV2Env(network, routes, args.seed+episode, args.gui,
            max_controlled=experiment['max_controlled_cavs'], reward_config=RewardConfig(**experiment['reward']),
            action_mode=experiment['action_mode'], scenario=experiment['scenario'], recording=recording)
        rows, losses = [], []
        reward_sum, samples = 0.0, 0
        roles = np.zeros(5, int)
        part_sums = dict.fromkeys(env.last_parts, 0.0)
        initial_updates = agent.updates
        try:
            observations = env.observe()
            global_state = env.global_state()
            for step in range(args.max_steps):
                actions, log_probs, policies = agent.act(observations)
                values = agent.values(observations, global_state)
                next_observations, rewards = env.step(actions, policies)
                next_global = env.global_state()
                next_values = agent.values(next_observations, next_global)
                for name, action in actions.items():
                    rows.append({'vehicle': name, 'step': step, 'observation': observations[name].copy(),
                        'critic_state': np.concatenate([observations[name], global_state]), 'action': action,
                        'log_prob': log_probs[name], 'value': values[name],
                        'reward': rewards[name]/agent.config.reward_scale,
                        'terminal': name not in next_observations or name in env.collisions,
                        'next_value': next_values.get(name, 0.0)})
                    roles[action] += 1
                reward_sum += sum(rewards.values())
                samples += len(actions)
                for key in part_sums:
                    part_sums[key] += env.last_parts[key]
                agent.environment_steps += 1
                if (step+1) % agent.config.rollout_steps == 0 or step+1 == args.max_steps:
                    loss = agent.update(rows)
                    if loss:
                        losses.append(loss)
                    rows.clear()  # strictly on-policy: never replay an old rollout
                if recording:
                    recording.note_learning(agent.updates-initial_updates, samples)
                observations, global_state = next_observations, next_global
        finally:
            if recording:
                recording.metadata.update(learner_updates=agent.updates-initial_updates, transitions=samples)
            env.close()
        record = {'episode': episode+1, 'demand': demand, 'pr': pr, 'reward': reward_sum,
                  'samples': samples, 'updates': agent.updates-initial_updates, 'role_counts': roles.tolist(),
                  'reward_components_mean': {k: v/args.max_steps for k, v in part_sums.items()},
                  'actor_loss': float(np.mean([r['actor_loss'] for r in losses])) if losses else None,
                  'critic_loss': float(np.mean([r['critic_loss'] for r in losses])) if losses else None}
        with (base/'training_metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if (episode+1) % CONFIG['checkpoint_interval'] == 0 or episode+1 == args.episodes:
            agent.save(models/f'checkpoint-{episode+1}.pt', episode+1, experiment)


if __name__ == '__main__':
    main()
