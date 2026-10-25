"""Train the game-theoretic Nash Double-Q merging adaptation using SUMO/TraCI."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from algorithms.nash_double_q import NashConfig, NashDoubleQAgent
from environments.nash_merge import NashMergeEnv, encode_game, interaction_groups, prepare_routes

CONFIG = {'MAX_EPISODES': 3000, 'MAX_STEPS': 1000, 'PAIR_RADIUS': 100.0,
          'epsilon_start': 0.99, 'epsilon_min': 0.05, 'epsilon_decay': 0.995,
          'checkpoint_interval': 50}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path('results_nash_double_q'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=CONFIG['MAX_EPISODES'], help='Total target episode count')
    parser.add_argument('--max-steps', type=int, default=CONFIG['MAX_STEPS'])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-controlled-cavs', type=int, default=5, help='Match the existing five-CAV control budget; 0 controls all')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args(argv)
    if args.max_controlled_cavs < 0:
        parser.error('--max-controlled-cavs must be non-negative')
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('Episode and step counts must be positive')
    base = args.base_dir.expanduser().resolve()
    models = base/'models'
    if not args.resume and models.exists() and any(models.glob('checkpoint-*.pt')):
        parser.error('Existing checkpoints: use --resume or a new --base-dir')
    base.mkdir(parents=True, exist_ok=True)
    if args.resume:
        agent, saved = NashDoubleQAgent.load(args.resume, resume=True)
        if args.seed != saved['seed']:
            parser.error(f'Resume with --seed {saved["seed"]} to preserve the training seed')
        first, epsilon = saved['episode'], saved['epsilon']
    else:
        agent = NashDoubleQAgent(seed=args.seed)
        first, epsilon = 0, CONFIG['epsilon_start']
    if first >= args.episodes:
        parser.error('--episodes must exceed the loaded episode count')
    metadata = {'method': 'nash_double_q_adaptation', 'arguments': vars(args),
                'training': CONFIG, 'learner': asdict(agent.config),
                'control': {'accelerations': [-1.5, 0.0, 1.5], 'step_seconds': 0.1}}
    (base/'training_settings.json').write_text(json.dumps(metadata, default=str, indent=2))
    log = base/'training_metrics.jsonl'
    for episode in range(first, args.episodes):
        rng = random.Random(args.seed+episode)
        demand = 800 if episode < 1000 else rng.choice([800, 1200] if episode < 2000 else [800, 1200, 1700])
        pr = 0.20 if episode < 1000 else 0.10 if episode < 2000 else rng.choice([0.05, 0.10, 0.20])
        network, routes = prepare_routes(base/'workspace', demand, pr)
        env = NashMergeEnv(network, routes, args.seed+episode, args.gui, max_controlled=args.max_controlled_cavs)
        total_reward, losses, joint_games = 0.0, [], 0
        initial_updates = agent.updates
        try:
            states = env.snapshot()
            for step in range(args.max_steps):
                games = interaction_groups(states, CONFIG['PAIR_RADIUS'])
                actions, transitions = {}, []
                for game in games:
                    state, masks = encode_game(game, states)
                    selected, _ = agent.select(state, masks, epsilon)
                    transitions.append((game, state, selected))
                    joint_games += int(game[1] is not None)
                    actions.update({name: selected[role] for role, name in enumerate(game) if name})
                next_states, rewards = env.step(actions)
                next_games = set(interaction_groups(next_states, CONFIG['PAIR_RADIUS']))
                for game, state, selected in transitions:
                    next_state, next_masks = encode_game(game, next_states)
                    # Re-pairing is a boundary of this two-player game, even if
                    # a surviving vehicle starts a new game in the next step.
                    done = game not in next_games
                    payoff = [rewards.get(name, 0.0) for name in game]
                    agent.remember(state, selected, payoff, next_state, done, next_masks)
                total_reward += sum(rewards.values())
                agent.environment_steps += 1
                if agent.environment_steps % agent.config.train_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        if not np.isfinite(loss):
                            raise FloatingPointError('Non-finite learning loss')
                        losses.append(loss)
                states = next_states
        finally:
            env.close()
        epsilon = max(CONFIG['epsilon_min'], epsilon*CONFIG['epsilon_decay'])
        record = {'episode': episode+1, 'reward': total_reward, 'loss': float(np.mean(losses)) if losses else None,
                  'updates': agent.updates-initial_updates, 'joint_games': joint_games,
                  'replay_size': len(agent.memory), 'demand': demand, 'pr': pr, 'epsilon': epsilon}
        with log.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if (episode+1) % CONFIG['checkpoint_interval'] == 0 or episode+1 == args.episodes:
            agent.save(models/f'checkpoint-{episode+1}.pt', episode+1, epsilon)


if __name__ == '__main__':
    main()
