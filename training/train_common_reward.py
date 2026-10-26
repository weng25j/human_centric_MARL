"""Train one paper comparison on the shared Traffic Shepherd SUMO reward."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from algorithms.benchmark_base import BenchmarkConfig, BenchmarkLearner
from algorithms.common_reward_agents import make_agent
from common.benchmark import LEARNED, reward_contract
from environments.common_reward_merge import CommonRewardMergeEnv, Observer, controls_for
from environments.shepherd_v2_scenarios import MIXTURES, prepare_scenario
from recording.live_sumo import VideoPlan, add_video_arguments, validate_video_arguments

CONFIG = {'MAX_EPISODES': 3000, 'MAX_STEPS': 1000, 'checkpoint_interval': 50}


def main(argv=None, method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=LEARNED, default=method, required=method is None)
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=CONFIG['MAX_EPISODES'])
    parser.add_argument('--max-steps', type=int, default=CONFIG['MAX_STEPS'])
    parser.add_argument('--scenario', choices=('ramp', 'lane_drop'), default='ramp')
    parser.add_argument('--hdv-mixture', choices=MIXTURES, default='default')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--audit-reward', action='store_true')
    parser.add_argument('--gui', action='store_true')
    add_video_arguments(parser)
    args = parser.parse_args(argv)
    validate_video_arguments(parser, args)
    if method is not None and args.method != method:
        parser.error(f'This entry point trains {method}')
    if args.episodes < 1 or args.max_steps < 1:
        parser.error('Episodes and max steps must be positive')
    base = (args.base_dir or Path('results_common')/args.method).expanduser().resolve()
    if not args.resume and (base/'training_settings.json').exists():
        parser.error('Existing run: use --resume or a new --base-dir')
    experiment = {'scenario': args.scenario, 'hdv_mixture': args.hdv_mixture,
                  'max_steps': args.max_steps, 'max_controlled_cavs': 5}
    if args.resume:
        agent, payload = BenchmarkLearner.load(args.resume, args.method, resume=True)
        if payload['seed'] != args.seed or payload['experiment'] != experiment:
            parser.error('Resume requires the checkpoint seed, scenario, HDV mixture and max-step horizon')
        first = payload['episode']
    else:
        agent, first = make_agent(args.method, seed=args.seed), 0
    if first >= args.episodes:
        parser.error('--episodes must exceed the loaded episode count')
    base.mkdir(parents=True, exist_ok=True)
    settings = {'method': args.method, **reward_contract(), 'arguments': vars(args),
                'experiment': experiment, 'learner': asdict(agent.config), 'training': CONFIG,
                'resume_replay': 'Replay is not stored; off-policy resumes refill replay before learning.'}
    (base/'training_settings.json').write_text(json.dumps(settings, default=str, indent=2), encoding='utf-8')
    audit = (base/'reward_audit.jsonl').open('a', encoding='utf-8') if args.audit_reward else None
    videos = VideoPlan(args, base/'recordings', args.method, 'training')
    try:
        for episode in range(first, args.episodes):
            rng = random.Random(args.seed+episode)
            demand = 800 if episode < 1000 else rng.choice([800, 1200] if episode < 2000 else [800, 1200, 1700])
            pr = .20 if episode < 1000 else .10 if episode < 2000 else rng.choice([.05, .10, .20])
            network, routes = prepare_scenario(base/'workspace', demand, pr, args.scenario, args.hdv_mixture)
            recording = videos.next(f'training_episode{episode+1}_seed{args.seed+episode}', args.max_steps,
                episode=episode+1, seed=args.seed+episode, demand=demand, pr=pr,
                scenario=args.scenario, hdv_mixture=args.hdv_mixture)
            env = CommonRewardMergeEnv(network, routes, args.seed+episode, args.gui, scenario=args.scenario,
                                       recording=recording)
            observer = Observer(args.method)
            epsilon = max(.05, .99*.995**episode)
            total, losses, action_count, parts = 0., [], np.zeros(5, int), dict.fromkeys(env.last_parts, 0.)
            updates, transitions = agent.updates, agent.transitions
            try:
                frame = observer.capture(env)
                for step in range(args.max_steps):
                    decision = agent.act(frame, epsilon=epsilon)
                    controls = controls_for(args.method, env, frame, decision.actions, decision.parameters)
                    policies = {n: decision.probabilities[i] for i, n in enumerate(frame.ids) if n is not None}
                    observer.record_actions(frame, decision.actions)
                    env.step(controls, policies)
                    following = observer.capture(env)
                    loss = agent.observe(frame, decision, env.team_reward, following, step, final=step+1 == args.max_steps)
                    if recording:
                        recording.note_learning(agent.updates-updates, agent.transitions-transitions)
                    if loss:
                        losses.append(loss)
                    total += env.team_reward
                    for i, name in enumerate(frame.ids):
                        if name is not None:
                            action_count[int(decision.actions[i])] += 1
                    for key in parts:
                        parts[key] += env.last_parts[key]
                    if audit:
                        audit.write(json.dumps({'episode': episode+1, 'step': step, 'reward': env.team_reward,
                            'components': env.last_parts, 'controlled_collision_count': len(set(controls)&env.collisions),
                            'controlled_count': len(controls)})+'\n')
                    frame = following
            finally:
                if recording:
                    recording.metadata.update(learner_updates=agent.updates-updates,
                        transitions=agent.transitions-transitions)
                env.close()
            record = {'method': args.method, 'episode': episode+1, 'demand': demand, 'pr': pr,
                'team_return': total, 'transitions': agent.transitions-transitions, 'updates': agent.updates-updates,
                'action_counts': action_count.tolist(), 'epsilon': epsilon,
                'reward_components_mean': {k: v/args.max_steps for k, v in parts.items()},
                'losses': {k: float(np.mean([v[k] for v in losses if k in v])) for k in {k for v in losses for k in v}}}
            with (base/'training_metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            if (episode+1) % CONFIG['checkpoint_interval'] == 0 or episode+1 == args.episodes:
                agent.save(base/'models'/f'checkpoint-{episode+1}.pt', episode+1, experiment)
    finally:
        if audit:
            audit.close()


if __name__ == '__main__':
    main()
