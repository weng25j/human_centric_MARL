"""Evaluate Nash Double-Q on the existing merging grid, with reports/arrays for Excel."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from algorithms.nash_double_q import NashDoubleQAgent
from common.merging_metrics import MergingMetrics, REPORT_METRICS, mean_ci
from environments.nash_merge import NashMergeEnv, encode_game, interaction_groups, prepare_routes

DEFAULT_SEEDS = [0,25,50,75,100,125,150,175,200,325,350,375,400,425,450,475,500,525,550,575]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--eval-lane-demands', default='800,1700')
    parser.add_argument('--eval-prs', default='0.05,0.10,0.20')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--max-controlled-cavs', type=int, default=5, help='Match the existing five-CAV control budget; 0 controls all')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args(argv)
    if args.max_controlled_cavs < 0:
        parser.error('--max-controlled-cavs must be non-negative')
    try:
        demands = [int(x) for x in args.eval_lane_demands.split(',')]
        prs = [float(x) for x in args.eval_prs.split(',')]
        seeds = [int(x) for x in args.seeds.split(',')]
        if min(demands) <= 0 or any(not 0 <= p <= 1 for p in prs) or args.max_steps < 1 or len(set(seeds)) != len(seeds):
            raise ValueError()
    except ValueError:
        parser.error('Supply positive demands/steps, PR fractions in [0,1], and unique integer seeds')
    agent, checkpoint = NashDoubleQAgent.load(args.model_dir)
    output = (args.output_dir or (args.model_dir if args.model_dir.is_dir() else args.model_dir.parent)/'evaluation').resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'evaluation_settings.json').write_text(json.dumps({'method': 'nash_double_q_adaptation',
        'arguments': vars(args), 'learner': asdict(agent.config), 'checkpoint_episode': checkpoint['episode'], 'checkpoint_training_seed': checkpoint['seed'],
        'pair_radius': 100.0, 'step_seconds': 0.1}, default=str, indent=2))
    for demand in demands:
        for pr in prs:
            records, traces, inference = [], [], []
            for seed in seeds:
                # Reset sampling per seed, including mixed-equilibrium sampling.
                agent.rng = np.random.default_rng(seed)
                network, routes = prepare_routes(output/'workspace', demand, pr)
                env = NashMergeEnv(network, routes, seed, args.gui, max_controlled=args.max_controlled_cavs)
                collector = MergingMetrics(env.connection)
                joint_games = 0
                try:
                    states = env.snapshot()
                    for _ in range(args.max_steps):
                        actions = {}
                        begin = time.perf_counter()
                        for game in interaction_groups(states):
                            state, masks = encode_game(game, states)
                            selected, _ = agent.select(state, masks, deterministic=True)
                            joint_games += int(game[1] is not None)
                            actions.update({name: selected[role] for role, name in enumerate(game) if name})
                        if actions:
                            inference.append(1000*(time.perf_counter()-begin))
                        states, _ = env.step(actions)
                        collector.step(env.collisions)
                finally:
                    env.close()
                record = {'seed': seed, 'joint_games': joint_games, **collector.result()}
                records.append(record)
                traces.append([collector.speed_trace, collector.decel_trace, collector.jerk_trace])
                print(f'Nash Double-Q: demand={demand}, PR={pr:g}, seed={seed}, joint games={joint_games}', flush=True)
            name = f'NASH_DOUBLE_Q_{demand}vph_{pr*100:g}PR'
            (output/f'metrics_{name}.json').write_text(json.dumps(records, indent=2))
            report = [f'NASH DOUBLE-Q: {demand} veh/h/lane | {pr*100:g}% PR (Over {len(seeds)} Seeds)',
                      'Values reported as: Mean ± 95% Confidence Interval',
                      'Method: Nash Double-Q SUMO adaptation; not an exact paper reproduction.']
            for key, label, unit in REPORT_METRICS:
                value, ci = mean_ci([row[key] for row in records])
                report.append(f'{label}: {value:.6g} ± {ci:.6g} {unit}')
            if inference:
                report.append(f'Inference P90: {np.percentile(inference,90):.6g} ms')
                report.append(f'Inference Mean: {np.mean(inference):.6g} ms')
            (output/f'report_{name}.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
            averaged = np.mean(traces, axis=0)
            fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
            for index, (metric, unit) in enumerate([('speed','m/s'), ('decel','m/s²'), ('jerk','m/s³')]):
                np.save(output/f'{metric}_array_{name}.npy', averaged[index])
                axes[index].plot(np.arange(args.max_steps)*0.1, averaged[index])
                axes[index].set_ylabel(f'{metric} ({unit})')
            axes[-1].set_xlabel('Simulation time (s)')
            fig.tight_layout()
            fig.savefig(output/f'traces_{name}.png', dpi=150)
            plt.close(fig)
            print('Saved',output/f'report_{name}.txt',flush=True)


if __name__ == '__main__':
    main()
