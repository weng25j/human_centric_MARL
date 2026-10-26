"""Evaluate all common-reward paper comparisons with identical SUMO metrics."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.benchmark_base import BenchmarkLearner
from algorithms.common_reward_agents import make_agent
from common.benchmark import FIXED, LABELS, METHODS, reward_contract
from common.merging_metrics import MergingMetrics, REPORT_METRICS, mean_ci
from environments.common_reward_merge import CommonRewardMergeEnv, Observer, controls_for
from environments.shepherd_v2_scenarios import MIXTURES, prepare_scenario
from evaluation.evaluate_traffic_shepherd_v2 import DEFAULT_SEEDS


def main(argv=None, method=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default=method, required=method is None)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--eval-lane-demands', default='800,1700')
    parser.add_argument('--eval-prs', default='0.05,0.10,0.20')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--scenario', choices=('ramp', 'lane_drop'), default='ramp')
    parser.add_argument('--hdv-mixtures', nargs='+', choices=MIXTURES, default=['default'])
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args(argv)
    if method is not None and args.method != method:
        parser.error(f'This entry point evaluates {method}')
    try:
        demands = [int(x) for x in args.eval_lane_demands.split(',')]
        prs = [float(x) for x in args.eval_prs.split(',')]
        seeds = [int(x) for x in args.seeds.split(',')]
        if min(demands) <= 0 or any(not 0 <= p <= 1 for p in prs) or args.max_steps < 1:
            raise ValueError()
        if any(len(set(grid)) != len(grid) for grid in (demands, prs, seeds, args.hdv_mixtures)):
            raise ValueError()
    except ValueError:
        parser.error('Use positive demands/steps, PRs in [0,1], and unique seeds/grid entries')
    if args.method in FIXED:
        if args.model_dir:
            parser.error('Fixed policies do not accept a model directory')
        agent, checkpoint = make_agent(args.method), {'episode': 0, 'seed': None}
    else:
        if not args.model_dir:
            parser.error('--model-dir is required for a learned method')
        agent, checkpoint = BenchmarkLearner.load(args.model_dir, args.method)
    if args.method == 'all_hdv_common':
        prs = [0.0]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output_dir or Path('results_common')/args.method/'evaluation'/stamp).expanduser().resolve()
    if (output/'evaluation_settings.json').exists():
        parser.error('Existing evaluation: select a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    settings = {**reward_contract(), 'method': args.method, 'arguments': vars(args),
                'training_seed': checkpoint['seed'], 'checkpoint_episode': checkpoint['episode'],
                'scenario': args.scenario, 'hdv_mixtures': args.hdv_mixtures, 'evaluation_seeds': seeds,
                'evaluation_steps': args.max_steps, 'max_controlled_cavs': 5}
    (output/'evaluation_settings.json').write_text(json.dumps(settings, default=str, indent=2), encoding='utf-8')
    for mixture in args.hdv_mixtures:
        for demand in demands:
            for pr in prs:
                key = f'{args.method}_{args.scenario}_{mixture}_{demand}vph_{pr*100:g}PR'
                records, traces, inference = [], [], []
                for seed in seeds:
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    network, routes = prepare_scenario(output/'workspace', demand, pr, args.scenario, mixture)
                    env = CommonRewardMergeEnv(network, routes, seed, args.gui, scenario=args.scenario)
                    observer, collector = Observer(args.method), MergingMetrics(env.connection)
                    counts, total, requested, decisions = np.zeros(5, int), 0., 0, 0
                    parts, hdv = dict.fromkeys(env.last_parts, 0.), {}
                    fields = ['time_s', 'vehicle', 'action', 'requested_acceleration', 'lane_delta',
                              'speed', 'acceleration', 'gap', 'ttc', 'reward', 'collided']
                    try:
                        with (output/f'actions_{key}_seed{seed}.csv').open('w', newline='', encoding='utf-8') as stream:
                            writer = csv.DictWriter(stream, fieldnames=fields)
                            writer.writeheader()
                            frame = observer.capture(env)
                            for step in range(args.max_steps):
                                start = time.perf_counter()
                                decision = agent.act(frame, evaluation=True)
                                if frame.present.any():
                                    inference.append(1000*(time.perf_counter()-start))
                                controls = controls_for(args.method, env, frame, decision.actions, decision.parameters)
                                observer.record_actions(frame, decision.actions)
                                env.step(controls, {n: decision.probabilities[i] for i, n in enumerate(frame.ids) if n is not None})
                                collector.step(env.collisions)
                                total += env.team_reward
                                for component in parts:
                                    parts[component] += env.last_parts[component]
                                for name, state in env.states.items():
                                    if not name.startswith('cav'):
                                        hdv[name] = {'vehicle': name, 'time_loss_s': state['loss'], 'status': 'unfinished'}
                                for status, names in [('arrived', env.arrivals), ('collided', env.collisions)]:
                                    for name in names:
                                        if name in hdv:
                                            hdv[name]['status'] = status
                                for i, name in enumerate(frame.ids):
                                    if name is None:
                                        continue
                                    counts[int(decision.actions[i])] += 1
                                    decisions += 1
                                    requested += int(env.last_commands[name] is not None)
                                    state = env.states.get(name, {})
                                    writer.writerow({'time_s': env.steps*env.dt, 'vehicle': name,
                                        'action': int(decision.actions[i]), 'requested_acceleration': env.last_commands[name],
                                        'lane_delta': controls[name].lane_delta, 'speed': state.get('speed'),
                                        'acceleration': state.get('acceleration'), 'gap': state.get('gap'),
                                        'ttc': state.get('ttc'), 'reward': env.team_reward, 'collided': int(name in env.collisions)})
                                frame = observer.capture(env)
                    finally:
                        env.close()
                    delay = [row['time_loss_s'] for row in hdv.values()]
                    records.append({'seed': seed, **collector.result(), 'team_return': total,
                        'requested_interventions': requested, 'requested_duty': 100*requested/max(1, decisions),
                        'action_counts': counts.tolist(), 'reward_components_mean': {k: v/args.max_steps for k, v in parts.items()},
                        'observed_hdv_delay90': float(np.percentile(delay, 90)) if delay else 0,
                        'hdv_unfinished': sum(row['status'] == 'unfinished' for row in hdv.values())})
                    traces.append([collector.speed_trace, collector.decel_trace, collector.jerk_trace])
                    (output/f'hdv_delays_{key}_seed{seed}.json').write_text(json.dumps(list(hdv.values()), indent=2), encoding='utf-8')
                    print(f'{args.method}: {args.scenario}/{mixture}, demand={demand}, PR={pr:g}, seed={seed}, actions={counts.tolist()}', flush=True)
                metadata = {k: v for k, v in settings.items() if k != 'arguments'}
                metadata.update(hdv_mixture=mixture, demand=demand, pr=pr)
                (output/f'metrics_{key}.json').write_text(json.dumps({'metadata': metadata, 'seeds': records}, indent=2), encoding='utf-8')
                label = f'{LABELS[args.method]} [COMMON/{args.scenario}/{mixture}]'
                report = [f'{label}: {demand} veh/h/lane | {pr*100:g}% PR (Over {len(seeds)} Seeds)',
                          'Values reported as: Mean ± 95% Confidence Interval',
                          'Paper mechanisms adapted to a shared SUMO benchmark; not an exact published-experiment reproduction.']
                extra = [('team_return', 'Common Team Return', 'reward'),
                         ('requested_interventions', 'Requested Interventions', 'CAV steps'),
                         ('requested_duty', 'Requested Duty', '%'),
                         ('observed_hdv_delay90', 'Observed HDV P90 Time Loss', 's')]
                for metric, title, unit in [*REPORT_METRICS, *extra]:
                    value, ci = mean_ci([r[metric] for r in records])
                    report.append(f'{title}: {value:.6g} ± {ci:.6g} {unit}')
                for component in parts:
                    value, ci = mean_ci([r['reward_components_mean'][component] for r in records])
                    report.append(f'Reward Component {component}: {value:.6g} ± {ci:.6g}')
                if inference:
                    report += [f'Inference Mean: {np.mean(inference):.6g} ms', f'Inference P90: {np.percentile(inference, 90):.6g} ms']
                report_path = output/f'report_{key}.txt'
                report_path.write_text('\n'.join(report)+'\n', encoding='utf-8')
                report_path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                mean_traces = np.mean(traces, axis=0)
                fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
                for index, metric in enumerate(('speed', 'decel', 'jerk')):
                    np.save(output/f'{metric}_array_{key}.npy', mean_traces[index])
                    axes[index].plot(np.arange(args.max_steps)*.1, mean_traces[index])
                    axes[index].set_ylabel(metric)
                axes[-1].set_xlabel('Simulation time (s)')
                fig.tight_layout()
                fig.savefig(output/f'traces_{key}.png', dpi=150)
                plt.close(fig)
                print('Saved', report_path, flush=True)


if __name__ == '__main__':
    main()
