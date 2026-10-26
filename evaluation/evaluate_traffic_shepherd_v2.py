"""Evaluate Shepherd 2.0 checkpoints, roles, HDV delays and five metric families."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.traffic_shepherd_v2 import ACTION_NAMES, TrafficShepherdV2
from common.merging_metrics import MergingMetrics, REPORT_METRICS, mean_ci
from environments.traffic_shepherd_v2 import RewardConfig, TrafficShepherdV2Env
from environments.shepherd_v2_scenarios import MIXTURES, prepare_scenario
from recording.live_sumo import VideoPlan, add_video_arguments, validate_video_arguments

DEFAULT_SEEDS = [0,25,50,75,100,125,150,175,200,325,350,375,400,425,450,475,500,525,550,575]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--eval-lane-demands', default='800,1700')
    parser.add_argument('--eval-prs', default='0.05,0.10,0.20')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--scenario', choices=('ramp', 'lane_drop'), help='Default: checkpoint scenario; lane_drop tests transfer')
    parser.add_argument('--hdv-mixtures', nargs='+', choices=MIXTURES)
    parser.add_argument('--max-controlled-cavs', type=int, help='Default: checkpoint budget')
    parser.add_argument('--gui', action='store_true')
    add_video_arguments(parser)
    args = parser.parse_args(argv)
    validate_video_arguments(parser, args)
    try:
        demands = [int(x) for x in args.eval_lane_demands.split(',')]
        prs = [float(x) for x in args.eval_prs.split(',')]
        seeds = [int(x) for x in args.seeds.split(',')]
        if min(demands) <= 0 or any(not 0 <= p <= 1 for p in prs) or args.max_steps < 1:
            raise ValueError()
        if len(set(seeds)) != len(seeds) or len(set(demands)) != len(demands) or len(set(prs)) != len(prs):
            raise ValueError()
        if args.max_controlled_cavs is not None and args.max_controlled_cavs < 0:
            raise ValueError()
        if args.hdv_mixtures and len(set(args.hdv_mixtures)) != len(args.hdv_mixtures):
            raise ValueError()
    except ValueError:
        parser.error('Use positive demands/steps, PRs in [0,1], non-negative budget and unique grid entries/seeds')
    agent, checkpoint = TrafficShepherdV2.load(args.model_dir)
    experiment = checkpoint['experiment']
    scenario = args.scenario or experiment['scenario']
    mixtures = args.hdv_mixtures or [experiment['hdv_mixture']]
    budget = experiment['max_controlled_cavs'] if args.max_controlled_cavs is None else args.max_controlled_cavs
    model_base = args.model_dir if args.model_dir.is_dir() else args.model_dir.parent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output_dir or model_base/'evaluation'/stamp).expanduser().resolve()
    if (output/'evaluation_settings.json').exists():
        parser.error('Evaluation output already exists; choose a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    (output/'evaluation_settings.json').write_text(json.dumps({'method': 'traffic_shepherd_v2',
        'arguments': vars(args), 'training_experiment': experiment, 'checkpoint_episode': checkpoint['episode'],
        'checkpoint_training_seed': checkpoint['seed'], 'scenario': scenario, 'hdv_mixtures': mixtures,
        'max_controlled_cavs': budget, 'step_seconds': 0.1}, default=str, indent=2), encoding='utf-8')
    videos = VideoPlan(args, output/'recordings', 'traffic_shepherd_v2', 'evaluation')
    for mixture in mixtures:
        for demand in demands:
            for pr in prs:
                name = f'SHEPHERD_V2_{scenario}_{mixture}_{experiment["ablation"]}_{demand}vph_{pr*100:g}PR'
                records, traces, inference = [], [], []
                for seed in seeds:
                    torch.manual_seed(seed)
                    network, routes = prepare_scenario(output/'workspace', demand, pr, scenario, mixture)
                    recording = videos.next(f'evaluation_{scenario}_{mixture}_d{demand}_pr{pr:g}_seed{seed}', args.max_steps,
                        seed=seed, demand=demand, pr=pr, scenario=scenario, hdv_mixture=mixture,
                        checkpoint=str(args.model_dir), checkpoint_episode=checkpoint['episode'], learner_updates=0)
                    env = TrafficShepherdV2Env(network, routes, seed, args.gui, budget,
                        RewardConfig(**experiment['reward']), experiment['action_mode'], scenario, recording=recording)
                    collector = MergingMetrics(env.connection)
                    role_counts, hdv, requested, control_steps = np.zeros(5, int), {}, 0, 0
                    fields = ['time_s', 'vehicle', 'action', 'role', 'requested_acceleration',
                              'speed', 'acceleration', 'gap', 'ttc', 'local_hdv_count',
                              'local_hdv_uncertainty', 'collided']
                    try:
                        with (output/f'roles_{name}_seed{seed}.csv').open('w', newline='', encoding='utf-8') as stream:
                            writer = csv.DictWriter(stream, fieldnames=fields)
                            writer.writeheader()
                            observations = env.observe()
                            for _ in range(args.max_steps):
                                start = time.perf_counter()
                                actions, _, policies = agent.act(observations, deterministic=True)
                                if actions:
                                    inference.append(1000*(time.perf_counter()-start))
                                # Save local context before the action and outcome after it.
                                context = {n: [env.states[h] for h in env.local_ids(n) if not h.startswith('cav')]
                                           for n in actions}
                                observations, _ = env.step(actions, policies)
                                collector.step(env.collisions)
                                for vehicle, state in env.states.items():
                                    if not vehicle.startswith('cav'):
                                        hdv[vehicle] = {'vehicle': vehicle, 'time_loss_s': state['loss'],
                                                        'status': 'unfinished'}
                                for vehicle in env.arrivals:
                                    if vehicle in hdv:
                                        hdv[vehicle]['status'] = 'arrived'
                                for vehicle in env.collisions:
                                    if vehicle in hdv:
                                        hdv[vehicle]['status'] = 'collided'
                                for vehicle, action in actions.items():
                                    role_counts[action] += 1
                                    requested += int(env.last_commands[vehicle] is not None)
                                    control_steps += 1
                                    state = env.states.get(vehicle, {})
                                    humans = context[vehicle]
                                    role = ACTION_NAMES[action] if experiment['action_mode'] == 'structured' else f'primitive_{action}'
                                    writer.writerow({'time_s': env.steps*env.dt, 'vehicle': vehicle,
                                        'action': action, 'role': role, 'requested_acceleration': env.last_commands[vehicle],
                                        'speed': state.get('speed'), 'acceleration': state.get('acceleration'),
                                        'gap': state.get('gap'), 'ttc': state.get('ttc'), 'local_hdv_count': len(humans),
                                        'local_hdv_uncertainty': float(np.mean([h['uncertainty'] for h in humans])) if humans else 0,
                                        'collided': int(vehicle in env.collisions)})
                    finally:
                        env.close()
                    delay = [h['time_loss_s'] for h in hdv.values()]
                    record = {'seed': seed, **collector.result(), 'role_counts': role_counts.tolist(),
                        'requested_interventions': requested, 'requested_duty': 100*requested/max(1, control_steps),
                        'observed_hdv_delay_mean': float(np.mean(delay)) if delay else 0,
                        'observed_hdv_delay90': float(np.percentile(delay, 90)) if delay else 0,
                        'hdv_completed': sum(h['status'] == 'arrived' for h in hdv.values()),
                        'hdv_unfinished': sum(h['status'] == 'unfinished' for h in hdv.values())}
                    records.append(record)
                    (output/f'hdv_delays_{name}_seed{seed}.json').write_text(json.dumps(list(hdv.values()), indent=2), encoding='utf-8')
                    traces.append([collector.speed_trace, collector.decel_trace, collector.jerk_trace])
                    print(f'Shepherd 2.0: {scenario}/{mixture}, demand={demand}, PR={pr:g}, seed={seed}, roles={role_counts.tolist()}', flush=True)
                (output/f'metrics_{name}.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
                label = f'TRAFFIC SHEPHERD 2.0 [{scenario}/{mixture}/{experiment["ablation"]}]'
                report = [f'{label}: {demand} veh/h/lane | {pr*100:g}% PR (Over {len(seeds)} Seeds)',
                          'Values reported as: Mean ± 95% Confidence Interval',
                          'Execution check or experiment result; effectiveness requires matched safety/fairness/stability comparisons.']
                extra = [('requested_interventions', 'Requested Role Interventions', 'CAV steps'),
                         ('requested_duty', 'Requested Role Duty', '%'),
                         ('observed_hdv_delay_mean', 'Observed HDV Mean Time Loss', 's'),
                         ('observed_hdv_delay90', 'Observed HDV P90 Time Loss', 's')]
                for key, metric, unit in [*REPORT_METRICS, *extra]:
                    value, ci = mean_ci([r[key] for r in records])
                    report.append(f'{metric}: {value:.6g} ± {ci:.6g} {unit}')
                if inference:
                    report += [f'Inference Mean: {np.mean(inference):.6g} ms',
                               f'Inference P90: {np.percentile(inference, 90):.6g} ms']
                (output/f'report_{name}.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
                averaged = np.mean(traces, axis=0)
                fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
                for index, metric in enumerate(('speed', 'decel', 'jerk')):
                    np.save(output/f'{metric}_array_{name}.npy', averaged[index])
                    axes[index].plot(np.arange(args.max_steps)*0.1, averaged[index])
                    axes[index].set_ylabel(metric)
                axes[-1].set_xlabel('Simulation time (s)')
                fig.tight_layout()
                fig.savefig(output/f'traces_{name}.png', dpi=150)
                plt.close(fig)
                print('Saved', output/f'report_{name}.txt', flush=True)


if __name__ == '__main__':
    main()
