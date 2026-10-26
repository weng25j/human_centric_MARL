"""Plot a Shepherd 2.0 CAV's roles/gaps and an optional HDV time-loss distribution."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.traffic_shepherd_v2 import ACTION_NAMES


def plot_roles(trace_csv, output=None, vehicle=None, delay_json=None):
    trace_csv = Path(trace_csv)
    with trace_csv.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No CAV decisions in this trace (possibly 0% PR or a short empty rollout)')
    if vehicle is None:
        vehicle = Counter(r['vehicle'] for r in rows).most_common(1)[0][0]
    selected = [r for r in rows if r['vehicle'] == vehicle]
    if not selected:
        raise ValueError(f'No trace for {vehicle}')
    def values(key):
        return np.asarray([float(r[key]) if r[key] else np.nan for r in selected])
    seconds = values('time_s')
    figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(seconds, values('speed'))
    axes[0].set_ylabel('Speed (m/s)')
    axes[1].plot(seconds, values('gap'))
    axes[1].set_ylabel('Leader gap (m)')
    axes[2].plot(seconds, values('acceleration'), label='realized')
    axes[2].plot(seconds, values('requested_acceleration'), alpha=.7, label='requested')
    axes[2].set_ylabel('Accel. (m/s²)')
    axes[2].legend()
    axes[3].step(seconds, values('action'), where='post')
    labels = [f'primitive_{i}' for i in range(5)] if selected[0]['role'].startswith('primitive_') else ACTION_NAMES
    axes[3].set_yticks(range(5), labels=[s.replace('_', ' ') for s in labels])
    axes[3].set_ylabel('Role')
    axes[3].set_xlabel('Simulation time (s)')
    for row in selected:
        if row['collided'] == '1':
            for axis in axes:
                axis.axvline(float(row['time_s']), color='red', alpha=.5)
    figure.suptitle(f'Traffic Shepherd 2.0: {vehicle}')
    figure.tight_layout()
    output = Path(output) if output else trace_csv.with_name(trace_csv.stem+'_plot.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print('Saved', output)
    if delay_json:
        data = json.loads(Path(delay_json).read_text(encoding='utf-8'))
        figure, axis = plt.subplots(figsize=(8, 5))
        plotted = False
        for status, label in [('arrived', 'arrived HDVs'), ('unfinished', 'unfinished HDVs (censored)'),
                              ('collided', 'collided HDVs')]:
            delays = sorted(r['time_loss_s'] for r in data if r['status'] == status)
            if delays:
                axis.step(delays, np.arange(1, len(delays)+1)/len(delays), where='post', label=label)
                plotted = True
        axis.set(xlabel='Last observed SUMO time loss (s)', ylabel='Empirical cumulative fraction', ylim=(0, 1.02))
        if plotted:
            axis.legend()
        figure.tight_layout()
        delay_output = output.with_name(output.stem+'_hdv_delay.png')
        figure.savefig(delay_output, dpi=150)
        plt.close(figure)
        print('Saved', delay_output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-csv', required=True, type=Path)
    parser.add_argument('--vehicle', help='Default: CAV with the most trace samples')
    parser.add_argument('--delay-json', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot_roles(args.trace_csv, args.output, args.vehicle, args.delay_json)


if __name__ == '__main__':
    main()
