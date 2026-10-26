"""Plot every common-reward method and export paired, seed-level differences."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.benchmark_methods import LABELS, METHODS
from common.merging_metrics import REPORT_METRICS, mean_ci


def collect_common(directories):
    paths = set()
    for directory in directories:
        source = Path(directory)
        paths.update([source.resolve()] if source.is_file() else (p.resolve() for p in source.rglob('metrics_*.json')))
    runs = []
    for path in sorted(paths):
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or value.get('metadata', {}).get('protocol') != 'shepherd_common_reward_v1':
            continue
        meta, rows = value['metadata'], value['seeds']
        if meta['method'] not in METHODS:
            raise ValueError(f'Unknown method in {path}')
        seeds = [row['seed'] for row in rows]
        if seeds != meta['evaluation_seeds'] or len(seeds) != len(set(seeds)) or not seeds:
            raise ValueError(f'Invalid seed provenance in {path}')
        if any(not np.isfinite(row[key]) for row in rows for key, _, _ in REPORT_METRICS):
            raise ValueError(f'Non-finite metric in {path}')
        runs.append((path, meta, rows))
    if not runs:
        raise ValueError('No versioned common-reward metrics found; legacy reports are not reward-matched comparisons')
    if len({meta['reward_fingerprint'] for _, meta, _ in runs}) != 1:
        raise ValueError('Different reward fingerprints: do not compare incompatible reward versions')
    return runs


def condition(meta):
    return (meta['scenario'], meta['hdv_mixture'], meta['demand'], meta['pr'],
            meta['evaluation_steps'], tuple(meta['evaluation_seeds']), meta['max_controlled_cavs'], meta['step_seconds'])


def write_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compare(directories, output, reference='shepherd_v2_common'):
    runs, output = collect_common(directories), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grouped, summary, differences = defaultdict(list), [], []
    for path, meta, rows in runs:
        grouped[condition(meta)].append((path, meta, rows))
        for metric, label, unit in [*REPORT_METRICS, ('team_return', 'Common Team Return', 'reward')]:
            value, ci = mean_ci([row[metric] for row in rows])
            summary.append({'Method': meta['method'], 'Training seed': meta['training_seed'],
                'Training environment steps': meta.get('training_environment_steps'),
                'Scenario': meta['scenario'], 'HDV mixture': meta['hdv_mixture'], 'Demand': meta['demand'],
                'PR': meta['pr'], 'Steps': meta['evaluation_steps'], 'Seeds': ','.join(map(str, meta['evaluation_seeds'])),
                'Metric': label, 'Mean': value, '95% CI': ci, 'Unit': unit,
                'Reward fingerprint': meta['reward_fingerprint'], 'Source metrics': str(path)})
    plotted = 0
    for key, entries in grouped.items():
        entries.sort(key=lambda e: (METHODS.index(e[1]['method']), str(e[0])))
        references = [entry for entry in entries if entry[1]['method'] == reference]
        for ref_path, ref_meta, ref_rows in references:
            ref_by_seed = {row['seed']: row for row in ref_rows}
            for path, meta, rows in entries:
                if path == ref_path:
                    continue
                # Keep training runs separate. Do not manufacture training-run
                # uncertainty by pooling rollout seeds or unrelated checkpoints.
                if meta['training_seed'] is not None and meta['training_seed'] != ref_meta['training_seed']:
                    continue
                if meta['training_seed'] is not None and (meta.get('training_environment_steps') is None
                        or meta.get('training_environment_steps') != ref_meta.get('training_environment_steps')):
                    continue
                for metric, label, unit in REPORT_METRICS:
                    delta = [ref_by_seed[row['seed']][metric]-row[metric] for row in rows]
                    value, ci = mean_ci(delta)
                    differences.append({'Reference': reference, 'Comparator': meta['method'],
                        'Training seed': ref_meta['training_seed'], 'Scenario': key[0], 'HDV mixture': key[1],
                        'Demand': key[2], 'PR': key[3], 'Steps': key[4], 'Paired evaluation seeds': len(delta),
                        'Metric': label, 'Reference minus comparator': value, '95% CI': ci, 'Unit': unit,
                        'Reference metrics': str(ref_path), 'Comparator metrics': str(path)})
        names = [LABELS[meta['method']]+(f' (train {meta["training_seed"]})' if meta['training_seed'] is not None else '')
                 for _, meta, _ in entries]
        metrics = [('throughput', 'Throughput (veh/h)'), ('speed', 'Mean speed (m/s)'),
                   ('collision_rate', 'Collision rate (%)'), ('ttc', 'TTC violations'),
                   ('jerk', 'Mean absolute jerk (m/s³)'), ('delay90', 'Arrived HDV P90 delay (s)')]
        figure, axes = plt.subplots(2, 3, figsize=(20, max(8, len(entries)*.65)), layout='constrained')
        colors = ['#218c74' if meta['method'] == reference else '#3867a8' for _, meta, _ in entries]
        for axis, (metric, title) in zip(axes.flat, metrics):
            values = [mean_ci([row[metric] for row in rows]) for _, _, rows in entries]
            axis.barh(np.arange(len(entries)), [v[0] for v in values], xerr=[v[1] for v in values], color=colors, capsize=3)
            axis.set_yticks(np.arange(len(entries)), labels=names, fontsize=8)
            axis.invert_yaxis()
            axis.set_title(title)
            axis.grid(axis='x', alpha=.2)
        scenario, mixture, demand, pr, steps, seeds, _, _ = key
        figure.suptitle(f'Common reward: {scenario}/{mixture} | {demand} veh/h/lane | {pr*100:g}% PR | {steps} steps | {len(seeds)} seeds\n'
                       'Error bars: evaluation-seed 95% CI; runs/checkpoints remain separate')
        suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:8]
        figure.savefig(output/f'comparison_{scenario}_{mixture}_{demand}_{pr:g}_{suffix}.png', dpi=140)
        plt.close(figure)
        plotted += 1
    write_table(output/'common_metrics.csv', summary)
    write_table(output/'paired_differences.csv', differences)
    (output/'comparison_manifest.json').write_text(json.dumps({'reports': len(runs), 'plots': plotted,
        'reference': reference, 'reward_fingerprint': runs[0][1]['reward_fingerprint'],
        'metric_sources': [str(path) for path, _, _ in runs],
        'interpretation': 'Paired differences compare equal grids, seeds, horizons, budgets and reward. No automatic superiority claim. '
                          '0% All-HDV is a separate composition, not a paired same-PR experiment.'}, indent=2), encoding='utf-8')
    print(f'Compared {len(runs)} common-reward reports; saved {plotted} dynamic comparison plots to {output}', flush=True)
    return len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-dirs', nargs='+', default=['results_common'])
    parser.add_argument('--output-dir', type=Path, default=Path('comparison_charts/common_reward'))
    parser.add_argument('--reference', choices=METHODS, default='shepherd_v2_common')
    args = parser.parse_args()
    compare(args.search_dirs, args.output_dir, args.reference)


if __name__ == '__main__':
    main()
