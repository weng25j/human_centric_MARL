"""Dynamic exported-CSV charts/tables, separated by reward and experiment protocol."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.combine_evaluation_reports import COLUMNS


def grouped_rows(csv_file):
    groups = defaultdict(list)
    with Path(csv_file).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if not row.get('Model'):
                continue
            key = tuple(row.get(column, '') for column in (
                'Reward protocol', 'Reward fingerprint', 'Scenario', 'HDV mixture',
                'Demand (vph/lane)', 'Penetration Rate (%)', 'Evaluation steps', 'Evaluation seeds', 'Control budget'))
            groups[key].append(row)
    if not groups:
        raise ValueError('No evaluation rows in CSV')
    return groups


def numeric(value):
    if not value:
        return None
    parts = value.split('±')
    try:
        mean, ci = float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.
        return (mean, ci) if math.isfinite(mean) and math.isfinite(ci) and ci >= 0 else None
    except ValueError:
        return None


def label(row):
    name = row['Model'].split(' [')[0]
    if row.get('Training seed'):
        name += f' (train {row["Training seed"]})'
    return name


def title(key):
    protocol, _, scenario, mixture, demand, pr, steps, _, _ = key
    return f'{protocol or "Legacy: reward contract unrecorded"} | {scenario}/{mixture} | {demand} veh/h/lane | {pr}% PR | {steps or "unrecorded"} steps'


def generate_charts(csv_file='combined_traffic_results.csv', output_dir='comparison_charts'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, rows in grouped_rows(csv_file).items():
        suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
        metrics = list(dict.fromkeys(COLUMNS.values()))+['Common Team Return (reward)']
        for index, metric in enumerate(metrics):
            valid = [(row, numeric(row.get(metric))) for row in rows]
            valid = [(row, value) for row, value in valid if value is not None]
            if not valid:
                continue
            fig, axis = plt.subplots(figsize=(12, max(4, len(valid)*.45)))
            names = [label(row) for row, _ in valid]
            colors = ['#218c74' if 'SHEPHERD' in name else '#3867a8' for name in names]
            axis.barh(np.arange(len(valid)), [v[0] for _, v in valid], xerr=[v[1] for _, v in valid], color=colors, capsize=3)
            axis.set_yticks(np.arange(len(valid)), labels=names)
            axis.invert_yaxis()
            axis.set_xlabel(metric)
            axis.set_title(title(key), fontsize=9)
            fig.tight_layout()
            destination = output/f'comparison_{suffix}_{index:02d}.png'
            fig.savefig(destination, dpi=150)
            plt.close(fig)
            manifest.append({'plot': str(destination), 'metric': metric, 'models': names,
                             'source_reports': [row.get('Source report') for row, _ in valid]})
    (output/'chart_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Generated {len(manifest)} charts for all reported methods in {output}')
    return manifest


def render_tables(csv_file='combined_traffic_results.csv', output_path='reports/figures/comprehensive_results_table.png'):
    output, groups = Path(output_path), grouped_rows(csv_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [('Model', 'Method'), ('Equivalent Hourly Throughput (veh/h)', 'Throughput'),
               ('Mean Speed (m/s)', 'Speed'), ('Collision Rate (%)', 'Collision %'),
               ('Avg Hard Braking', 'Hard braking'), ('Mean Abs Jerk (m/s^3)', 'Jerk'),
               ('90th-Percentile Delay (s)', 'HDV P90 delay')]
    paths = []
    for key, rows in groups.items():
        text = [[label(row) if source == 'Model' else row.get(source, '') for source, _ in columns] for row in rows]
        fig, axis = plt.subplots(figsize=(19, max(3, len(rows)*.45+1.5)))
        axis.axis('off')
        table = axis.table(cellText=text, colLabels=[name for _, name in columns], cellLoc='center', loc='center',
                           colWidths=[.24]+[.125]*6)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2f4f4f')
                cell.set_text_props(color='white', weight='bold')
            elif 'SHEPHERD' in text[r-1][0]:
                cell.set_facecolor('#e6f2ff')
        axis.set_title(title(key), fontsize=10)
        fig.tight_layout()
        suffix = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
        destination = output if len(groups) == 1 else output.with_name(output.stem+'_'+suffix+output.suffix)
        fig.savefig(destination, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths.append(destination)
    print(f'Generated {len(paths)} tables including all methods and PR conditions')
    return paths
