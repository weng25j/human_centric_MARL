"""All-method output discovery and comparison provenance/seed pairing guards."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import load_workbook

from analysis.combine_evaluation_reports import export_results
from analysis.compare_common_reward import collect_common, compare
from analysis.plot_results import grouped_rows, render_tables
from common.benchmark import LABELS, METHODS, reward_contract
from common.merging_metrics import REPORT_METRICS


def fixture(root, method, steps=1000, fingerprint=None):
    metadata = {**reward_contract(), 'method': method, 'scenario': 'ramp', 'hdv_mixture': 'default',
        'demand': 800, 'pr': .2, 'evaluation_seeds': [0, 25], 'evaluation_steps': steps,
        'training_seed': 42, 'checkpoint_episode': 1, 'training_environment_steps': 1000, 'max_controlled_cavs': 5}
    if fingerprint:
        metadata['reward_fingerprint'] = fingerprint
    seeds = [{'seed': seed, **{key: float(i+1) for i, (key, _, _) in enumerate(REPORT_METRICS)}, 'team_return': 1.}
             for seed in [0, 25]]
    report = root/f'report_{method}.txt'
    report.write_text(f'{LABELS[method]} [COMMON/ramp/default]: 800 veh/h/lane | 20% PR (Over 2 Seeds)\n'
                      'Mean Speed: 10 ± 1 m/s\n', encoding='utf-8')
    report.with_suffix('.metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    path = root/f'metrics_{method}.json'
    path.write_text(json.dumps({'metadata': metadata, 'seeds': seeds}), encoding='utf-8')
    return path


class ComparisonOutputTests(unittest.TestCase):
    def test_export_and_dynamic_tables_include_all_eleven_methods(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for method in METHODS:
                fixture(root, method)
            self.assertEqual(export_results([root], root/'out.csv', root/'out.xlsx'), len(METHODS))
            book = load_workbook(root/'out.xlsx', read_only=True, data_only=True)
            try:
                cells = list(book['summary'].values)
                rows = [dict(zip(cells[0], row)) for row in cells[1:]]
                self.assertEqual({row['Method key'] for row in rows}, set(METHODS))
                self.assertEqual({row['Reward fingerprint'] for row in rows}, {reward_contract()['reward_fingerprint']})
            finally:
                book.close()
            self.assertEqual(len(grouped_rows(root/'out.csv')), 1)
            self.assertEqual(len(render_tables(root/'out.csv', root/'table.png')), 1)
            self.assertTrue((root/'table.png').is_file())

    def test_only_matching_seed_horizon_and_reward_runs_are_paired(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture(root, 'shepherd_v2_common')
            fixture(root, 'maac_common')
            self.assertEqual(compare([root], root/'comparison'), 2)
            with (root/'comparison/paired_differences.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(REPORT_METRICS))
            self.assertTrue(all(float(row['Reference minus comparator']) == 0 for row in rows))
            fixture(root, 'maac_common', steps=2000)
            compare([root], root/'different_horizon')
            with (root/'different_horizon/paired_differences.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            fixture(root, 'maac_common', fingerprint='different')
            with self.assertRaisesRegex(ValueError, 'reward fingerprints'):
                collect_common([root])

    def test_legacy_json_and_invalid_seeds_cannot_enter_matched_comparisons(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'metrics_legacy.json').write_text('[{"seed":0}]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'No versioned'):
                collect_common([root])
            path = fixture(root, 'trust_marl_common')
            value = json.loads(path.read_text())
            value['seeds'][1]['seed'] = 0
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'seed provenance'):
                collect_common([root])


if __name__ == '__main__':
    unittest.main()
