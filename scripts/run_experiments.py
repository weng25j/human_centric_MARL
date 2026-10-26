"""Run the main training/evaluation workflows serially, then export their results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.benchmark_methods import METHODS as COMMON_METHODS, FIXED as COMMON_FIXED, LEARNED as COMMON_LEARNED

DEFAULT_METHODS = ['shepherd', 'flat_mappo', 'z_merge', 'nash_double_q', 'baselines']
METHODS = DEFAULT_METHODS + ['shepherd_v2'] + list(COMMON_METHODS)
MODULES = {'shepherd': 'traffic_shepherd', 'flat_mappo': 'flat_marl', 'z_merge': 'z_merge',
           'nash_double_q': 'nash_double_q', 'shepherd_v2': 'traffic_shepherd_v2'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'results'], default='all')
    parser.add_argument('--suite', choices=['legacy', 'common'], default='legacy', help='Common: reward-matched paper comparisons and Shepherd 2.0')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--run-dir', default=os.environ.get('MARL_RUN_DIR'))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('MARL_SEED', '42')))
    parser.add_argument('--eval-demands', default=os.environ.get('MARL_EVAL_DEMANDS', '800,1700'))
    parser.add_argument('--eval-prs', default=os.environ.get('MARL_EVAL_PRS', '0.05,0.10,0.20'))
    parser.add_argument('--zmerge-episodes', type=int, default=int(os.environ.get('MARL_ZMERGE_EPISODES', '5')))
    parser.add_argument('--python', default=sys.executable, help='Python from the prepared environment')
    parser.add_argument('--shepherd-model-dir', type=Path)
    parser.add_argument('--flat-model-dir', type=Path)
    parser.add_argument('--zmerge-model-prefix', type=Path)
    parser.add_argument('--nash-model-dir', type=Path)
    parser.add_argument('--nash-episodes', type=int, default=3000)
    parser.add_argument('--nash-train-steps', type=int, default=1000)
    parser.add_argument('--nash-eval-steps', type=int, default=4000)
    parser.add_argument('--nash-eval-seeds', help='Optional comma-separated seed subset for Nash evaluation')
    parser.add_argument('--v2-model-dir', type=Path)
    parser.add_argument('--v2-episodes', type=int, default=3000)
    parser.add_argument('--v2-train-steps', type=int, default=1000)
    parser.add_argument('--v2-eval-steps', type=int, default=4000)
    parser.add_argument('--v2-eval-seeds', help='Optional comma-separated seed subset for Shepherd 2.0')
    parser.add_argument('--v2-scenario', choices=['ramp', 'lane_drop'])
    parser.add_argument('--v2-hdv-mixture', choices=['default', 'cautious', 'aggressive'])
    parser.add_argument('--v2-ablation', choices=['full', 'no_fairness', 'no_stability', 'no_control_cost', 'primitive_actions'])
    parser.add_argument('--common-model-root', type=Path, help='Folder containing METHOD/models/checkpoint-N.pt')
    parser.add_argument('--common-episodes', type=int, default=3000)
    parser.add_argument('--common-train-steps', type=int, default=1000)
    parser.add_argument('--common-eval-steps', type=int, default=4000)
    parser.add_argument('--common-eval-seeds', help='Optional comma-separated seed subset for common-reward evaluation')
    parser.add_argument('--common-scenario', choices=['ramp', 'lane_drop'], default='ramp')
    parser.add_argument('--common-hdv-mixture', choices=['default', 'cautious', 'aggressive'], default='default')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without creating files or running jobs')
    args = parser.parse_args(argv)
    args.methods = args.methods or (list(COMMON_METHODS) if args.suite == 'common' else DEFAULT_METHODS)
    if args.suite == 'common' and any(m not in COMMON_METHODS for m in args.methods):
        parser.error('The common suite accepts only *_common methods; legacy rewards are separate')
    if args.stage != 'all' and not args.run_dir:
        parser.error('--run-dir is required for separate stages')
    if args.zmerge_episodes < 1:
        parser.error('--zmerge-episodes must be positive')
    if any(m in COMMON_METHODS for m in args.methods) and min(args.common_episodes, args.common_train_steps, args.common_eval_steps) < 1:
        parser.error('Common-reward episode and step counts must be positive')
    if 'shepherd_v2' in args.methods:
        if min(args.v2_episodes, args.v2_train_steps, args.v2_eval_steps) < 1:
            parser.error('Shepherd 2.0 episode and step counts must be positive')
        if args.stage == 'evaluate' and args.v2_ablation:
            parser.error('The evaluation ablation is read from the checkpoint; --v2-ablation is a training option')
    try:
        if any(int(x) <= 0 for x in args.eval_demands.split(',')):
            raise ValueError()
        if any(not 0 <= float(x) <= 1 for x in args.eval_prs.split(',')):
            raise ValueError()
    except ValueError:
        parser.error('Demands must be positive integers and PRs must be fractions in [0, 1].')
    return args


def model_location(args, method, run_dir):
    if method in COMMON_LEARNED:
        root = args.common_model_root.expanduser().resolve() if args.common_model_root else run_dir/'models'
        return root/method/'models'
    supplied = {'shepherd': args.shepherd_model_dir, 'flat_mappo': args.flat_model_dir,
                'z_merge': args.zmerge_model_prefix, 'nash_double_q': args.nash_model_dir,
                'shepherd_v2': args.v2_model_dir}[method]
    if supplied:
        return supplied.expanduser().resolve()
    base = run_dir/'models'/method
    if method in ('nash_double_q', 'shepherd_v2'):
        return base/'models'
    if method == 'shepherd':
        return base/'MAPPO_Curriculum_seed_0809/models'
    if method == 'flat_mappo':
        return base/'0809/models'
    candidates = list(base.glob('*/models/zmerge_final_qnet.pth'))
    if not candidates:
        if args.dry_run:
            return base/'<generated-run>/models/zmerge_final'
        raise FileNotFoundError(f'No completed Z-Merge model under {base}')
    latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    return Path(str(latest).removesuffix('_qnet.pth'))


def validate_weights(method, location):
    if method == 'z_merge':
        for suffix in ['_qnet.pth', '_paramnet.pth']:
            if not Path(f'{location}{suffix}').is_file():
                raise FileNotFoundError(f'Missing model: {location}{suffix}')
    else:
        if not location.is_dir() or not any(
            p.stem.startswith('checkpoint-') and p.stem[11:].isdigit() and int(p.stem[11:]) > 0
            for p in location.glob('checkpoint-*.pt')
        ):
            raise FileNotFoundError(f'No numbered checkpoint in {location}')


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir or ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')).expanduser().resolve()
    python = str(Path(args.python).absolute()) if Path(args.python).is_file() else args.python
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    # Put the selected virtual environment's SUMO executable on the child PATH.
    executable = shutil.which(python) or python
    env['PATH'] = str(Path(executable).parent) + os.pathsep + env.get('PATH', '')
    env.setdefault('MPLBACKEND', 'Agg')
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    env.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    print(f'Run directory: {run_dir}', flush=True)

    def run(label, module, arguments, cwd):
        command = [python, '-m', module, *map(str, arguments)]
        print(f'[{label}] cwd={cwd}\n  {shlex.join(command)}', flush=True)
        if args.dry_run:
            return
        log = run_dir/'logs'/f'{label}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', encoding='utf-8') as stream:
            with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors='replace') as child:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Use the selected interpreter for actual package versions in this snapshot.
        settings = subprocess.check_output([python, str(ROOT/'scripts/show_settings.py')], env=env, text=True)
        (run_dir/'settings.json').write_text(settings)
        (run_dir/'pipeline_options.json').write_text(json.dumps(vars(args), default=str, indent=2))

    if args.stage in ['all', 'train', 'evaluate']:
        for method in dict.fromkeys(args.methods):
            cwd = run_dir/'workspaces'/method
            if not args.dry_run:
                cwd.mkdir(parents=True, exist_ok=True)
                if not (cwd/'SUMO_network').exists():
                    shutil.copytree(ROOT/'SUMO_network', cwd/'SUMO_network')
            if args.stage in ['all', 'train'] and method not in ('baselines', *COMMON_FIXED):
                base = run_dir/'models'/method
                if not args.dry_run:
                    if base.exists() and (any(base.rglob('*.pt')) or any(base.rglob('*.pth'))):
                        raise FileExistsError(f'Existing trained models in {base}; select a new run directory.')
                    base.mkdir(parents=True, exist_ok=True)
                training_options = ['--seed', args.seed, '--base-dir', str(base)+os.sep]
                if method in COMMON_LEARNED:
                    training_options += ['--method', method, '--episodes', args.common_episodes,
                        '--max-steps', args.common_train_steps, '--scenario', args.common_scenario,
                        '--hdv-mixture', args.common_hdv_mixture]
                if method == 'nash_double_q':
                    training_options += ['--episodes', args.nash_episodes, '--max-steps', args.nash_train_steps]
                if method == 'shepherd_v2':
                    training_options += ['--episodes', args.v2_episodes, '--max-steps', args.v2_train_steps]
                    for flag, value in [('--scenario', args.v2_scenario), ('--hdv-mixture', args.v2_hdv_mixture),
                                        ('--ablation', args.v2_ablation)]:
                        if value:
                            training_options += [flag, value]
                module = 'training.train_common_reward' if method in COMMON_LEARNED else 'training.train_'+MODULES[method]
                run(f'train_{method}', module, training_options, cwd)
            elif args.stage == 'train':
                print('Baselines are fixed policies: no training required.', flush=True)
            if args.stage in ['all', 'evaluate']:
                common_options = ['--method', method, '--scenario', args.common_scenario,
                    '--hdv-mixtures', args.common_hdv_mixture, '--max-steps', args.common_eval_steps,
                    '--eval-lane-demands', args.eval_demands, '--eval-prs', args.eval_prs]
                if args.common_eval_seeds:
                    common_options += ['--seeds', args.common_eval_seeds]
                if method in COMMON_FIXED:
                    # Fixed policies have no checkpoint; still use isolated outputs.
                    common_options += ['--output-dir', run_dir/'evaluation'/method/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')]
                    run(f'evaluate_{method}', 'evaluation.evaluate_common_reward', common_options, cwd)
                elif method == 'baselines':
                    run('evaluate_baselines', 'evaluation.evaluate_rule_based',
                        ['--eval-lane-demands', args.eval_demands, '--eval-prs', args.eval_prs], cwd)
                else:
                    model = model_location(args, method, run_dir)
                    if not args.dry_run:
                        validate_weights(method, model)
                        # Evaluate copied weights; evaluators write reports beside them.
                        copied = run_dir/'evaluation'/method/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                        copied.mkdir(parents=True, exist_ok=True)
                        if method == 'z_merge':
                            for suffix in ['_qnet.pth', '_paramnet.pth']:
                                shutil.copy2(f'{model}{suffix}', copied/f'zmerge{suffix}')
                            model = copied/'zmerge'
                        else:
                            checkpoints = [p for p in model.glob('checkpoint-*.pt') if p.stem[11:].isdigit()]
                            checkpoint = max(checkpoints, key=lambda p: int(p.stem[11:]))
                            shutil.copy2(checkpoint, copied/checkpoint.name)
                            model = copied
                    options = ['--model-prefix', os.path.relpath(model, cwd), '--episodes', args.zmerge_episodes] if method == 'z_merge' else [
                        '--model-dir', model, '--eval-lane-demands', args.eval_demands, '--eval-prs', args.eval_prs]
                    if method == 'nash_double_q':
                        options += ['--max-steps', args.nash_eval_steps]
                        if args.nash_eval_seeds:
                            options += ['--seeds', args.nash_eval_seeds]
                    if method == 'shepherd_v2':
                        options += ['--max-steps', args.v2_eval_steps]
                        for flag, value in [('--seeds', args.v2_eval_seeds), ('--scenario', args.v2_scenario),
                                            ('--hdv-mixtures', args.v2_hdv_mixture)]:
                            if value:
                                options += [flag, value]
                    if method in COMMON_LEARNED:
                        options = [*common_options, '--model-dir', model, '--output-dir', model/'reports']
                    module = 'evaluation.evaluate_common_reward' if method in COMMON_LEARNED else 'evaluation.evaluate_'+MODULES[method]
                    run(f'evaluate_{method}', module, options, cwd)
    if args.stage in ['all', 'evaluate', 'results']:
        run('results', 'analysis.combine_evaluation_reports',
            ['--search-dirs', run_dir, '--output-csv', run_dir/'results/combined_traffic_results.csv',
             '--output-xlsx', run_dir/'results/combined_traffic_results.xlsx'], ROOT)
        if any(m in COMMON_METHODS for m in args.methods):
            run('common_comparison', 'analysis.compare_common_reward',
                ['--search-dirs', run_dir, '--output-dir', run_dir/'results/common_comparison'], ROOT)
    print('Finished.', flush=True)


if __name__ == '__main__':
    main()
