# Human-Centric Multi-Agent Reinforcement Learning

**Traffic Shepherd is the main novel algorithm developed in this repository.** It studies human-centric control for mixed-autonomy highway on-ramp merging in SUMO, where connected and automated vehicles (CAVs) interact with human-driven vehicles (HDVs). Flat MAPPO, Z-Merge, and the All-HDV, Rule-CAV, and Selfish-CAV baselines are retained as comparison methods.

## Algorithms and entry points

Every learned method below has its own training and evaluation program. Run them from the repository root with `python -m` followed by the module name, without `.py`.

| Method | Algorithm implementation | Training program | Evaluation program |
| --- | --- | --- | --- |
| **Traffic Shepherd (main novel method)** | [Human-centric environment/controller](environments/traffic_shepherd.py) with the [MAPPO learner](algorithms/mappo.py) | [`training.train_traffic_shepherd`](training/train_traffic_shepherd.py) | [`evaluation.evaluate_traffic_shepherd`](evaluation/evaluate_traffic_shepherd.py) |
| Flat MAPPO / flat MARL | [MAPPO learner](algorithms/mappo.py) with the local-observation environment in its [trainer](training/train_flat_marl.py) | [`training.train_flat_marl`](training/train_flat_marl.py) | [`evaluation.evaluate_flat_marl`](evaluation/evaluate_flat_marl.py) |
| Z-Merge | [Parameterized DQN agent](algorithms/z_merge.py) and [Z-Merge environment](environments/z_merge.py) | [`training.train_z_merge`](training/train_z_merge.py) | [`evaluation.evaluate_z_merge`](evaluation/evaluate_z_merge.py) |
| Nash Double-Q (non-Chen comparison) | [Joint payoff critic and equilibrium solver](algorithms/nash_double_q.py) | [`training.train_nash_double_q`](training/train_nash_double_q.py) | [`evaluation.evaluate_nash_double_q`](evaluation/evaluate_nash_double_q.py) |
| All-HDV, Rule-CAV, Selfish-CAV | [Non-learning baseline policies](algorithms/rule_based.py) | No training: fixed policies with no learned weights | [`evaluation.evaluate_rule_based`](evaluation/evaluate_rule_based.py), which evaluates all three |

Traffic Shepherd's algorithm is split across its controller/environment and the shared MAPPO learner. Its human-centric observations, human-predictability tracking, neighbor fingerprints, semantic interventions, and reward calculations remain in `environments/traffic_shepherd.py`. Sharing the optimizer implementation with flat MAPPO does not replace Traffic Shepherd: the Shepherd workflow uses 14 observation features and 3 semantic actions, while flat MAPPO uses 6 features and 5 acceleration actions. Their training and evaluation programs remain separate.

The reorganization preserves the logic of the original four method families and their existing checkpoint formats. The separate experimental MA2C implementation and training prototype are also retained in `algorithms/ma2c.py` and `experiments/train_ma2c.py`; their integration remains experimental, as described below.

Author: Justin Weng, CUHK. The [project report](reports/documents/1155282333_Justin%20Weng_A%20Human-Centric%20MARL.pdf) and [poster](reports/documents/poster_JUSTIN_WENG_CUHK.pdf) describe the research. This is research code; the experimental prototypes have separate limitations listed below.

## Traffic Shepherd 2.0

[Traffic Shepherd 2.0](docs/traffic_shepherd_v2.md) implements the original research plan as a separate SUMO/TraCI controller: five semantic roles, a local shared PPO actor with a centralized training critic, full efficiency/safety/stability/fairness/control-cost rewards, optional lane-drop and HDV-mixture transfer tests, and reward/action ablations. It has independent [training](training/train_traffic_shepherd_v2.py), [evaluation](evaluation/evaluate_traffic_shepherd_v2.py), checkpoints and regression tests. The original Traffic Shepherd and every existing comparison retain their implementations and settings.

After environment setup, train, evaluate and export 2.0 with:

```bash
python scripts/run_experiments.py --methods shepherd_v2 --run-dir runs/shepherd_v2_01
```

The existing no-flags runner keeps its previous default methods; 2.0 is opt-in. See the [2.0 guide](docs/traffic_shepherd_v2.md) for architecture, exact features/reward, commands, resume, ablations, outputs, live SUMO tests and research limitations.

## New game-theoretic comparison

The [Nash Double-Q merging adaptation](docs/nash_double_q.md) adds a structurally different, non-Chen baseline based on Li et al.'s 2024 interactive-merging publication. It learns two vehicles' joint-action payoff matrices and selects a Nash equilibrium, including mixed strategies when necessary. It is not a flat MARL or PPO variant. The linked review explains the literature sources, equations, declared SUMO adaptations, settings, fairness limits, and the distinction from an exact paper reproduction.

```bash
# Refresh dependencies for the Nash equilibrium solver, then activate as below.
python3.12 scripts/setup_environment.py
source scripts/activate_environment.sh
python scripts/run_experiments.py --methods nash_double_q --run-dir runs/nash_01
```

Existing Traffic Shepherd, flat MAPPO, Z-Merge and fixed-baseline implementations remain unchanged. Nash Double-Q is now included in the full pipeline by default; pass `--methods` explicitly to select only particular methods.

## Repository guide

| Location | Contents |
| --- | --- |
| `algorithms/` | `mappo.py`: shared MAPPO implementation; `z_merge.py`: parameterized DQN; `rule_based.py`: baseline policies; `ma2c.py`: experimental actor-critic agent. |
| `environments/` | `traffic_shepherd.py`: 14-feature human-centric controller; `z_merge.py`: 22-feature Z-Merge environment; `pettingzoo_on_ramp.py`: experimental parallel environment. |
| `training/` | Curriculum training entry points for Traffic Shepherd, flat MARL, Z-Merge, and Nash Double-Q. |
| `evaluation/` | Checkpoint evaluation and metric collection; non-learning baseline benchmarks. |
| `recording/` | SUMO GUI playback and MP4 recording for Shepherd and flat MARL. |
| `analysis/` | Report aggregation, summary tables, comparison plots, and poster figures. |
| `common/` | Replay buffers, neural networks, tensor helpers, legacy agent bases, and K-FAC optimizer. `mappo_*` and `ma2c_*` identify the implementation they support. |
| `experiments/` | Earlier MA2C/RLlib training prototypes, GUI demos, an import check, and an incomplete HDV-control sketch. |
| `scripts/` | Computer setup/activation, settings inspection, full experiment pipeline, and legacy resume/recording helpers. |
| `SUMO_network/` | Network, route, simulation configuration, and GUI settings XML. |
| `results/` | Existing Traffic Shepherd checkpoints/evaluations and non-learning baseline reports. |
| `results_dongchen_baseline/` | Existing flat MARL checkpoints/evaluations, also labelled `DONGCHEN` in output filenames. |
| `results_zmerge/` | Existing paired Z-Merge network weights and evaluation reports. |
| `results_nash_double_q/` | New Nash Double-Q checkpoints, per-seed evaluation metrics, reports and traces. |
| `reports/documents/`, `reports/figures/` | Research report/poster documents and previously exported presentation figures. |
| `graphs_output/`, `comparison_charts/` | Existing generated metric and policy comparison charts. |
| `combined_traffic_results.csv`, `combined_traffic_results.csv.xlsx` | Aggregated evaluation data and an existing spreadsheet export. |

See the [complete file migration guide](docs/file_migration.md) for old-to-new filenames. Run entry points as modules, for example `python -m training.train_traffic_shepherd`, rather than running files inside these folders directly. All commands below assume your working directory is the repository root; relative network, model, and output paths depend on it.

## Computer environment setup

Install **Python 3.12**, Git, and enough free disk for several GB of dependencies plus experiment outputs. CPU training is supported; a GPU is optional. On Ubuntu/Debian, the system packages below supply virtual-environment support and OpenCV's shared libraries (install Python 3.12 using your OS-supported method if unavailable):

```bash
sudo apt-get install python3.12-venv libgl1 libglib2.0-0
git clone https://github.com/weng25j/human_centric_MARL.git
cd human_centric_MARL
python3.12 scripts/setup_environment.py
source scripts/activate_environment.sh
python scripts/show_settings.py
```

If the repository is already cloned, start with the setup command in that checkout. The setup script creates `.venv`, installs the official CPU build of PyTorch, all original pinned libraries, SUMO, and the Excel writer, then checks dependencies, the simulator, and training imports. It can be rerun to refresh the same virtual environment. It does not install OS packages or change algorithm settings.

On Windows PowerShell:

```powershell
py -3.12 scripts/setup_environment.py
& .\.venv\Scripts\Activate.ps1
$env:MPLBACKEND = "Agg"
$env:OMP_NUM_THREADS = "2"
$env:PYTHONDONTWRITEBYTECODE = "1"
python scripts/show_settings.py
```

Linux execution is verified. Windows commands are provided, but Windows/GUI execution has not been tested here; the pinned simulator/PyTorch distributions must support your OS and architecture. GUI recording additionally needs `sumo-gui` and a graphical display. The default CPU setup avoids CUDA downloads. For GPU use, select a compatible official PyTorch index with `--torch-index-url`; driver/GPU operation is not validated by this setup.

For an existing environment, use `python scripts/setup_environment.py --venv PATH`. In the prepared cloud machine, that path is `/workspace/venvs/human-centric-marl`. Set `MARL_VENV` to the same path before sourcing `scripts/activate_environment.sh`. The original UTF-16 `requirements.txt` is read directly; extra simulator/Excel requirements are in `requirements-tools.txt`.

See [current settings and environment variables](docs/current_settings.md) for full trainer CONFIG values, effective learner settings, evaluator seeds/horizons, and every supported workflow variable. `python scripts/show_settings.py` reports their actual values in your current shell. No credentials or API keys are needed.

## Run all methods and export a spreadsheet

From the repository root after setup, launch **all supported training, evaluation, and spreadsheet-export stages** with:

```bash
source scripts/activate_environment.sh
python scripts/run_experiments.py
```

No method or stage flags are required. A new timestamped folder under `runs/` stores the entire run. On Windows, activate `.venv` with PowerShell as shown above, then use the same Python command.

To preview the commands or choose a named output folder:

```bash
python scripts/run_experiments.py --run-dir runs/experiment_01 --dry-run
python scripts/run_experiments.py --run-dir runs/experiment_01
```

This serially trains and evaluates **Traffic Shepherd**, **flat MAPPO**, **Z-Merge**, and **Nash Double-Q**, evaluates **All-HDV, Rule-CAV, and Selfish-CAV**, then combines evaluation outputs into:

- `runs/experiment_01/results/combined_traffic_results.xlsx`
- `runs/experiment_01/results/combined_traffic_results.csv`

Training retains the existing 3,000-episode defaults and can be lengthy. Baselines are fixed policies and do not need training. The experimental MA2C/RLlib prototypes are excluded because their integration remains incomplete. Omit `--run-dir` for a new timestamped `runs/` directory.

The runner copies SUMO inputs into per-method workspaces, keeping the checkout's route files intact. It records commands/output in `logs/` and settings in `settings.json` and `pipeline_options.json`. Model copies are evaluated in new timestamped folders so evaluation output does not overwrite original weights. Existing trained model directories block retraining into the same run; choose a fresh directory. The scripts stop on errors, including missing checkpoints.

Separate stages and selected methods:

```bash
python scripts/run_experiments.py --stage train --run-dir runs/experiment_02
python scripts/run_experiments.py --stage evaluate --run-dir runs/experiment_02
python scripts/run_experiments.py --stage results --run-dir runs/experiment_02

# Only the main novel method, with a narrower evaluation grid:
python scripts/run_experiments.py --methods shepherd --run-dir runs/shepherd_01 \
  --eval-demands 800 --eval-prs 0.10
```

`evaluate` automatically exports results. To use existing weights, supply `--shepherd-model-dir`, `--flat-model-dir`, `--nash-model-dir`, and/or `--zmerge-model-prefix` with `--stage evaluate` and matching `--methods`. These paths are relative to the shell where you start the script; the runner makes them absolute. Use `--python PATH` if the intended virtual environment is not active. `--help` lists all options. The underlying evaluation seeds and step counts remain unchanged; narrowing the grid still runs all 20 MAPPO/baseline seeds.

To combine **all existing evaluation data** without starting training:

```bash
python -m analysis.combine_evaluation_reports \
  --search-dirs results results_dongchen_baseline results_zmerge results_nash_double_q \
  --output-csv reports/data/combined_traffic_results.csv \
  --output-xlsx reports/data/combined_traffic_results.xlsx
```

Workbook sheets contain a report-level summary, numeric metrics with separate confidence intervals, complete original report text, saved speed/deceleration/jerk time-series values, and an index of plots/recordings/arrays. Source paths preserve provenance. Large time-series exports split across Excel sheets automatically; `--no-arrays` exports summaries and an artifact index without embedding the values. Different runs are retained separately rather than averaged. Unreported Z-Merge metadata stays blank, and average queue length is kept separate from average maximum queue length. Raw per-seed values absent from saved reports cannot be reconstructed. Reports in nonstandard folders can be included through `--search-dirs`.

## Training

The original three trainers use curriculum traffic demand and CAV penetration rate (PR). Demand is vehicles/hour/lane; PR is a fraction such as `0.10` for 10%. The current defaults train for 3,000 episodes. Edit each module's `CONFIG` dictionary to change training length, rollout size, learning rates, or checkpoint interval. These settings are not CLI flags for the original trainers. Nash Double-Q additionally exposes `--episodes` and `--max-steps`; its learner settings are listed in the [method guide](docs/nash_double_q.md).

```bash
# Traffic Shepherd: 14 observation features, 3 semantic actions.
python -m training.train_traffic_shepherd --seed 42 --base-dir ./results/

# Flat MARL: 6 observation features, 5 acceleration actions.
python -m training.train_flat_marl --seed 42 --base-dir ./results_dongchen_baseline/

# Z-Merge: 22 observation features, parameterized discrete/continuous actions.
python -m training.train_z_merge --seed 42 --base-dir ./results_zmerge/
```

Add `--gui` to watch SUMO, at a substantial runtime cost. Use `--help` on any of these modules to list supported arguments.

Output conventions inherited from the original runners:

| Trainer | Run directory under `--base-dir` | Saved models |
| --- | --- | --- |
| Traffic Shepherd | `MAPPO_Curriculum_seed_0809/` (fixed name) | `models/checkpoint-<episode>.pt` |
| Flat MARL | `0809/` (fixed name) | `models/checkpoint-<episode>.pt` |
| Z-Merge | `ZMerge_Curriculum_seed_<seed>_<timestamp>/` | `models/zmerge_ep_<episode>_qnet.pth` and `_paramnet.pth`; final pair uses `zmerge_final`. |

Choose a fresh, **already existing** base directory for independent runs to avoid overwriting the fixed MAPPO run names. For example, `mkdir -p results/my_run` and use `--base-dir ./results/my_run/`. `--seed` does not change the fixed Shepherd output directory name.

To resume a MAPPO policy:

```bash
python -m training.train_traffic_shepherd \
  --model-dir ./results/MAPPO_Curriculum_seed_0809/models \
  --start-episode 3000 --seed 42
```

Set `--start-episode` to the checkpoint's episode count and increase `CONFIG['MAX_EPISODES']` if continuing beyond 3,000. The loader chooses the highest-numbered checkpoint in the **directory**. It restores actor weights and, in training mode, the actor optimizer; it does not restore the complete training state, critic, or rollout memory. Flat MARL accepts the same resume arguments. For Z-Merge, `--model-dir` accepts the prefix of the saved Q/parameter network pair, either as an explicit path or relative to `results_zmerge/`. Both network weights are restored, but replay, optimizers, exploration history, and the gradient-step counter are not; this is a weights-only warm start.

The Shepherd training entry point always calls training; its legacy `--option evaluate` argument does **not** switch to evaluation. Use its dedicated evaluator below. Z-Merge's `--option evaluate --model-dir PREFIX` now delegates to its dedicated evaluator, and flat MARL also supports `--option evaluate`.

**Generated inputs:** trainers and evaluators overwrite route XML in `SUMO_network/`. Run experiments serially in a checkout, or use separate copies for concurrent runs. Back up local route edits first. `test.net.xml` is the network used by these programs; `updated_highway.net.xml` has different edge IDs and is not interchangeable with their route files.

## Evaluation

Use a model directory containing compatible weights. Shepherd and flat MARL use the same checkpoint naming format but different neural-network dimensions; their checkpoints are not interchangeable. Pass the directory, not an individual `.pt` file: the shared loader scans for the highest-numbered `checkpoint-*.pt`.

```bash
python -m evaluation.evaluate_traffic_shepherd \
  --model-dir ./results/MAPPO_Curriculum_seed_0809/models \
  --eval-lane-demands 800,1700 --eval-prs 0.05,0.10,0.20

python -m evaluation.evaluate_flat_marl \
  --model-dir ./results_dongchen_baseline/0809/models \
  --eval-lane-demands 800,1700 --eval-prs 0.05,0.10,0.20

# All-HDV, Rule-CAV, and Selfish-CAV; no trained model required.
python -m evaluation.evaluate_rule_based \
  --eval-lane-demands 800,1700 --eval-prs 0.05,0.10,0.20
```

These evaluators use 20 predefined seeds and 4,000 steps per seed at 0.1 seconds/step. Narrow the grid, for example `--eval-lane-demands 800 --eval-prs 0.10`, for a smaller run; this still runs all 20 seeds. All-HDV runs only at 0% PR. The MAPPO evaluators write reports, plots, and speed/deceleration/jerk `.npy` arrays into the model directory. Rule-based reports go to `results/baselineFinal/`. Metrics include throughput, speed, travel time, collision and TTC measures, braking, jerk, HDV delay, merge success, queues, and intervention cost.

Z-Merge accepts an **explicit model prefix or a prefix relative to `results_zmerge/`**, without the weight suffix:

```bash
# Replace RUN_DIRECTORY with an actual directory in results_zmerge/.
python -m evaluation.evaluate_z_merge \
  --model-prefix RUN_DIRECTORY/models/zmerge_final --episodes 5
```

For that example, both `results_zmerge/RUN_DIRECTORY/models/zmerge_final_qnet.pth` and `..._paramnet.pth` must exist. This evaluator runs 1,000 steps per episode using the existing `highway_onramp_actual.rou.xml` and randomized SUMO seeds. It has no demand/PR grid flags and saves a text summary under `results_zmerge/`.

The MAPPO loader returns `False` when no checkpoint is found, and some runners continue with initialized weights. Check for the **`Checkpoint loaded:`** message before treating results as trained-policy evaluation. Use checkpoints you trust.

## Recording and analysis

Recording requires a graphical session. The dedicated recorders accept model directories, demand/PR lists, and use predefined seeds:

```bash
python -m recording.record_traffic_shepherd \
  --model-dir ./results/MAPPO_Curriculum_seed_0809/models --demand 800 --pr 0.10
python -m recording.record_flat_marl \
  --model-dir ./results_dongchen_baseline/0809/models --demand 800 --pr 0.10
```

MP4 files go into `recordings/` under the model directory. The comprehensive MAPPO evaluators also accept `--gui` and `--record-video`. Existing `temp_frames_*` directories contain historical intermediate screenshots.

```bash
# Select additional folders using --search-dirs as needed.
python -m analysis.combine_evaluation_reports
# Uses combined_traffic_results.csv:
python -m analysis.render_results_table
python -m analysis.plot_policy_comparisons
python -m analysis.plot_evaluation_metrics
```

The table goes to `reports/figures/comprehensive_results_table.png`; comparison charts go to `comparison_charts/`, and comprehensive metric plots to `graphs_output/`. The exporter writes a CSV and an Excel workbook. Its default search roots include the existing results directories plus `results_nash_double_q/`, including Z-Merge; use `--search-dirs` for custom runs. Default root output names are overwritten when explicitly rerun; use a separate output path to preserve earlier exports.

Other analysis modules are specialized: `plot_training_rewards` expects `results/episode_rewards.npy`; `plot_safety_benchmarks` expects per-policy/per-seed arrays; `plot_speed_comparison` has explicit input path constants; `plot_success_vs_intervention` uses hardcoded poster values rather than recomputing experiment results. Inspect those inputs before using them for a new study.

## Prototypes and validation

`experiments/` preserves earlier work separately from the main workflows:

- `train_ma2c.py`: experimental MA2C/PettingZoo integration.
- `train_rllib_ppo.py`: incomplete RLlib PPO example with a missing `marl_on_ramp_env` module and placeholder network/route paths.
- `run_random_on_ramp.py`: GUI random-action demo with legacy root-relative XML paths.
- `watch_z_merge.py`: GUI playback with a hardcoded `zmerge_v1` weight prefix.
- `hdv_control_sketch.py`: incomplete example requiring `my_custom_env` and a prediction agent.
- `check_rllib_imports.py`: optional dependency import check, not an automated test suite.

The relocation preserves algorithms and checkpoint formats. Validation covers module imports, CLI help, bounded training runs, and representative headless simulations; it does not establish completion of full curriculum training, the full evaluation grid, or GUI recording. Focused Z-Merge regression tests are available with `python -m unittest discover -s tests -v`. See the [paper comparison and validation notes](docs/z_merge_review.md) for the corrected Double PDQN update, preserved experiment settings, and remaining differences from the paper. Existing Z-Merge weight files still load, but need retraining for results from the corrected learner. Full curriculum convergence is not established by these checks.
