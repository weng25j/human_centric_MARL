# Traffic Shepherd 2.0

Traffic Shepherd 2.0 is a separate implementation of the supplied **Traffic-Shepherd-Short-Paper-en.pdf** research plan. It implements sparse CAV control through five interpretable traffic roles, centralized training with decentralized execution (CTDE), and a reward covering efficiency, safety, stability, HDV fairness, and intervention cost. It uses real **SUMO and TraCI**, not a surrogate simulator.

For reward-matched comparisons against Z-Merge, Trust-MARL, MA2C, QMIX, MAAC and the existing controls, use the newer [common-reward suite](common_reward_comparisons.md): `python scripts/run_experiments.py --suite common`. It retrains `shepherd_v2_common` with the same actor/critic design and a scalar equal to the mean of this version's per-CAV rewards. The standalone workflow below remains available with its original reward/checkpoints.

The original Traffic Shepherd remains the original novel method. Its environment, MAPPO learner, training/evaluation programs and checkpoints are unchanged. Flat MAPPO, Z-Merge, Nash Double-Q and the fixed policies retain their implementations and settings. Nash Double-Q remains a documented SUMO adaptation. The PDF supplies research requirements; its hypotheses and expected outcomes are not treated as established results.

## Plan to implementation

| Plan requirement | 2.0 implementation |
| --- | --- |
| Sparse influence over HDVs | Commands only CAVs. Default five-CAV budget, with all HDVs and other CAVs under SUMO control. |
| Five interpretable roles | Non-intervention, gap creation, flow stabilization, conservative following and release are five separate policy outputs. |
| CTDE | Shared local actor; separate shared critic receives local observation plus network-wide pooled state during training only. |
| Partial observations and human uncertainty | Local leader/follower information, short temporal HDV acceleration histories, bounded V2X summaries, nearby CAV policy fingerprints. |
| Human-centric objectives | Team reward explicitly includes all five metric families, including HDV tail delay and intervention frequency. |
| Ramp and lane-drop geometry | Original ramp network by default; isolated, generated two-to-one lane-drop network for transfer tests. |
| Heterogeneous HDVs | Original cautious/normal/aggressive mixture by default; explicit cautious-heavy/aggressive-heavy options. |
| Ablations | Remove fairness, stability, control cost, or replace semantic actions with primitive accelerations. Each requires its own training run. |
| Reproducible evaluation | Existing default demand/PR grid, 20 seeds and 4,000-step horizon; per-seed metrics, HDV delay distributions, role traces, plots and CSV/Excel export. |

The plan leaves architectures, feature definitions, reward normalizers and optimizer details open. Values below are explicit 2.0 design choices, rather than claims of settings specified by the plan or any referenced publication. This is a research implementation; convergence and superiority require full experiments.

## Programs

| File | Purpose |
| --- | --- |
| `algorithms/traffic_shepherd_v2.py` | Shared actor, centralized critic, clipped PPO, per-vehicle GAE and versioned checkpoints. |
| `environments/traffic_shepherd_v2.py` | Labeled TraCI connection, observations, five role controllers and reward. |
| `environments/shepherd_v2_scenarios.py` | Isolated route generation and optional lane-drop network generation using `netconvert`. |
| `training/train_traffic_shepherd_v2.py` | Curriculum, rollout collection, learning, ablations and resume. |
| `evaluation/evaluate_traffic_shepherd_v2.py` | Checkpoint evaluation, metrics, role traces, HDV losses and plots. |
| `analysis/plot_shepherd_v2_roles.py` | CAV role/speed/gap/acceleration plots and completion-aware HDV delay distributions. |
| `tests/test_traffic_shepherd_v2.py` | Numerical/control/checkpoint regression tests and opt-in live SUMO checks. |

## Observations and decentralized execution

The actor receives **42 features**, normalized then clipped to [-10,10]:

| Features | Count | Encoding |
| --- | --- | --- |
| Ego | 6 | Speed/30, acceleration/5, remaining lane distance/1,000, lane index/3, signed distance to bottleneck/300 clipped to [-1,1], ramp-route flag. |
| Leader and follower | 8 | Each: speed/30, gap/150 capped at 1, relative speed/30, presence flag; search distance 150 m. |
| Local HDVs | 4 | Mean temporal acceleration standard deviation/5, cross-sectional speed standard deviation/10, fraction below 2 m/s, count/20. Neighborhood radius 100 m; temporal history five simulator samples. |
| V2X zones | 9 | Three longitudinal windows relative to ego: [-150,-50), [-50,50), [50,150) m, within 30 m laterally. Each contributes count/20, mean speed/30 and fraction below 2 m/s. |
| Neighbor fingerprints | 15 | Previous five-action probability vectors of the nearest three CAVs within 100 m. Missing neighbors are zero; unannounced policies use a uniform distribution. |

No driver-type label or future trajectory enters the observation. Uncertainty is an observable variability proxy, not a learned driver-intent model or calibrated trust probability. SUMO provides the bounded local/V2X observations; this assumes those measurements and CAV messages are available without delay or packet loss. Zone counts are normalized counts, not measured density in veh/km. Zones are geometric approximations rather than a full V2X protocol.

The critic receives these 42 features plus **12 centralized features**: vehicle count/100, CAV fraction, mean speed/30, speed standard deviation/10, mean acceleration/5, acceleration standard deviation/5, HDV mean and P90 accumulated time loss/100, stopped fraction, latest arrival count/5, colliding-vehicle count/5, and TTC<2.5 s fraction. These summaries are clipped to [-10,10]. They are never passed to `agent.act()`.

Variable vehicle populations do not occupy interchangeable fixed learning slots. Trajectories and GAE are keyed by vehicle ID. A vehicle leaving the five-CAV control set ends that local trajectory. Time limits and rollout boundaries bootstrap the current critic value; they do not connect to a new episode. Temporal histories advance once per simulator step, regardless of how often observations are read.

## Five role controllers

The actor selects a role every 0.1 s. Native SUMO lane changing stays active. Commands use `setAcceleration(..., duration=0.1)`; `speedMode=31` retains native safety checks. Requested role accelerations are bounded to [-3,1.5] m/s². Native safety responses may differ from requests and may brake harder.

| Index | Role | Low-level behavior |
| --- | --- | --- |
| 0 | Non-intervention | `setSpeed(-1)` clears previous longitudinal commands and returns to native SUMO car following. |
| 1 | Gap creation | Target `max(0, min(ego speed, leader speed)-2)` m/s, opening space ahead by slowing gently. |
| 2 | Flow stabilization | Track the mean speed of nearby HDVs; use leader speed when there are no nearby HDVs. |
| 3 | Conservative following | Desired gap `2.5+(1.8+u)*ego speed` m, with uncertainty `u` clipped to [0,1]. Track `leader speed+0.2*(actual gap-desired gap)`, capped by allowed speed. |
| 4 | Release | Track allowed speed to restore flow after a conflict. |

Roles 1–4 use `clip(0.5*(target speed-ego speed), -3,1.5)`. For roles 1 and 3, multiply by `1-0.5*u`, where `u` is local mean HDV temporal acceleration standard deviation divided by 3 and clipped to [0,1]. A braking-distance speed cap then limits the command using `sqrt(leader_speed²+2*3*max(0,gap-2.5))`. SUMO safety checks also remain enabled. These are controller heuristics, not a formal safety guarantee. Gap creation does not reserve a gap or force an HDV to merge.

Only selected CAVs receive commands. A previously selected CAV leaving the budget is explicitly returned to native control. Selection uses SUMO vehicle-ID order, up to five CAVs, to keep the existing control budget. `--max-controlled-cavs 0` explicitly changes this to all CAVs; record this difference when comparing results.

## Reward

Every acted CAV receives the same network-wide team reward, plus an extra -10 if that CAV collided, including if SUMO removed it immediately:

```text
R = 1.0 E - 10.0 S - 5.0 T - 0.5 F - 0.1 C - 10*own_collision
learner_reward = R / 20
```

At each step (`N=max(1, live vehicle count)`; empty means are zero):

- **E, efficiency:** mean speed/20 + arrivals/N − mean positive time-loss increment per second. SUMO time loss measures travel-time inefficiency; throughput and completed travel time are also measured during evaluation.
- **S, safety:** number of colliding vehicles + mean `max(0,(2.5−TTC)/2.5)` + fraction braking below −3 m/s². Infinite TTC has zero cost. Removed vehicles remain in collision accounting.
- **T, stability:** population acceleration variance/25 + mean absolute per-vehicle jerk/50 + mean five-sample per-vehicle speed variance/100. Newly observed vehicles have no jerk sample yet.
- **F, fairness:** `(HDV mean accumulated time loss + HDV P90 accumulated time loss)/100`, over live HDVs. Both terms discourage redistributing delay to the tail.
- **C, control cost:** sum over acted CAVs of `1 + abs(requested acceleration)/3` for active roles, divided by number of acted CAVs. Native non-intervention has zero requested cost; active zero-acceleration commands still incur intervention cost.

The time-loss reward is a population statistic with changing membership; it is not a causal counterfactual fairness guarantee. Collisions appear in the team safety cost and additionally in the involved CAV's terminal reward. The team signal can be sparse/noisy at low PR. The actor's access to only local information is intentional under CTDE.

The coefficients 1/10/5/0.5/0.1 and reward scale 20 carry over the original coefficient magnitudes, while 2.0 introduces the plan's fuller terms and normalizers. This reward belongs only to 2.0. Its changed information/actions/reward mean comparisons evaluate complete controllers, not a pure optimizer ablation.

## Learning defaults

Two separate MLPs with two 256-wide tanh hidden layers; actor outputs five logits, critic one scalar. Adam actor LR 3e−4, critic LR 1e−3, gamma 0.99, GAE lambda 0.95, PPO clip 0.2, entropy coefficient 0.01, gradient norm cap 0.5. Collect 256 simulator steps per on-policy rollout, then run four shuffled minibatch epochs of up to 256 agent samples. The last partial rollout updates at episode end. Old rollouts are discarded rather than placed in replay. PPO uses the clipped surrogate described by [Schulman et al.](https://arxiv.org/abs/1707.06347).

Training defaults: 3,000 episodes × 1,000 simulator steps, seed 42, checkpoints every 50 episodes and at the final episode. Curriculum is unchanged in values: before episode 1,000, demand 800 and PR 20%; episodes 1,000–1,999 sample 800/1,200 at PR 10%; subsequent episodes sample 800/1,200/1,700 and PR 5/10/20%. Per-episode traffic choices use seed+episode. This makes resumed episode choices reproducible without relying on unrelated global random calls.

## Setup and full workflow

Use the repository's existing Python 3.12 setup; there are **no new package requirements**:

```powershell
py -3.12 scripts/setup_environment.py
& .\.venv\Scripts\Activate.ps1
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '2'
python scripts/run_experiments.py --methods shepherd_v2 --run-dir runs/shepherd_v2_01
```

Linux: run `python3.12 scripts/setup_environment.py`, then `source scripts/activate_environment.sh`, followed by the same runner command. `sumo` must be on PATH; lane-drop generation also needs `netconvert`, supplied by a full SUMO installation. GUI mode needs `sumo-gui` and a display. The controller uses [documented TraCI vehicle commands](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html), and lane-drop inputs use [SUMO PlainXML/netconvert](https://sumo.dlr.de/docs/Networks/PlainXML.html).

The runner trains 2.0, evaluates a copy of the newest checkpoint, and writes CSV and Excel under `runs/shepherd_v2_01/results/`. All simulation inputs live in isolated workspaces. The no-flags command `python scripts/run_experiments.py` retains its previous five-method default list; **2.0 is opt-in**. To include everything:

```bash
python scripts/run_experiments.py --run-dir runs/all_with_v2 \
  --methods shepherd shepherd_v2 flat_mappo z_merge nash_double_q baselines
```

Preview commands without creating files with `--dry-run`. Relevant runner options are `--v2-model-dir`, `--v2-episodes`, `--v2-train-steps`, `--v2-eval-steps`, `--v2-eval-seeds`, `--v2-scenario`, `--v2-hdv-mixture`, and training-only `--v2-ablation`. Existing methods' flags/defaults are unchanged.

## Train, evaluate and resume separately

```bash
python -m training.train_traffic_shepherd_v2 --base-dir results_shepherd_v2 --seed 42
python -m evaluation.evaluate_traffic_shepherd_v2 \
  --model-dir results_shepherd_v2/models

python -m training.train_traffic_shepherd_v2 --base-dir results_shepherd_v2 \
  --resume results_shepherd_v2/models --seed 42 --episodes 4000
```

An evaluation directory defaults to a fresh UTC-timestamped subdirectory; an explicitly supplied `--output-dir` refuses to overwrite an existing evaluation. Training refuses to overwrite checkpoints without `--resume`. Loading requires a 2.0 method/version tag, correct tensor shapes and an existing numbered checkpoint; missing/wrong-method checkpoints fail. Original Shepherd checkpoints cannot initialize 2.0 because feature/action/network dimensions differ.

Checkpoints contain actor, critic, both optimizers, learner and experiment configurations, training episode/seed, update/step counters and Python/NumPy/PyTorch RNG states. Resume restores all of these at an episode boundary. Rollout buffers and live SUMO state are not stored. Resume must use the original training seed, max-step horizon, scenario, mixture, budget and ablation; incompatible requests fail. It is not mid-episode crash recovery. Only load trusted PyTorch checkpoint files.

## Lane-drop, behavioral transfer and ablations

The ramp default uses the existing `test.net.xml`, unchanged route generator, two mainline lanes and ramp demand 35% of per-lane demand. The lane-drop test is a **new** isolated network: 700 m two-lane approach, 300 m two-lane merge zone, 700 m single-lane exit. The right lane drops; only the left lane connects to the exit. It has mainline demand `2*demand` and no ramp flow. It is a geometry transfer experiment, not an identical-demand-throughput match to the ramp.

To evaluate a ramp-trained checkpoint on the lane drop without retraining:

```bash
python -m evaluation.evaluate_traffic_shepherd_v2 \
  --model-dir results_shepherd_v2/models --scenario lane_drop \
  --hdv-mixtures default cautious aggressive
```

Mixture probabilities (cautious/normal/aggressive): default 0.2/0.6/0.2, cautious-heavy 0.6/0.3/0.1, aggressive-heavy 0.1/0.3/0.6. Only the explicit new scenario options change mixtures or geometry. Original HDV parameter values are preserved. Evaluation inherits the checkpoint scenario, mixture, budget and ablation unless a supported transfer option is explicitly supplied. Evaluation never silently changes a checkpoint's reward/action ablation.

Train an ablation from scratch in a separate directory:

```bash
python -m training.train_traffic_shepherd_v2 \
  --base-dir runs/v2_no_fairness --ablation no_fairness --seed 42
```

Other choices are `no_stability`, `no_control_cost`, `primitive_actions` and `full`. The first three set only their respective reward coefficient to zero. Primitive actions replace the five roles with accelerations [-1.5,-0.75,0,0.75,1.5] m/s², retaining the same actor/critic, observations, team reward, control cost and SUMO safety mode. All five primitive commands are active interventions. Use matched independent training seeds and evaluation conditions to assess an ablation.

## Evaluation and outputs

Default evaluation uses the existing 20 seeds, 4,000 steps at 0.1 s, demands 800/1,700 veh/h/lane, PR 5/10/20%. Actor actions use argmax; the actual probability vector is still broadcast to local neighbors. Broad critic state is unnecessary at evaluation.

Each condition writes:

- `report_SHEPHERD_V2_*.txt`: means and 95% normal-approximation CIs over evaluation seeds, using existing report labels; scenario, mixture and ablation appear in the model label.
- `metrics_*.json`: raw per-seed metric records, requested-role counts/duty and observed HDV delay statistics.
- `roles_*_seedN.csv`: timestamp, CAV ID, selected role, requested acceleration, realized speed/acceleration, leader gap/TTC, local HDV uncertainty and collision flag.
- `hdv_delays_*_seedN.json`: last observed SUMO time loss for each HDV and arrived/collided/unfinished status. Unfinished values are censored by the horizon, not completed-trip delays.
- Speed/deceleration/jerk `.npy` arrays and trace plots, plus `evaluation_settings.json` and isolated SUMO inputs.

Standard metrics use the shared `MergingMetrics` collector and existing thresholds: TTC 2.5/1.5 s, hard braking −3 m/s², queue speed below 2 m/s. Standard intervention metrics count measured CAV braking below −0.1 m/s² across all CAVs, consistent with the existing protocol. **Requested Role Duty** separately counts active policy commands, including acceleration and active zero-acceleration actions, over controlled CAV decisions. Standard HDV delay metrics summarize arrived HDVs; observed-population HDV metrics include unfinished vehicles and are labelled separately. No-arrival means are zero under the existing protocol, so inspect arrivals and unfinished counts before interpreting short runs. The merge-success measure is ramp-specific; lane-drop reports must be assessed using throughput/safety/stability/delay rather than the ramp success percentage.

```bash
python -m analysis.combine_evaluation_reports --search-dirs results_shepherd_v2 \
  --output-csv reports/data/shepherd_v2.csv \
  --output-xlsx reports/data/shepherd_v2.xlsx
```

The exporter includes reports, numeric metrics, saved time series and plot indices. Full role CSVs and HDV JSON distributions remain adjacent to their reports. Separate experiments are kept separate, rather than averaged together.

Visualize the longest observed CAV trajectory with its selected roles, requested/realized acceleration, speed and merging gap; optionally show HDV time-loss distributions split by completion status:

```bash
python -m analysis.plot_shepherd_v2_roles \
  --trace-csv PATH/roles_SHEPHERD_V2_ramp_default_full_800vph_20PR_seed0.csv \
  --delay-json PATH/hdv_delays_SHEPHERD_V2_ramp_default_full_800vph_20PR_seed0.json
```

Replace `PATH` with an actual evaluation folder. `--vehicle CAV_ID` selects a particular CAV; `--output PATH.png` chooses the plot destination. Red lines mark collisions. Generated plots can be indexed by rerunning the results-export stage.

## Validation and interpretation

Run regression tests, including live SUMO checks, in PowerShell:

```powershell
$env:SHEPHERD_V2_SUMO_TESTS = '1'
python -m unittest discover -s tests -v
```

On Linux use `SHEPHERD_V2_SUMO_TESTS=1 python -m unittest discover -s tests -v`. The live test executes every role in both scenarios and checks finite observations/rewards. Other tests cover closed-form identity-aware GAE, terminal/truncation handling, actor/critic parameter updates, checkpoint and RNG restoration, actor locality, fingerprints, control budgeting/native release, removed collisions, reward ablations and runner isolation.

A short real training → checkpoint evaluation → export check:

```bash
python scripts/run_experiments.py --methods shepherd_v2 --run-dir runs/v2_smoke \
  --v2-episodes 1 --v2-train-steps 500 \
  --v2-eval-steps 1000 --v2-eval-seeds 0,25 \
  --eval-demands 800 --eval-prs 0.20
```

Use a fresh run directory. This is an execution/learning check, not a converged policy. A full 3,000-episode study, multiple independent training seeds, all evaluation conditions, and matched ablations are still needed to test the plan's hypothesis. Report failures as well as gains: improved throughput or mean speed alone does not establish success if TTC violations, hard braking, instability or HDV P90 delay worsen. Evaluation-seed CIs are not uncertainty across independent training runs. Current native safety checks, geometric V2X, synthetic HDVs and short uncertainty histories do not establish real-world safety or calibrated human behavior.

### Completed checks on 2026-10-26

- Windows, SUMO/TraCI 1.27.1, Python 3.10.20, CPU PyTorch 2.13.0 and NumPy 2.2.6. Validation used the existing `rllib_env` interpreter and cached test/export dependencies; the repository's Python 3.12 setup and dependency pins were not changed.
- **All 35 regression tests passed**, including the existing Nash Double-Q, Z-Merge and workflow/export tests, plus live SUMO checks for every role in both geometries.
- Final ramp smoke run: 500 training steps, **1,445 CAV transitions, 28 optimizer updates**, all five roles sampled; saved checkpoint evaluated for 1,000 steps on each of seeds 0 and 25 at 800 veh/h/lane and 20% PR.
- Final lane-drop smoke run: 500 training steps, **1,235 CAV transitions, 24 optimizer updates**; checkpoint evaluation completed 1,000 steps on seed 0 under the same demand/PR. SUMO reported HDV emergency-braking events; successful execution is not a safety or performance claim.
- All four ablations completed 120-step learning runs with four optimizer updates each. Reward-only ablations produced identical actions before their first update, verifying that changing their reward coefficient did not inadvertently change the initial policy's RNG stream.
- A resumed no-fairness run exactly matched the actor and critic tensors of an uninterrupted two-episode run in this bounded CPU/SUMO test. This is evidence for episode-boundary restoration in the tested setup, not a cross-platform determinism guarantee.
- A ramp-trained checkpoint also completed a 500-step lane-drop transfer check with the aggressive HDV mixture at 10% PR, with reports and role traces.
- Both final pipelines produced CSV and Excel, each with **3,000 numeric time-series values**; role/gap/acceleration plots and HDV delay distribution plots were generated and indexed. Models, metrics and workbooks remain locally under `runs/shepherd_v2_final_ramp/` and `runs/shepherd_v2_final_lane_drop/`; these generated artifacts are ignored by Git.
- The original algorithm/training/evaluation files, source SUMO XML and dependency pins are unchanged. Route preparation isolates the legacy learner's import-time RNG reset so it cannot overwrite 2.0 sampling or restored checkpoint RNG state. The regression test also verifies source XML bytes remain unchanged.
