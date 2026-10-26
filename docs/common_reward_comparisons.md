# Common-reward research comparisons

This suite trains and evaluates **Traffic Shepherd 2.0 and ten comparisons under one SUMO/TraCI reward contract**. It adds the three methods referenced in the supplied Traffic Shepherd plan and two further paper-based MARL methods, QMIX and MAAC. The original Traffic Shepherd, flat MAPPO, Z-Merge, Nash Double-Q and fixed-policy programs/checkpoints remain available with their original settings.

These are independent **common-reward SUMO adaptations of published algorithm mechanisms**, not claims to reproduce the papers' complete experiments, reported numbers or original reward functions. Standardizing the reward deliberately changes the referenced methods' original tasks. Trust-MARL's payoff coefficients and simulator-specific choices are especially important limitations, detailed below. Nash Double-Q continues to be labelled an adaptation.

## Methods and original sources

| Method key | Original source | Implemented mechanism | Training / evaluation modules |
| --- | --- | --- | --- |
| `shepherd_v2_common` | Supplied Traffic Shepherd research plan | Five semantic roles, local PPO actor, centralized critic | `training.train_traffic_shepherd_v2_common` / `evaluation.evaluate_traffic_shepherd_v2_common` |
| `flat_mappo_common` | Existing flat MAPPO comparison | Six-feature local actor, centralized PPO value learning, five primitive accelerations | `training.train_flat_marl_common` / `evaluation.evaluate_flat_marl_common` |
| `z_merge_common` | Ibork, Won and Das, [Z-Merge (2025)](https://arxiv.org/html/2511.14910v1) | Parameterized Double DQN, measured 42-feature local/zone state, hybrid lane/acceleration/gap actions | `training.train_z_merge_common` / `evaluation.evaluate_z_merge_common` |
| `trust_marl_common` | Pan, Wang, Claudel and Shi, [Trust-MARL (2025)](https://arxiv.org/abs/2506.12600) | LSTM, directed dynamic trust, trust-triggered lane games and hybrid MASAC | `training.train_trust_marl` / `evaluation.evaluate_trust_marl` |
| `ma2c_common` | Chu, Wang, Codecà and Li, [MA2C (2019/2020)](https://arxiv.org/html/1903.04527) | Recurrent decentralized A2C, neighborhood fingerprints and spatially discounted neighbor observations | `training.train_ma2c` / `evaluation.evaluate_ma2c` |
| `qmix_common` | Rashid et al., [QMIX, ICML 2018](https://proceedings.mlr.press/v80/rashid18a.html) | GRU utilities, state-conditioned monotonic mixer and joint TD learning | `training.train_qmix` / `evaluation.evaluate_qmix` |
| `maac_common` | Iqbal and Sha, [Actor-Attention-Critic, ICML 2019](https://proceedings.mlr.press/v97/iqbal19a.html) | Multihead action-conditioned attention critic, soft policy learning and counterfactual credit assignment | `training.train_maac` / `evaluation.evaluate_maac` |
| `nash_double_q_common` | Existing documented Nash Double-Q adaptation | Joint payoff critics, verified Nash selection, online selection / target valuation | `training.train_nash_double_q_common` / `evaluation.evaluate_nash_double_q_common` |
| `all_hdv_common`, `rule_cav_common`, `selfish_cav_common` | Existing fixed comparison rules | Fixed policies under the shared actuator/reward protocol; no training | `evaluation.evaluate_common_reward --method METHOD` |

QMIX and MAAC are the two additional methods selected for this repository. They are established research algorithms, rather than newly published 2026 methods. They provide different coordination mechanisms: factorization of a cooperative team value versus selective attention to other agents' observations/actions. Both fit sparse cooperative vehicle control without making either an alias of Shepherd's PPO optimizer. Their original benchmarks were not this highway, so the vehicle observation/action mappings are explicit adaptations.

## One reward, including Shepherd 2.0

`common/benchmark.py` is the single reward contract, named `shepherd_common_reward_v1`. Every method receives the same scalar from `CommonRewardMergeEnv`; learners divide it by **20 exactly once**:

```text
R_team = E - 10*S - 5*T - 0.5*F - 0.1*C
         - 10 * colliding_acted_CAVs / max(1, acted_CAVs)
```

The E/S/T/F/C definitions are the exact functions from [Traffic Shepherd 2.0](traffic_shepherd_v2.md): speed/arrivals/time loss; collisions/TTC/hard braking; acceleration variance/jerk/speed oscillation; HDV mean/P90 time loss; and requested intervention frequency/magnitude. The shared scalar is mathematically the mean of the standalone 2.0 per-CAV rewards, including collision penalties. All agents receive that scalar, and QMIX receives it once per joint transition rather than multiplying it by the agent count.

The standalone 2.0 trainer retains its original individual extra collision penalty. For a reward-matched study, retrain **`shepherd_v2_common`** and the comparisons together. Legacy model files cannot be loaded as common-reward checkpoints. No legacy environment, reward function, network input, checkpoint or training hyperparameter was rewritten.

Trust does **not** blend self/cooperative learning rewards in this suite. MA2C does **not** redistribute rewards spatially. Those changes would violate the requested common objective. Trust/game reasoning still shapes Trust-MARL's policy; MA2C still uses recurrent local actor-critic updates and spatial/fingerprint inputs. Entropy regularization and value-factorization losses remain method-specific learning mechanisms, not alterations to the external reward function.

The contract hashes the reward source and coefficients. Checkpoints and evaluation metadata store this fingerprint, reward scale and protocol. Wrong-method, legacy or incompatible-reward checkpoints fail visibly. Reward equality is tested on synthetic transitions including removed collisions, and live SUMO tests check identical scalar rewards across simultaneously controlled CAVs.

## Common traffic and actuation

- Same existing ramp network and route generator, in isolated workspaces; no source XML is overwritten. Default cautious/normal/aggressive HDV mixture remains 0.2/0.6/0.2.
- Five controlled CAV slots. Slots persist by vehicle ID; a departing vehicle's successor starts a fresh history. HDVs and unselected CAVs remain under SUMO control.
- 0.1 s step, longitudinal `speedMode=31`, native safety checks and native lane changing. Explicit lane requests also retain lane-change safety mode 1621.
- Requested accelerations in [-3,1.5] m/s², for every learned or fixed comparison. Native SUMO responses can differ, including emergency braking. This is a heuristic simulator safety protocol, not a safety guarantee.
- Fixed Rule-CAV retains the 25 m gap and 15 m/s smoothing targets; Selfish-CAV targets allowed speed. Both targets use the common bounded actuator, so these are explicitly labelled common-protocol versions of the fixed rules. All-HDV uses 0% PR and no learned commands.
- Same 3,000-episode, 1,000-step training defaults and demand/PR curriculum values. Early episodes use 800 veh/h/lane and 20% PR; the existing 800/1,200 and then 800/1,200/1,700 demand stages are retained.
- Default evaluation: demands 800/1,700, PR 5/10/20%, existing 20 seeds, 4,000 steps per seed. All-HDV is a separate 0% composition. Lane-drop and mixture transfer remain explicit options.

Same rewards and traffic do not make information/action spaces identical. This compares complete controllers; a pure optimizer ablation would additionally need matched observations and actions. Do not describe these results as exact paper reproduction or as an algorithm-only ablation.

## What each new learner does

### Z-Merge common

`algorithms/z_merge_common.py` reuses the corrected action-conditioned Q/parameter network definitions while leaving the original Z-Merge workflow intact. Its 42 inputs contain four ego features, six surrounding vehicles × five features, three road zones × mean speed/count, and two queue counts. These are measured from SUMO, not placeholder zone/neighbor values. The zones map to the repository's existing pre-merge, merge and ramp edges; distances use SUMO Cartesian coordinates rather than geographic Haversine distances.

Actions are left, right, parameterized acceleration, parameterized desired gap [5,20] m, and native non-intervention. Illegal edge-boundary lane requests are masked in execution and in the next-state greedy selection. Double PDQN selects using both online networks and values that selected action using both target networks. The parameter network optimizes the sampled discrete action's Q while critic weights are frozen. Removed vehicles retain terminal replay entries. Continuous acceleration is mapped to the common bounds, rather than the paper's wider range. V2X is supplied as simulator observations, with no MOSAIC network/latency model. The paper's six-component reward is replaced by the common reward.

### Trust-MARL common

`algorithms/trust_marl.py` implements a shared LSTM hybrid actor and two centralized soft critics. The actor emits a tanh Gaussian longitudinal acceleration and categorical keep/left/right probabilities; training accounts for the continuous transformation's density correction and enumerates the three ego lane choices. Target actors/critics and off-policy replay support MASAC updates. Five-step local observation histories contain 42 base features plus three neighbor trust scores.

Directed trust starts at 0.5 and follows `T_next=0.4*T+0.6*perceived_cooperation`, with threshold 0.4. The article leaves perceived cooperation task-dependent: this adaptation uses observed braking/TTC/lane behavior, with no access to the HDV type label. Cooperative CAV interactions use joint-utility lane-game selection; other interactions use a verified mixed-strategy two-player game with an observed-response prior. Game log-priors modulate the actor's lane logits and are stored with replay; executed actions therefore match the policy being trained.

Important fidelity limit: the paper does not supply fitted payoff regression coefficients or a complete executable MASAC specification. `algorithms/trust_game.py` declares a one-second kinematic payoff proxy and a response prior. It does **not** reproduce the paper's fitted Stackelberg/Harsanyi payoffs or original trust-weighted reward. This comparator implements its temporal/trust/game/MASAC structure under the requested shared objective; it must be reported as a SUMO adaptation, not a verified Trust-MARL reproduction.

### MA2C common

The recurrent branch of `algorithms/common_on_policy.py` is decentralized A2C, not PPO. Separate actor/value LSTMs receive local observations, neighboring policy fingerprints and three neighbor ego vectors spatially discounted by alpha=0.75. There are 60 inputs, 64 recurrent units, five primitive longitudinal actions and 20-step on-policy updates. Lambda=1 produces bootstrapped n-step returns; there is no PPO clipping or replay of old policies.

The original MA2C paper controls traffic lights, uses fixed intersection agents and spatially discounted local rewards. Here the agents are moving CAVs with changing neighborhoods and bounded five-sample histories; parameters are shared across interchangeable vehicles. The common global reward is retained without spatial reward redistribution. These are necessary, documented adaptations to vehicle control and reward equality. The separate earlier MA2C prototype remains under `algorithms/ma2c.py` and `experiments/`.

### QMIX

`algorithms/qmix.py` learns a GRU utility for each controlled CAV from five samples of local observation plus its own previous action (47 inputs). A centralized hypernetwork generates nonnegative mixer weights from 12 global features and five presence flags. The monotonic mixer combines active-agent utilities into one joint Q, so independent greedy action choices maximize that joint estimate. Learning minimizes joint TD error with a delayed utility/mixer target; empty next teams do not bootstrap.

The adaptation uses shared utilities for homogeneous CAVs, stable masked slots for changing vehicle populations and bounded five-step recurrent windows instead of full StarCraft episode histories. All CAVs choose the same five primitive accelerations. The original monotonicity/centralized-training/decentralized-execution mechanism is retained. Numerical tests check mixer derivatives, inactive masking, greedy joint consistency and closed-form bootstrapping.

### MAAC

`algorithms/maac.py` implements four-head, action-conditioned attention over other active agents' observation/action embeddings. Each ego query is evaluated for every possible ego action, with self and absent vehicles excluded. Centralized soft critic learning uses target policies/critics; decentralized actors use an exact discrete policy expectation and counterfactual baseline holding other actions fixed. A singleton receives a zero neighbor context, avoiding undefined attention.

The adaptation uses one shared discrete actor/attentive critic for homogeneous CAVs, 42 local input features, five primitive accelerations and dynamic presence masks. Target networks use soft updates. Tests verify that another agent's action affects the ego critic, the ego's replayed action does not contaminate its counterfactual enumeration, and inactive/self attention is zero.

## Settings and checkpoints

`scripts/show_settings.py` reads `BenchmarkConfig` and `CONFIG_OVERRIDES` from source. Every training run saves the effective config in `training_settings.json` and its checkpoint.

| Setting | Default |
| --- | --- |
| Gamma / reward scale | 0.99 / 20 for all methods |
| Default hidden width | 64; common PPO 256; Nash 128; Z-Merge fixed [256,512,512,128] |
| Actor / critic Adam LR | 3e-4 / 1e-3; Z-Merge both 1e-4; Nash uses 3e-4 |
| Off-policy replay / batch / learning starts | 50,000 joint entries / 64 / 256 |
| Off-policy learning frequency | Every four SUMO steps after warm-up |
| QMIX / Nash hard target interval | 500 updates |
| Z-Merge hard target interval | 35,000 updates |
| MAAC / Trust soft target tau | 0.01 |
| MAAC / Trust entropy temperature | 0.1 |
| Gradient norm cap | 0.5 |
| Common PPO | Two 256-wide tanh layers, 256-step rollout, batch 256, four epochs, clip 0.2, entropy 0.01, GAE lambda 0.95 |
| MA2C | 64-unit LSTM, 20-step rollout, one update, entropy 0.01, n-step lambda 1 |
| Epsilon for Z-Merge/QMIX/Nash | Starts 0.99, episode decay 0.995, minimum 0.05; policy-gradient learners use their own stochastic policies |

Episode-boundary checkpoints store every online/target network, optimizer, effective config, seed, episode, update/step/transition counters and Python/NumPy/PyTorch RNG states. Replay is intentionally not stored: resumed off-policy runs refill replay before further learning and are not bit-identical replay continuations. Temporal histories reset at episode boundaries. On-policy batches are flushed before saving. Resume must preserve scenario, mixture, seed and step horizon. Trust/game histories are reconstructed causally in each new SUMO episode.

## Run all comparisons

Use the repository's existing prepared environment; there are no additional dependency pins. SUMO and TraCI must be installed as described in the README. `nashpy` is already pinned for Nash and is also used to verify Trust-MARL's small games.

```bash
python scripts/run_experiments.py --suite common --run-dir runs/common_01
```

This trains all eight learned methods, evaluates them and the three fixed policies, exports CSV/Excel and generates common-reward comparison plots and paired-difference tables. The no-flags runner retains its previous method defaults. Preview the common suite with `--dry-run`.

A bounded execution check with all learners, not a convergence study:

```bash
python scripts/run_experiments.py --suite common --run-dir runs/common_smoke \
  --common-episodes 1 --common-train-steps 1000 \
  --common-eval-steps 1000 --common-eval-seeds 0,25 \
  --eval-demands 800 --eval-prs 0.20
```

Select methods, for example:

```bash
python scripts/run_experiments.py --suite common --run-dir runs/selected_01 \
  --methods shepherd_v2_common z_merge_common trust_marl_common ma2c_common qmix_common maac_common
```

`--common-scenario lane_drop` and `--common-hdv-mixture cautious|aggressive` explicitly select transfer environments/mixtures. Original defaults remain unchanged. Train/evaluate/results stages work as before with `--stage`; `--common-model-root PATH` points to an external folder laid out as `PATH/METHOD/models/checkpoint-N.pt`. Checkpoints are copied for evaluation. Use fresh run directories to avoid overwriting experiments.

## Train and evaluate one method

Every training/evaluation module in the first table accepts the same options. For example:

```bash
python -m training.train_trust_marl --base-dir results_common/trust_marl_common --seed 42
python -m evaluation.evaluate_trust_marl \
  --model-dir results_common/trust_marl_common/models

python -m training.train_qmix --base-dir results_common/qmix_common --seed 42
python -m evaluation.evaluate_qmix --model-dir results_common/qmix_common/models
```

The underlying generic APIs also accept `--method METHOD`:

```bash
python -m training.train_common_reward --method maac_common --base-dir results_common/maac_common
python -m evaluation.evaluate_common_reward --method rule_cav_common
```

Resume to a higher total episode target:

```bash
python -m training.train_qmix --base-dir results_common/qmix_common \
  --resume results_common/qmix_common/models --seed 42 --episodes 4000
```

For a shortened training check, repeat its original `--max-steps` when resuming. `--audit-reward` writes per-step raw components, scalar reward and collision denominators to `reward_audit.jsonl`. It is optional because long studies can produce large audit logs. Evaluation always saves per-seed reward-component means and common team return.

## Results, comparisons and interpretation

Evaluations write reports, numeric seed records, action/acceleration/lane/gap/TTC traces, HDV time-loss distributions, three numeric time series and plots. Metadata records method, reward fingerprint/scale/weights, scenario, mixture, demands/PR, control budget, evaluation seeds/horizon, checkpoint episode, training seed and training step count. Different experiments remain distinct.

```bash
python -m analysis.combine_evaluation_reports --search-dirs runs/common_01 \
  --output-csv reports/data/common.csv --output-xlsx reports/data/common.xlsx
python -m analysis.compare_common_reward --search-dirs runs/common_01 \
  --output-dir reports/figures/common
python -m analysis.plot_policy_comparisons --csv reports/data/common.csv --output-dir comparison_charts/common
python -m analysis.plot_evaluation_metrics --csv reports/data/common.csv --output-dir graphs_output/common
python -m analysis.render_results_table --csv reports/data/common.csv --output reports/figures/common_table.png
```

The spreadsheet exporter discovers all method labels and carries protocol columns. Public chart/table entry points discover models dynamically, rather than filtering to an old fixed method list or only 20% PR. They separate reward fingerprints, scenarios, mixtures and evaluation conditions. Specialized historical plotting helpers remain available for old poster artifacts.

`compare_common_reward` accepts only versioned common-reward seed records and rejects mixed reward fingerprints or invalid seed provenance. Paired differences require equal scenarios, mixtures, demand/PR, evaluation seeds, horizons, step sizes and control budgets; learned comparators also require matching training seeds and environment-step budgets. Training runs/checkpoints are kept separate. The differences are **Shepherd/reference minus comparator**, with 95% evaluation-seed confidence intervals. These intervals do not quantify uncertainty across independent training runs. All-HDV's 0% composition is shown separately, rather than misrepresented as a same-PR pair.

Source reports with legacy/unrecorded rewards can still be exported, but are excluded from strict common-reward comparisons. Empty/missing checkpoint paths fail instead of evaluating initialized policies. Short runs can have no arrivals; inspect arrival and unfinished-HDVs counts, and distinguish completed-trip tail delay from censored time loss. Positive speed/throughput differences alone are not success: inspect TTC, collisions, hard braking, jerk/waves and HDV tail delay together. No output automatically declares statistical superiority.

## Tests

```powershell
$env:SHEPHERD_V2_SUMO_TESTS = '1'
$env:COMMON_REWARD_SUMO_TESTS = '1'
python -m unittest discover -s tests -v
```

Tests cover the original methods; exact common reward arithmetic; stable slots and identity resets; real parameter updates and checkpoint round-trips for all eight learners; wrong-method/reward rejection; no-CAV frames; mixed trust games; hybrid SAC density and action masks; MA2C temporal/fingerprint dependence; QMIX monotonicity, greedy consistency and targets; MAAC attention/counterfactual isolation; Z-Merge Double-Q masks; live SUMO execution/learning for all five paper methods; and all-eleven-method export/plot provenance guards. Bounded training/evaluation validates execution and artifacts, not the papers' published results or learned-policy convergence.

## Completed execution validation: 2026-10-26

Validated locally on Windows with Python 3.10.20, CPU PyTorch 2.13.0, NumPy 2.2.6 and SUMO/TraCI 1.27.1, using the existing `rllib_env` interpreter and cached versions of the repository's test/export dependencies. The Python 3.12 setup and dependency pins were preserved.

**All 53 tests passed**, with both live-SUMO test switches enabled. Actual actor, parameter-network, utility, mixer and critic weights changed in their respective learning tests; every learner's online/target networks and optimizers round-tripped through versioned checkpoints.

The final smoke study used the unmodified ramp, demand 800 veh/h/lane, 20% PR, training seed 42 and one complete 1,000-step training episode per learner:

| Learner | CAV-agent transitions | Optimizer updates |
| --- | ---: | ---: |
| Shepherd 2.0 common | 3,595 | 68 |
| Flat MAPPO common | 4,100 | 68 |
| Z-Merge common | 3,934 | 186 |
| Trust-MARL common | 4,318 | 186 |
| MA2C common | 4,096 | 50 |
| QMIX | 4,107 | 186 |
| MAAC | 4,119 | 186 |
| Nash Double-Q common | 4,110 | 219 |

All eight saved checkpoints and all three fixed policies then completed **1,000-step evaluations on each of seeds 0 and 25**: 22 successful evaluation rollouts. All-HDV used its required 0% PR. The final study therefore exercised 30,000 simulator steps across training and evaluation, beyond the smaller numerical/live regression checks and preliminary runs. These are execution/learning checks, not converged policies. SUMO's emergency-braking and lane-end stop warnings from the untrained policies remain visible in logs; successful software execution does not establish safe or superior traffic behavior.

Additional artifact checks:

- Recomputed all 1,500 per-step rewards from the three preliminary reference-method audit logs and their exact coefficients/collision denominators.
- Checked that evaluation checkpoint copies are byte-identical to their saved training checkpoints. Early report provenance counters were enriched from those exact checkpoints without changing any recorded seed metrics.
- Verified one reward fingerprint across all eleven final reports, matching methods, seeds, step horizons, training budgets and action bounds.
- Opened the combined workbook and checked all eleven method keys and **33,000 numeric time-series values**; all saved arrays have the expected shape and finite values.
- Generated **189 paired metric differences**, 44 per-metric charts, two condition-separated summary plots and two tables. Inspected a rendered comparison figure to confirm all ten 20%-PR method labels are visible; All-HDV appears in its separate 0%-PR condition.
- Confirmed all original algorithm/training/evaluation files, SUMO input XML and dependency pins remain unchanged relative to the original checkout.

The local artifacts are ignored by Git:

- `runs/common_reference_methods/`: Z-Merge, Trust-MARL and MA2C training/evaluations.
- `runs/common_extra_methods/`: QMIX and MAAC training/evaluations.
- `runs/common_control_methods/`: Shepherd 2.0, flat MAPPO, Nash Double-Q and fixed controls.
- `runs/common_reward_validation/combined_traffic_results.xlsx` and `.csv`: all-eleven-method aggregation.
- `runs/common_reward_validation/comparison/`: matched-condition plots, numeric metrics and paired differences.
- `runs/common_reward_validation/validation_summary.json`: actual transition/update/rollout/export counts and reward fingerprint.

A full multi-seed training study and the complete evaluation grid remain necessary before making performance or convergence claims. No published-paper benchmark result is claimed by these checks.
