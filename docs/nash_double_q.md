# Nash Double-Q merging comparison

Traffic Shepherd remains the main novel algorithm in this repository. This module adds a **non-Chen, game-theoretic comparison**. It is not a flat MAPPO/A2C variant: its decision depends on an explicit two-player joint-action game and a computed Nash equilibrium.

## Literature and research scope

The selected merging publication is:

> Li, Lin, et al. “Nash double Q-based multi-agent deep reinforcement learning for interactive merging strategy in mixed traffic.” *Expert Systems with Applications*, volume 237 (2024), article 121458.

- [Publisher article](https://www.sciencedirect.com/science/article/pii/S0957417423019607)
- [Accessible literature review listing the merging paper, section 3.3.2, item 5](https://github.com/huahuaedi/MARL_in_CAV_control_review#332-on-ramps-merging)
- Mathematical foundation: Hu and Wellman, [“Nash Q-Learning for General-Sum Stochastic Games,” JMLR (2003)](https://www.jmlr.org/papers/v4/hu03a.html).
- Double-Q foundation: van Hasselt, Guez, and Silver, [“Deep Reinforcement Learning with Double Q-learning”](https://arxiv.org/abs/1509.06461).
- Equilibrium solver: [Nashpy](https://github.com/drvinceknight/Nashpy), version 0.0.43, using support enumeration with unilateral-deviation checks.

The online search found the merging publication through the linked review. Other located repositories advertised an unpublished causal merging framework or code awaiting release. Chen's methods were excluded as requested. The managed environment blocks the direct arXiv/API routes, and the selected article's full methods text and an official implementation were not available in this research session. **This is an independently implemented Nash Double-Q SUMO adaptation, not a verified reproduction of Li et al.'s exact architecture, equations, state/action design, or results.** No original hyperparameters or benchmark numbers are claimed. The general Nash-Q and Double-DQN mechanisms are implemented and numerically tested; all simulator choices below are explicit benchmark adaptations. Exact reproduction would require checking the full merging paper before calling this its reference implementation.

## What the implementation does

1. Take up to five active CAVs, matching the existing Traffic Shepherd control budget. Pair nearby CAVs within 100 m; pairings are deterministic disjoint nearest-neighbor matches. Unmatched CAVs use a singleton game. HDVs and other CAVs remain under SUMO control.
2. Build a joint state from two vehicles' local observations: speed, lane position, leader speed/gap, follower speed/gap, plus two presence flags. Missing vehicles are zero-padded. This is a 14-component joint observation, not Traffic Shepherd's 14-feature observation.
3. Predict **two separate 3×3 payoff matrices** with a shared neural critic. Each payoff depends on *both* vehicles' actions. Each player chooses deceleration −1.5 m/s², constant-speed acceleration 0, or acceleration +1.5 m/s². Lateral motion stays under native SUMO lane-changing behavior.
4. Solve the stage game. A pure Nash equilibrium is a joint action from which neither player benefits by changing its own action alone. If multiple pure equilibria exist, select the highest total predicted payoff, breaking remaining ties by action index. If no pure equilibrium exists, enumerate mixed-strategy supports and verify the maximum unilateral improvement is below tolerance. Invalid speed-bound actions are masked. No independent-greedy or random fallback is labelled an equilibrium.
5. Sample from the mixed equilibrium when necessary. Taking the independently most likely actions can destroy an equilibrium, so evaluation also samples mixed strategies using the evaluation seed. Pure equilibria are deterministic. Epsilon exploration randomizes only among legal actions during training.
6. Apply actions through TraCI, collect each player's own reward, and store **joint** transitions. This retains distinct incentives, rather than forcing both players to share one team payoff.
7. Select the next equilibrium with the online critic; evaluate its expected payoffs with the target critic. This separates next-decision selection from target valuation, reducing the maximization coupling of plain Q-learning. Update both payoff estimates with Huber loss and periodically synchronize the target critic.

For player `i`, if `pi_online` is the verified next-state equilibrium distribution:

```text
y_i = reward_i / reward_scale
      + gamma * (1 - game_terminal) * E_joint_action~pi_online[Q_target_i(next_state, joint_action)]
loss = mean Huber(Q_online_i(state, executed_joint_action), y_i)
```

A singleton's absent player has only the neutral action and zero continuation value. A departed vehicle, changed control membership, or new pairing ends the old two-player game; the surviving vehicle can enter a new game. These local game-fragment boundaries are a tractability approximation, not proof of a network-wide Markov-perfect equilibrium. The physical time limit is treated as a truncation with a valid next-state bootstrap. All sampled experience is off-policy replay; this is not PPO or a policy-gradient algorithm.

## Benchmark adaptations and fairness

| Component | This implementation |
| --- | --- |
| Training/evaluation simulator | SUMO 1.27.1 via TraCI, existing `SUMO_network/test.net.xml`. |
| Traffic inputs | Traffic Shepherd's exact existing mixed-traffic route generator, called in isolated workspaces. Source route XML is not overwritten. |
| Reward | Same individual human-centric reward terms/weights as Traffic Shepherd: speed efficiency, follower hard-braking/stability, follower waiting time, braking intervention cost, −10 collision penalty, and zero safe-departure reward. It omits Shepherd's spatial reward smoothing. This comparison reward is not claimed to be Li et al.'s reward. |
| State | Two normalized local six-feature observations and two presence flags. No Shepherd V2X/trust/fingerprint features. |
| Action | Joint selection of two primitive longitudinal commands; no Shepherd semantic gap-creation/pace-making controller. Native SUMO lateral behavior remains active. |
| Control budget | First five active CAVs in SUMO's ID order, matching the existing fixed five-slot controllers. `--max-controlled-cavs 0` is an explicitly different unlimited-control experiment. |
| Safety | Same explicit longitudinal `speedMode=0` as the other learned comparisons; physical collisions remove vehicles. Equilibrium means predicted payoff stability, **not collision-free safety**. Releasing a vehicle from the control budget returns it to native SUMO speed control. |
| Training horizon/curriculum | 3,000 episodes, 1,000 steps/episode, 0.1 s/step; same demand/penetration curriculum stages as Traffic Shepherd. |
| Evaluation | Same default 20 seeds, 800/1,700 veh/h/lane grid, 5/10/20% PR, 4,000 steps at 0.1 s. |

This compares different complete controllers under a common traffic/metric protocol. It is not an algorithm-only ablation: information, action spaces, reward smoothing, and coordination mechanisms differ. The existing controllers and their settings are unchanged.

Default learner settings (`NashConfig` in `algorithms/nash_double_q.py`): hidden layers 128/128 with ReLU; 18 output payoffs (two players × nine joint actions); Adam learning rate 3e−4; gamma 0.99; batch 64; replay capacity 50,000 game transitions; learning starts after 256 transitions; update every four simulation steps; hard target synchronization every 500 learning updates; reward scale 20; gradient norm cap 0.5. Epsilon starts at 0.99, decays per episode by 0.995, and bottoms out at 0.05. These are declared adaptation choices, not reported paper hyperparameters.

## Installation and commands

Refresh the existing environment to install the additional equilibrium solver:

```bash
python3.12 scripts/setup_environment.py
source scripts/activate_environment.sh
```

Run training and evaluation separately:

```bash
python -m training.train_nash_double_q --base-dir results_nash_double_q --seed 42
python -m evaluation.evaluate_nash_double_q \
  --model-dir results_nash_double_q/models \
  --eval-lane-demands 800,1700 --eval-prs 0.05,0.10,0.20
```

For existing custom virtual environments, pass `--venv PATH` to the setup script and activate that same environment. Training/evaluation need `sumo` on PATH. No additional remote service, API key, or GPU is required.

Use the shared runner (including logs, isolated inputs, and Excel export):

```bash
python scripts/run_experiments.py --methods nash_double_q --run-dir runs/nash_01
```

A small execution check that actually learns, then evaluates two seeds:

```bash
python scripts/run_experiments.py --methods nash_double_q --run-dir runs/nash_smoke \
  --nash-episodes 1 --nash-train-steps 500 \
  --nash-eval-steps 1000 --nash-eval-seeds 0,25 \
  --eval-demands 800 --eval-prs 0.20
```

This smoke run does not produce a converged comparison policy. Use a fresh run directory each time. To compare a trained Nash model with Traffic Shepherd, run the same evaluation grid, seeds, horizon and traffic-generator version for both methods. Train several independent seeds and retain each training seed's evaluation separately rather than treating evaluation seeds as independent training runs.

## Outputs, evaluation and resuming

- `BASE/models/checkpoint-N.pt`: named/versioned checkpoint with both critics, optimizer, learner configuration, episode/epsilon, counters and RNG state.
- `BASE/training_metrics.jsonl`: actual learning updates, loss, reward, replay count, and number of two-CAV game decisions per episode. These counts distinguish real coordination from solo control.
- Evaluator: `report_NASH_DOUBLE_Q_*.txt`, per-seed `metrics_*.json`, speed/deceleration/jerk arrays, a trace plot, and `evaluation_settings.json` with checkpoint episode and arguments.
- The shared CSV/Excel exporter automatically recognizes these report headers and embeds the saved time-series arrays. Its default search roots now include `results_nash_double_q/`.

The metrics use existing comparison definitions: collision rate counts all colliding vehicles, including vehicles removed immediately; TTC/near-collision events use 2.5/1.5 s; hard braking uses −3 m/s²; intervention braking uses −0.1 m/s²; queued vehicles are below 2 m/s. Like the old evaluators, merge success counts ramp arrivals excluding vehicles that stopped below 0.1 m/s or collided, so unfinished ramp vehicles remain in the denominator. Short rollouts can consequently report little throughput or low merge success. No-arrival travel/delay means use zero, consistent with the existing reports. Jerk is zero on first observation and wave intensity uses per-vehicle speed variance for histories longer than ten samples. A paper-style success or reward metric would be a separate protocol.

Resume with the original training seed and a higher target episode count:

```bash
python -m training.train_nash_double_q --base-dir results_nash_double_q \
  --resume results_nash_double_q/models --seed 42 --episodes 4000
```

Replay contents are intentionally not stored, so this restores the trained model/optimizer/RNG/counters and continues at a new episode, not a bit-identical continuation of every historical replay sample. Wrong-method or missing checkpoints fail visibly. New training without `--resume` refuses to overwrite existing checkpoints. Each connection closes in a `finally` block.

## Validation

Run `python -m unittest discover -s tests -v`. Tests include mixed-strategy matching pennies, equilibrium changes caused solely by the other player's payoffs, action masks, a closed-form Double-Q target with disagreeing online/target games, terminal masking, actual parameter updates, target synchronization, checkpoint round-trips, dynamic/singleton groups, five-CAV budgeting, native-control release, and removed-collision accounting.

Execution checks validate learning updates and saved artifacts, not convergence, safety guarantees, statistical superiority, or reproduction of the selected article's published results.

Completed checks on this implementation:

- All 24 regression tests passed, including the existing Z-Merge and workflow/export tests.
- A real 500-step training smoke run performed 94 learning updates, collected 1,571 game transitions, and exercised 46 two-CAV game decisions.
- The resulting checkpoint was evaluated for 1,000 steps on each of seeds 0 and 25 at 800 veh/h/lane and 20% PR, with the five-CAV control cap. Those rollouts exercised 1,566 and 487 two-CAV decisions respectively, including physical collision/removal handling.
- The pipeline created the report, per-seed JSON metrics, three saved time series and trace plot, then produced CSV and Excel outputs. Excel contained the `NASH DOUBLE-Q` report and 3,000 time-series values.
- The existing controller source, network/route XML, dependency pins in `requirements.txt`, and saved experiment artifacts were unchanged. Only the additional `nashpy` tool requirement is new.
