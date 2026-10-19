# Z-Merge paper comparison

Reference: Ibork, Won, and Das, *Z-Merge: Multi-Agent Reinforcement Learning for On-Ramp Merging with Zone-Specific V2X Traffic Information*, [arXiv:2511.14910v1](https://arxiv.org/abs/2511.14910v1). The uploaded version is the comparison source, especially “Multi-Agent Parameterized Double DQN,” Eq. (3), and Table 1.

## Scope

The correction concerns the shared Double PDQN learner and execution/transition bugs in the Z-Merge training and evaluation entry points. Traffic Shepherd remains the primary novel method, and its logic, flat MAPPO, and the deterministic baselines are unchanged.

The existing `environments/z_merge.py` is unchanged, including its 22-feature observation vector, placeholder zone/neighbor features, five action indices, physical acceleration/gap bounds, and human-centric comparison reward. SUMO networks, traffic generation, seeds, 3,000-episode curriculum, 1,000-step rollout/evaluation horizon, reward scaling, epsilon schedule, batch sizes, replay capacity, learning rates, discount, target-update interval, and hidden-layer widths retain their existing production values. Shortened tests do not change those defaults.

This is a correction of the paper's **learning algorithm in the repository's existing experiment**, rather than a reproduction of the paper's entire system. The paper uses 42 observation features, zone-specific V2X information, MOSAIC/SimServ, different traffic settings, and six reward components. Replacing the current observation/reward or simulation design would violate the requested scope. The paper also reports a different training batch/exploration setup; these experiment parameters are deliberately preserved here.

## Corrected learning logic

| Component | Previous behavior | Corrected behavior |
| --- | --- | --- |
| Bellman target | Took the maximum using target-network Q-values. | Selects `k* = argmax_k Q_online(s', k, mu_online(s')_k)`; evaluates that action with `Q_target(s', k*, mu_target(s')_k*)`, as Eq. (3) specifies. |
| Action-conditioned critic | Every Q head could depend on all five continuous outputs. | Evaluates each head with its corresponding argument only. Lane-left, lane-right, and maintain-state ignore continuous arguments, consistent with the action definitions. |
| Parameter-network objective | Summed all Q heads, while also accumulating critic gradients. | Minimizes `-mean(Q(s, k, mu(s)_k))` with replayed discrete actions, differentiating through critic inputs while freezing critic weights during this update. |
| Terminal samples | Dropped vehicles absent from the next observation. | Stores these transitions with a zero placeholder next state and `done=True`; the target does not bootstrap them. Existing environment reward semantics are preserved. |
| Target synchronization | Hard update every 35,000 optimizer steps. | Preserved, with regression coverage. |

The paper writes the actor objective for a discrete action `k` but does not specify a separate sampling distribution for `k`. The implementation uses the discrete action in each sampled replay transition. This is an explicit implementation choice for its displayed expectation. Huber TD loss follows Table 1 (the prose displays a squared TD expression).

The critic applies action masks to the existing input representation and evaluates the matching Q head for each action. The four hidden layers, head count, learned tensor shapes, output bounds, and external `(discrete_action, five_parameters)` interface are preserved. The three unused continuous coordinates remain in that interface for compatibility with the unchanged environment and old checkpoint shapes.

## Execution corrections

- Training owns one SUMO connection per episode and keeps its configured seed. Reading initial observations no longer calls the environment's `reset()` and starts a second SUMO process.
- Training no longer hides arbitrary exceptions as successful episodes. Connections close in `finally` blocks. Evaluation also closes its connection on failure.
- Training begins a gradient update once replay contains at least the configured batch size.
- New run directories are joined correctly even when `--base-dir` lacks a trailing slash, and parent directories are created.
- Saving and loading use the same paired-network prefix. Explicit relative/absolute prefixes and legacy prefixes under `results_zmerge/` work. Missing weights fail visibly.
- `training.train_z_merge --option evaluate --model-dir PREFIX` invokes the dedicated evaluator, using its existing defaults.

Checkpoint filenames and weight tensor shapes are compatible with earlier files. Old weights were learned with the previous logic, so **retrain Z-Merge before reporting corrected-method results**. Loading weights is a warm start, not restoration of replay, optimizer state, exploration history, or the target-update counter.

## Regression tests

Run from the repository root in the prepared Python environment:

```bash
python -m unittest discover -s tests -v
```

Tests use fixed critics that disagree on the greedy action to verify the Double PDQN target numerically, including the terminal mask. They also check action-parameter independence, actor/critic gradient separation, actual updates to both online networks, the 35,000-step target interval, explicit/legacy checkpoint round-trips, single-start SUMO ownership, terminal replay retention, evaluation dispatch, and failure cleanup.

Integration validation uses temporary copies of the existing SUMO inputs so repository route files and saved experiments are not overwritten. Smoke checks verify execution and artifacts; they do not establish learning convergence or reproduce the paper's reported performance. GUI recording and the incomplete prototypes under `experiments/` are outside this correction.

Completed checks:

- All 10 unit regression tests passed.
- Z-Merge executed one complete configured 1,000-step training episode with learning enabled: 936 optimizer updates, 2,527 replay transitions, and 3 terminal transitions. The test used the existing `--start-episode 2999` mechanism to bound the run; it did not represent 2,999 earlier trained episodes or measure convergence.
- The newly trained paired checkpoint reloaded with identical deterministic actions and continuous outputs. Its evaluator completed two 1,000-step episodes and wrote a report; missing checkpoints returned a nonzero process status.
- Traffic Shepherd and flat MAPPO each completed a bounded one-episode training run and a checkpoint-based evaluation smoke check. Those evaluation checks used two seeds and 300 steps in an in-memory test copy, with the existing demand/PR options narrowed to 800 vehicles/hour/lane and 10% PR. Production defaults were not edited.
- All-HDV, Rule-CAV, and Selfish-CAV each completed the same two-seed, 300-step smoke check and wrote their three reports.
- Existing network/route XML, dependency pins, saved model/results directories, and the complete Z-Merge environment file remained unchanged.
