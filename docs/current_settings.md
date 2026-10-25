# Current experiment and environment settings

These are the production defaults; the new scripts do not change algorithm logic or experiment hyperparameters. Inspect your active interpreter and the current source configuration at any time:

```bash
python scripts/show_settings.py
```

The command prints only an explicit list of project settings, package versions, and selected non-secret environment variables. It does not dump the process environment or credentials. Each pipeline invocation records `settings.json` and `pipeline_options.json` in its run directory.

## Computer environment

- Python 3.12; verified with 3.12.14 on Linux x86-64.
- CPU PyTorch 2.13.0, NumPy 2.2.6, pandas 2.3.3, Gymnasium 1.2.2, PettingZoo 1.26.1, SUMO/TraCI/sumolib 1.27.1.
- Additional workflow tools: `eclipse-sumo==1.27.1` supplies the simulator, and `openpyxl==3.1.5` writes Excel workbooks. These are pinned in `requirements-tools.txt`; original library pins remain in the UTF-16 `requirements.txt`.
- Linux is tested. The Python setup/runner scripts also use Windows-compatible paths; Windows execution and GUI recording have not been tested here.
- No API keys, credentials, database, or remote training service are needed. Installation needs the package indexes; execution needs local SUMO/TraCI socket access. GPU support is optional and requires the appropriate PyTorch wheel and driver.

## Environment variables

No custom environment variable is mandatory if you activate the virtual environment and use the script defaults. Scripts preserve already configured proxy and certificate settings.

| Variable | Default/recommended value | Purpose |
| --- | --- | --- |
| `MARL_VENV` | `<repository>/.venv` | Setup destination and Bash activation helper environment. |
| `PATH` | Virtual environment's `bin/` or `Scripts/` first | Finds Python and SUMO. Activation sets it; runner also prepends the selected Python directory. |
| `MPLBACKEND` | `Agg` | Headless figures. Select an interactive backend only for a graphical session. |
| `MPLCONFIGDIR` | `<repository>/.cache/matplotlib` | Writable Matplotlib cache. |
| `XDG_CACHE_HOME` | `<repository>/.cache` | Writable application/font cache, set by Bash activation helper. |
| `OMP_NUM_THREADS` | `2` | Conservative CPU threading default; adjust for your machine. |
| `PYTHONDONTWRITEBYTECODE` | `1` | Avoid Python bytecode files in the checkout. |
| `SUMO_HOME` | Optional | Only needed for a separately installed SUMO toolchain; not required when the pinned package supplies `sumo` on PATH. |
| `PYTHONPATH` | Set by pipeline for its children | Allows modules to run against copied SUMO inputs without modifying the checkout. No manual setting needed. |
| `MARL_RUN_DIR` | New `runs/<UTC timestamp>` for `--stage all` | Default isolated pipeline output path; CLI `--run-dir` overrides it. Separate stages require an explicit path or this variable. |
| `MARL_SEED` | `42` | Trainer base seed; overridden by `--seed`. |
| `MARL_EVAL_DEMANDS` | `800,1700` | Evaluation lane demand grid for MAPPO and fixed baselines; `--eval-demands` overrides it. |
| `MARL_EVAL_PRS` | `0.05,0.10,0.20` | Evaluation CAV fractions for MAPPO and fixed baselines; `--eval-prs` overrides it. |
| `MARL_ZMERGE_EPISODES` | `5` | Z-Merge evaluation episodes; `--zmerge-episodes` overrides it. |

`MARL_*` workflow variables are read by the new scripts, not the original training/evaluation modules. Threading/backend settings affect runtime, not training hyperparameters. `PYTHONHASHSEED` is not set by the helper; the main trainers already seed Python `random`, NumPy, and PyTorch through `--seed`.

## Training defaults

### `train_flat_marl`

```json
{
  "actor_hidden_size": 256,
  "critic_hidden_size": 256,
  "MAX_EPISODES": 3000,
  "EPISODES_BEFORE_TRAIN": 10,
  "ROLL_OUT_N_STEPS": 4000,
  "EVAL_INTERVAL": 50,
  "EVAL_EPISODES": 10,
  "BATCH_SIZE": 256,
  "MEMORY_CAPACITY": 50000,
  "reward_gamma": 0.99,
  "MAX_GRAD_NORM": 0.5,
  "ENTROPY_REG": 0.01,
  "reward_type": "global_R",
  "TARGET_UPDATE_STEPS": 5,
  "TARGET_TAU": 0.01,
  "actor_lr": 0.0003,
  "critic_lr": 0.001,
  "reward_scale": 20.0
}
```

### `train_traffic_shepherd`

```json
{
  "actor_hidden_size": 256,
  "critic_hidden_size": 256,
  "MAX_EPISODES": 3000,
  "EPISODES_BEFORE_TRAIN": 10,
  "ROLL_OUT_N_STEPS": 4000,
  "EVAL_INTERVAL": 50,
  "EVAL_EPISODES": 10,
  "BATCH_SIZE": 256,
  "MEMORY_CAPACITY": 50000,
  "reward_gamma": 0.99,
  "MAX_GRAD_NORM": 0.5,
  "ENTROPY_REG": 0.01,
  "reward_type": "global_R",
  "TARGET_UPDATE_STEPS": 5,
  "TARGET_TAU": 0.01,
  "actor_lr": 0.0003,
  "critic_lr": 0.001,
  "reward_scale": 20.0
}
```

### `train_z_merge`

```json
{
  "state_dim": 22,
  "MAX_EPISODES": 3000,
  "EPISODES_BEFORE_TRAIN": 10,
  "ROLL_OUT_N_STEPS": 1000,
  "EVAL_INTERVAL": 50,
  "EVAL_EPISODES": 10,
  "BATCH_SIZE": 128,
  "MEMORY_CAPACITY": 50000,
  "epsilon_start": 0.99,
  "epsilon_decay": 0.995,
  "epsilon_min": 0.05,
  "reward_scale": 20.0
}
```

## Effective settings and evaluation

- Traffic Shepherd uses 14 observation features and 3 semantic actions; flat MAPPO uses 6 features and 5 acceleration actions. Both keep five padded CAV slots and hidden actor/critic width 256 in their runners.
- The shared MAPPO learner's `max_steps` default is **1,000**, while the trainer CONFIG requests 4,000 rollout steps. An interaction ends at the learner cap. This existing behavior is documented, not changed by the workflow tools.
- Z-Merge uses 22 observation features, 5 action indices, hidden layers `[256, 512, 512, 128]`, Huber loss, Adam learning rates `1e-4` for both networks, discount `0.995`, target hard synchronization every 35,000 gradient updates, and a **100,000-transition effective replay buffer**. Its trainer's `MEMORY_CAPACITY=50000` field is currently unused; the learner uses its own default.
- Training curriculum: episodes below 1,000 use 800 vehicles/hour/lane and 20% PR; episodes 1,000–1,999 sample demands 800/1,200 at 10% PR; later episodes sample demands 800/1,200/1,700 and PR 5/10/20%. Per-episode SUMO seeds are base seed plus episode count.
- MAPPO and fixed-baseline evaluation use 4,000 steps, 0.1 seconds/step, demands 800/1,700, PR 5/10/20%, and 20 seeds: `0,25,50,75,100,125,150,175,200,325,350,375,400,425,450,475,500,525,550,575`. All-HDV is evaluated only at 0% PR.
- Z-Merge evaluation uses five episodes by default, up to 1,000 steps per episode at 0.1 seconds/step, randomized simulator seeds, and its existing route input. It does not expose the same demand/PR grid or report the same set of metrics. Its report omits demand, PR, seed count and confidence intervals; the export leaves those fields blank.
- Fixed baselines use `safe_gap=25.0 m` and `smoothing_speed=15.0 m/s`. They have no learned parameters or training phase.
- Main workflows run headlessly by default. The legacy experimental MA2C/RLlib prototypes remain outside the automatic pipeline because their integration is incomplete.


## Nash Double-Q comparison

The new non-Chen, game-theoretic comparison has independent settings and does not alter the methods above. See [its method guide](nash_double_q.md) for the publication, equations, simulator adaptations and exact-reproduction limits. `scripts/show_settings.py` reads its full `NashConfig` directly from source.

- State 14 (two local six-feature observations and presence flags), three longitudinal actions (−1.5, 0, +1.5 m/s²), two separate 3×3 payoff matrices, hidden layers 128/128.
- Nash equilibrium of the online game selects the expected target-critic payoff; mixed strategies are sampled with seeded randomness.
- Gamma 0.99, Adam 3e−4, batch 64, replay 50,000, learning starts at 256 transitions, train every four simulator steps, hard target copy every 500 updates, reward scale 20, gradient clip 0.5.
- Defaults: 3,000 training episodes × 1,000 steps, same demand/PR curriculum, and the existing evaluation grid/20 seeds × 4,000 steps. The controller handles up to five CAVs by default.
- Pipeline options: `--nash-episodes`, `--nash-train-steps`, `--nash-eval-steps`, `--nash-eval-seeds`, and `--nash-model-dir`. No new custom environment variables or credentials are required. Dependency: `nashpy==0.0.43` in `requirements-tools.txt`.
