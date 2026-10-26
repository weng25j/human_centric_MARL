# Live SUMO training and evaluation videos

The common-reward suite and standalone Traffic Shepherd 2.0 record the actual
`sumo-gui` viewport during their existing TraCI training/evaluation loops. These
are native simulator snapshots with live vehicles and roads encoded as MP4,
with captions added. Training continues to collect transitions and update the
learner during the recorded rollout; evaluation uses the loaded checkpoint.

## Run the workflow with videos

From the repository root, in your prepared Python environment:

```bash
# All eight learners and three fixed comparisons, with training/evaluation clips.
python scripts/run_experiments.py --suite common --run-dir runs/common_video_01 --record-training --record-evaluation

# Just reward-matched Shepherd 2.0.
python scripts/run_experiments.py --methods shepherd_v2_common --run-dir runs/shepherd_video_01 --record-training --record-evaluation

# Standalone Shepherd 2.0, retaining its checkpoint/ablation support.
python scripts/run_experiments.py --methods shepherd_v2 --run-dir runs/v2_video_01 --record-training --record-evaluation
```

These commands retain the production training/evaluation budgets. By default,
each learner records its first training episode and each method records its first
evaluation grid run, for at most 1,000 simulation steps per clip. Fixed policies
have evaluation videos only. Unselected runs use headless SUMO unless `--gui`
is supplied to the individual program. Recording is disabled unless requested.

A shorter end-to-end execution check (not a converged training study):

```bash
python scripts/run_experiments.py --methods shepherd_v2_common --run-dir runs/video_check --common-episodes 1 --common-train-steps 1000 --common-eval-steps 1000 --common-eval-seeds 0 --eval-demands 800 --eval-prs 0.20 --record-training --record-evaluation
```

At 0.1 seconds per step, this produces 100-second training and evaluation clips.
Allow at least 300–500 steps to see approaching traffic reach the merge; the
initial empty road is the real simulation startup.

## Individual methods and existing checkpoints

```bash
python -m training.train_common_reward --method qmix_common --base-dir runs/qmix_video --record-video
python -m evaluation.evaluate_common_reward --method qmix_common --model-dir runs/qmix_video/models --output-dir runs/qmix_evaluation --record-video

python -m training.train_traffic_shepherd_v2 --base-dir runs/v2_video --record-video
python -m evaluation.evaluate_traffic_shepherd_v2 --model-dir runs/v2_video/models --output-dir runs/v2_evaluation --record-video

python -m evaluation.evaluate_common_reward --method rule_cav_common --output-dir runs/rule_video --record-video
```

Every common-suite method is supported: `shepherd_v2_common`, `flat_mappo_common`,
`z_merge_common`, `trust_marl_common`, `ma2c_common`, `qmix_common`, `maac_common`,
`nash_double_q_common`, `all_hdv_common`, `rule_cav_common`, `selfish_cav_common`.
The per-method wrapper modules accept the same flags. Training `--resume` records
the first selected episodes of that invocation using their resumed episode numbers.

| Option | Default | Effect |
| --- | --- | --- |
| `--record-video` | Off | Individual program: record and automatically launch `sumo-gui` for selected runs. |
| `--record-training` / `--record-evaluation` | Off | Runner: record the corresponding stages of common-suite/standalone 2.0 methods. |
| `--video-runs N` | 1 | First N episodes or evaluation grid runs per method and invocation. Evaluation order: mixture, demand, PR, seed. |
| `--video-max-steps N` | 1000 | Cap each clip independently of the experiment horizon. Training/evaluation continues after capture ends. |
| `--video-every N` | 1 | Capture every N existing steps. FPS is `10/N`, keeping playback at simulation speed. |

For example, `--video-runs 2 --video-every 2 --video-max-steps 500` records two
50-second clips at 5 FPS, retaining the original experiment horizons. Use new
output/run directories for new experiments; existing videos are never silently
overwritten. Recording adds wall time and temporary disk usage.

## Outputs

- Training: `<base-dir>/recordings/training_episodeN_seedS.mp4`.
- Evaluation: `<output-dir>/recordings/evaluation_SCENARIO_MIXTURE_dD_prP_seedS.mp4`.
- Runner: `models/METHOD/recordings/` and recursively under `evaluation/METHOD/`.
  Standalone 2.0 evaluation uses the copied checkpoint's timestamped evaluation directory.
- Each `.video.json` sidecar records method, phase, seed, scenario, demand/PR,
  duration, FPS, frame count, completed experiment steps, per-frame CAV/control
  counts, and snapshot/observed times. Training adds episode transition/update
  totals and per-frame learning counters; evaluation records checkpoint provenance
  and zero updates.
- CSV/Excel metric export continues normally. The workbook's `artifacts` sheet
  indexes MP4 paths and sizes. Videos remain external files.

Videos are 1280×720 MPEG-4 Part 2 (`mp4v`) in an MP4 container. The native view is
fitted without stretching, with method, phase, simulation time and control counts.
Original colors remain: red CAVs, green cautious HDVs, white normal HDVs, blue
aggressive HDVs. The default protocol controls at most five CAVs; not every red
vehicle necessarily receives an agent action. Cameras focus on the ramp merge
or lane-drop junction as appropriate.

Playback follows simulation time, independent of the wall time spent learning
or encoding. Optimizer pauses are not inserted as frozen frames. A short training
video does not establish convergence or superior traffic performance.

## Capture and requirements

[`recording/live_sumo.py`](../recording/live_sumo.py) uses SUMO's
[scheduled native snapshots](https://eclipse.dev/sumo/docs/sumo-gui.html#screenshots)
in a private GUI settings file. It preserves repository network/routes/settings,
seeds, rewards, action logic and physics. Each control step still makes exactly
one TraCI `simulationStep`. No extra simulator steps are used to flush screenshots.
Snapshot time `(step-1)*0.1` is processed during completed step `step`; both
times are recorded explicitly. This avoids synchronous GUI screenshot IPC,
which can deadlock on Windows. GUI IPC and process shutdown have bounded waits.
Recorded GUIs are serialized across processes in this checkout using the already
pinned `filelock`, since concurrent native OpenGL windows proved unreliable on
the Windows desktop. A second recording waits up to ten minutes for the active
GUI to close. Headless runs are unaffected.
A 25 ms GUI delay, a startup layout wait and 50 ms of event-loop time per step
affect wall time only. Giving the GUI time on every step also covers intervals
without snapshots, which were otherwise unreliable on Windows with `--video-every 2`.

You need `sumo-gui` on PATH, a working desktop/OpenGL display and the already
pinned `opencv-python`. No FFmpeg executable or screen capture package is needed.
Linux needs an X display (or a configured Xvfb/OpenGL display). Keep the SUMO
window available while recording. Native capture excludes unrelated desktop
windows.

Every scheduled PNG is checked. The finished MP4 is completely decoded and
checked for expected frame count and dimensions before success is reported.
Missing GUI frames, encoding problems or invalid output raise errors. Temporary
PNGs are deleted after successful verification and preserved beside the output
on failure for diagnosis.

Legacy recording programs/flags remain available. The new runner flags target
the common suite and standalone 2.0; use common-protocol counterparts to record
comparisons through this shared implementation.

## Tests

```bash
python -m unittest discover -s tests -v
```

Default tests cover capture schedules/limits, complete MP4 decoding, run budgets,
failure retention, bounded GUI shutdown and runner flag forwarding. Real GUI
tests are opt-in because they open windows. In PowerShell:

```powershell
$env:SUMO_VIDEO_TESTS = '1'
python -m unittest discover -s tests -p test_live_sumo_video.py -v
```

These train/evaluate real common-protocol and standalone Shepherd 2.0 models,
verify nonzero updates and traffic movement inside the native viewport, verify
clip limits without truncating evaluation, compare recorded/headless seeded
traffic metrics exactly, record a fixed policy and check the workbook MP4 index.
`SHEPHERD_V2_SUMO_TESTS=1` and `COMMON_REWARD_SUMO_TESTS=1` additionally enable
the existing simulator/paper-method integration tests.
