"""Record the live sumo-gui view without replaying or advancing the simulation.

SUMO's scheduled native snapshots avoid GUI screenshot IPC deadlocks on Windows.
https://eclipse.dev/sumo/docs/sumo-gui.html#screenshots
"""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def add_video_arguments(parser, include_switch=True):
    if include_switch:
        parser.add_argument('--record-video', action='store_true',
                            help='Record live SUMO MP4s; automatically enables the GUI for selected runs')
    parser.add_argument('--video-runs', type=int, default=1,
                        help='Record the first N episodes/evaluation grid runs in this invocation (default 1)')
    parser.add_argument('--video-max-steps', type=int, default=1000,
                        help='Record at most N steps per clip; the experiment continues to its original horizon')
    parser.add_argument('--video-every', type=int, default=1,
                        help='Capture every N simulation steps (default 1 = 10 FPS at real simulation speed)')


def validate_video_arguments(parser, args):
    if min(args.video_runs, args.video_max_steps, args.video_every) < 1:
        parser.error('Video run count, capture limit and capture interval must be positive')


class VideoPlan:
    """Limit recording independently of the training/evaluation experiment grid."""
    def __init__(self, args, directory, method, phase):
        self.args, self.directory, self.method, self.phase = args, Path(directory), method, phase
        self.count = 0

    def next(self, name, max_steps, **metadata):
        if not self.args.record_video or self.count >= self.args.video_runs:
            return None
        self.count += 1
        return LiveSUMORecorder(self.directory/f'{name}.mp4',
            min(max_steps, self.args.video_max_steps), every=self.args.video_every,
            metadata={'method': self.method, 'phase': self.phase, **metadata})


class LiveSUMORecorder:
    """One scheduled snapshot sequence, encoded and decoded to verify an MP4.

    Pass this object to TrafficShepherdV2Env/ CommonRewardMergeEnv. The environment
    calls after_step once per existing simulationStep and finish after SUMO exits.
    Snapshot time is the SUMO scheduling time, one step before the completed
    control step; both times are recorded explicitly in the sidecar JSON.
    """
    def __init__(self, output, max_steps, every=1, dt=.1, metadata=None, timeout=10.):
        if max_steps < 1 or every < 1 or dt <= 0 or timeout <= 0:
            raise ValueError('Recording limits, interval, step duration and timeout must be positive')
        # Fail before starting SUMO if the encoder dependency is unavailable.
        import cv2
        self.cv2 = cv2
        self.output = Path(output).expanduser().resolve()
        self.sidecar = self.output.with_suffix('.video.json')
        if self.output.exists() or self.sidecar.exists():
            raise FileExistsError(f'Recording already exists: {self.output}')
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir = Path(tempfile.mkdtemp(prefix='sumo-frames-', dir=self.output.parent))
        self.max_steps, self.every, self.dt, self.timeout = max_steps, every, dt, timeout
        self.metadata, self.frames = dict(metadata or {}), []
        self.finished, self.steps = False, 0
        self.gui_lock = None
        self.settings = self.frames_dir/'view.xml'
        root = ET.Element('viewsettings')
        ET.SubElement(root, 'viewport', zoom='220',
                      x='1000' if self.metadata.get('scenario') == 'lane_drop' else '600', y='-10')
        ET.SubElement(root, 'delay', value='25')
        # At completed step 1 the GUI processes snapshot time 0, at step 2 time .1, etc.
        for step in range(1, max_steps+1, every):
            ET.SubElement(root, 'snapshot', time=f'{(step-1)*dt:.6f}', file=self.frame_path(step).as_posix())
        ET.ElementTree(root).write(self.settings, encoding='utf-8', xml_declaration=True)

    def frame_path(self, step):
        return self.frames_dir/f'frame_{step:08d}.png'

    def sumo_options(self):
        # Native OpenGL GUI instances can interfere on the same Windows desktop.
        # Serialize recorded GUIs across processes; headless SUMO is unaffected.
        from filelock import FileLock, Timeout
        lock_path = Path(__file__).resolve().parents[1]/'.cache'/'sumo_gui_recording.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.gui_lock = FileLock(str(lock_path))
        try:
            self.gui_lock.acquire(timeout=0)
        except Timeout:
            print('Waiting for another live SUMO recording to close its GUI...', flush=True)
            self.gui_lock.acquire(timeout=600)
        return ['--gui-settings-file', str(self.settings), '--start', '--quit-on-end',
                '--window-size', '1280,720', '--window-pos', '50,50']

    def release_gui(self):
        if self.gui_lock is not None:
            self.gui_lock.release()

    def connected(self, connection):
        # TraCI exposes no public socket timeout setting. Bound GUI IPC waits so
        # a paused/stalled native GUI cannot hang the experiment indefinitely.
        connection._socket.settimeout(30.)
        time.sleep(.1)  # allow the newly created native view to finish layout

    def close_connection(self, connection):
        # Connection.close(wait=True) has an unbounded Popen.wait on Windows.
        # Every expected PNG has already been read, so no extra step is needed.
        process = connection._process
        try:
            connection.close(wait=False)
        finally:
            try:
                if process is not None:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
            finally:
                self.release_gui()

    def after_step(self, env):
        self.steps = env.steps
        # Give the native GUI event loop time to render even on steps with no
        # snapshot. A viewsettings delay alone does not pace TraCI on Windows.
        time.sleep(.05)
        if env.steps > self.max_steps or (env.steps-1) % self.every:
            return
        path = self.frame_path(env.steps)
        deadline = time.monotonic()+self.timeout
        while True:
            if path.exists() and path.stat().st_size > 0:
                picture = self.cv2.imread(str(path))
                if picture is not None:
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'SUMO GUI did not render step {env.steps}; snapshots preserved in {self.frames_dir}')
            time.sleep(.01)
        self.frames.append({'step': env.steps, 'snapshot_time_s': (env.steps-1)*self.dt,
            'observed_time_s': env.steps*self.dt, 'vehicles': len(env.states),
            'cav_count': sum(n.startswith('cav') for n in env.states),
            'controlled_ids': sorted(env.last_commands), 'accelerations': env.last_commands.copy()})

    def note_learning(self, updates, transitions):
        """Annotate a captured training frame after the existing optimizer call."""
        if self.frames and self.frames[-1]['step'] == self.steps:
            self.frames[-1].update(learner_updates=updates, transitions=transitions)

    def finish(self):
        if self.finished:
            return
        expected = len(range(1, min(self.steps, self.max_steps)+1, self.every))
        if not self.frames or len(self.frames) != expected:
            raise RuntimeError(f'Incomplete SUMO recording; snapshots preserved in {self.frames_dir}')
        cv2, fps = self.cv2, 1/(self.dt*self.every)
        temporary = self.frames_dir/'encoded.mp4'
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError(f'MP4 encoder unavailable; snapshots preserved in {self.frames_dir}')
        try:
            import numpy as np
            from common.benchmark_methods import LABELS
            method = self.metadata.get('method', 'agent')
            label = LABELS.get(method, 'Traffic Shepherd 2.0' if method == 'traffic_shepherd_v2' else method)
            title = f"{label} | {self.metadata.get('phase', 'simulation')}"
            for record in self.frames:
                picture = cv2.imread(str(self.frame_path(record['step'])))
                if picture is None:
                    raise RuntimeError(f'Unreadable SUMO snapshot: {self.frame_path(record["step"])}')
                h, w = picture.shape[:2]
                scale = min(1280/w, 630/h)
                w, h = 2*int(w*scale/2), 2*int(h*scale/2)
                canvas = np.full((720, 1280, 3), 245, np.uint8)
                top, left = 45+(630-h)//2, (1280-w)//2
                canvas[top:top+h, left:left+w] = cv2.resize(picture, (w, h))
                cv2.putText(canvas, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (40, 40, 40), 2)
                legend = 'CAV: red | HDV: green (cautious), white (normal), blue (aggressive)'
                cv2.putText(canvas, legend, (18, 661), cv2.FONT_HERSHEY_SIMPLEX, .5, (70, 70, 70), 1)
                caption = (f"Live SUMO / TraCI | step {record['step']} | t={record['observed_time_s']:.1f}s"
                           f" | vehicles {record['vehicles']} | controlled CAVs {len(record['controlled_ids'])}")
                cv2.putText(canvas, caption, (18, 704), cv2.FONT_HERSHEY_SIMPLEX, .6, (40, 40, 40), 1)
                writer.write(canvas)
        finally:
            writer.release()
        capture, decoded = cv2.VideoCapture(str(temporary)), 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (720, 1280):
                    raise RuntimeError('Encoded MP4 has invalid dimensions')
                decoded += 1
        finally:
            capture.release()
        if decoded != len(self.frames):
            raise RuntimeError(f'MP4 validation failed ({decoded}/{len(self.frames)} frames); kept {self.frames_dir}')
        temporary.rename(self.output)
        self.sidecar.write_text(json.dumps({**self.metadata, 'source': 'live sumo-gui native snapshots',
            'fps': fps, 'frame_count': decoded, 'width': 1280, 'height': 720,
            'duration_s': decoded/fps, 'step_seconds': self.dt, 'capture_every': self.every,
            'experiment_steps_completed': self.steps, 'capture_max_steps': self.max_steps,
            'frames': self.frames}, indent=2), encoding='utf-8')
        self.finished = True
        shutil.rmtree(self.frames_dir)
        print(f'Saved live SUMO video: {self.output} ({decoded} frames, {decoded/fps:.1f}s)', flush=True)
