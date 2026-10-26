"""MP4 integrity, bounded recording and optional real SUMO train/evaluate checks."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from openpyxl import load_workbook

from analysis.combine_evaluation_reports import export_results
from recording.live_sumo import LiveSUMORecorder, VideoPlan, add_video_arguments, validate_video_arguments


def decode(path):
    reader, frames = cv2.VideoCapture(str(path)), []
    try:
        fps = reader.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
        return fps, frames
    finally:
        reader.release()


class VideoTests(unittest.TestCase):
    def test_recorded_guis_are_serialized_between_processes(self):
        from filelock import FileLock, Timeout
        with tempfile.TemporaryDirectory() as td:
            recorder = LiveSUMORecorder(Path(td)/'clip.mp4', 1)
            recorder.sumo_options()
            competing = FileLock(recorder.gui_lock.lock_file)
            try:
                with self.assertRaises(Timeout):
                    competing.acquire(timeout=0)
            finally:
                recorder.release_gui()
            with competing.acquire(timeout=0):
                pass

    def test_gui_connection_waits_and_shutdown_are_bounded(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            recorder = LiveSUMORecorder(Path(td)/'clip.mp4', 1)
            connection = MagicMock()
            with patch('recording.live_sumo.time.sleep'):
                recorder.connected(connection)
            connection._socket.settimeout.assert_called_once_with(30.)
            connection._process.wait.side_effect = [subprocess.TimeoutExpired('sumo-gui', 5), None]
            recorder.close_connection(connection)
            connection.close.assert_called_once_with(wait=False)
            connection._process.terminate.assert_called_once()
            connection._process.kill.assert_not_called()

    def test_schedule_mp4_decode_and_capture_limit(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'spaces & names'/'clip.mp4'
            recorder = LiveSUMORecorder(output, max_steps=5, every=2)
            schedule = ET.parse(recorder.settings).getroot().findall('snapshot')
            self.assertEqual([float(s.get('time')) for s in schedule], [0., .2, .4])
            self.assertEqual([Path(s.get('file')) for s in schedule], [recorder.frame_path(s) for s in (1, 3, 5)])
            env = SimpleNamespace(steps=0, states={'cav0': {}}, last_commands={'cav0': -1.})
            for step in range(1, 9):
                env.steps = step
                if step in (1, 3, 5):
                    cv2.imwrite(str(recorder.frame_path(step)), np.full((100, 200, 3), step*30, np.uint8))
                recorder.after_step(env)
                recorder.note_learning(step, 2*step)
            recorder.finish()
            recorder.finish()  # cleanup/finalization is idempotent
            fps, frames = decode(output)
            self.assertEqual(fps, 5.)
            self.assertEqual(len(frames), 3)
            self.assertTrue(all(f.shape == (720, 1280, 3) for f in frames))
            self.assertGreater(np.abs(frames[0].astype(float)-frames[-1]).mean(), 5)
            metadata = json.loads(output.with_suffix('.video.json').read_text())
            self.assertEqual(metadata['experiment_steps_completed'], 8)
            self.assertEqual(metadata['duration_s'], .6)
            self.assertEqual(metadata['frames'][-1]['learner_updates'], 5)
            self.assertFalse(recorder.frames_dir.exists())
            with self.assertRaises(FileExistsError):
                LiveSUMORecorder(output, 5)

    def test_failure_is_visible_and_snapshots_are_retained(self):
        with tempfile.TemporaryDirectory() as td:
            recorder = LiveSUMORecorder(Path(td)/'missing.mp4', 1, timeout=.02)
            env = SimpleNamespace(steps=1, states={}, last_commands={})
            with self.assertRaisesRegex(RuntimeError, 'did not render'):
                recorder.after_step(env)
            self.assertTrue(recorder.settings.exists())
            self.assertFalse(recorder.output.exists())
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                recorder.finish()
        with tempfile.TemporaryDirectory() as td:
            recorder = LiveSUMORecorder(Path(td)/'encoder.mp4', 1)
            env = SimpleNamespace(steps=1, states={}, last_commands={})
            cv2.imwrite(str(recorder.frame_path(1)), np.zeros((50, 50, 3), np.uint8))
            recorder.after_step(env)
            with patch.object(cv2, 'VideoWriter') as writer:
                writer.return_value.isOpened.return_value = False
                with self.assertRaisesRegex(RuntimeError, 'encoder unavailable'):
                    recorder.finish()
            self.assertTrue(recorder.frame_path(1).exists())
            self.assertFalse(recorder.output.exists())

    def test_recording_is_opt_in_and_run_budget_does_not_limit_experiment(self):
        parser = argparse.ArgumentParser()
        add_video_arguments(parser)
        args = parser.parse_args([])
        validate_video_arguments(parser, args)
        with tempfile.TemporaryDirectory() as td:
            plan = VideoPlan(args, Path(td)/'not_created', 'test', 'training')
            self.assertIsNone(plan.next('clip', 1000))
            self.assertFalse(plan.directory.exists())
            args.record_video, args.video_max_steps = True, 3
            recorder = plan.next('clip', 1000)
            self.assertEqual(recorder.max_steps, 3)
            self.assertIsNone(plan.next('another', 1000))
        for flag in ('--video-runs', '--video-max-steps', '--video-every'):
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                validate_video_arguments(parser, parser.parse_args([flag, '0']))


@unittest.skipUnless(os.environ.get('SUMO_VIDEO_TESTS') == '1',
                     'Set SUMO_VIDEO_TESTS=1 to open sumo-gui and verify live MP4s')
class RealSUMOVideoTests(unittest.TestCase):
    def test_common_training_evaluation_baseline_and_export(self):
        from training.train_common_reward import main as train
        from evaluation.evaluate_common_reward import main as evaluate
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Long enough to see traffic enter the merge viewport, with real PPO updates.
            train(['--method', 'shepherd_v2_common', '--base-dir', str(base/'train'),
                   '--episodes', '1', '--max-steps', '350', '--record-video'])
            train_video = next((base/'train/recordings').glob('*.mp4'))
            metadata = json.loads(train_video.with_suffix('.video.json').read_text())
            self.assertGreater(metadata['learner_updates'], 0)
            self.assertGreater(metadata['frames'][-1]['learner_updates'], 0)
            self.assertTrue(any(f['controlled_ids'] for f in metadata['frames']))
            fps, frames = decode(train_video)
            self.assertEqual((fps, len(frames)), (10, 350))
            # Compare only the simulator viewport, excluding the added text counters.
            self.assertGreater(np.count_nonzero(cv2.absdiff(frames[30][100:620], frames[-1][100:620]) > 50), 20)
            options = ['--method', 'shepherd_v2_common', '--model-dir', str(base/'train/models'),
                       '--eval-lane-demands', '800', '--eval-prs', '.2', '--seeds', '0,25', '--max-steps', '80']
            evaluate([*options, '--output-dir', str(base/'evaluation'), '--record-video', '--video-max-steps', '50'])
            clips = list((base/'evaluation/recordings').glob('*.mp4'))
            self.assertEqual(len(clips), 1)  # the second seed still evaluates without recording
            metadata = json.loads(clips[0].with_suffix('.video.json').read_text())
            self.assertEqual(metadata['experiment_steps_completed'], 80)
            self.assertEqual(metadata['frame_count'], 50)
            self.assertEqual(metadata['learner_updates'], 0)
            evaluate([*options, '--output-dir', str(base/'headless')])
            recorded = json.loads(next((base/'evaluation').glob('metrics_*.json')).read_text())
            headless = json.loads(next((base/'headless').glob('metrics_*.json')).read_text())
            self.assertEqual(recorded['seeds'], headless['seeds'])
            evaluate(['--method', 'rule_cav_common', '--output-dir', str(base/'fixed'),
                      '--eval-lane-demands', '800', '--eval-prs', '.2', '--seeds', '0',
                      '--max-steps', '40', '--record-video'])
            self.assertEqual(len(decode(next((base/'fixed/recordings').glob('*.mp4')))[1]), 40)
            export_results([base/'evaluation'], base/'out.csv', base/'out.xlsx')
            book = load_workbook(base/'out.xlsx', read_only=True)
            try:
                self.assertTrue(any(row[1] == '.mp4' for row in list(book['artifacts'].values)[1:]))
            finally:
                book.close()

    def test_standalone_v2_trains_and_evaluates_with_video(self):
        from training.train_traffic_shepherd_v2 import main as train
        from evaluation.evaluate_traffic_shepherd_v2 import main as evaluate
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            train(['--base-dir', str(base/'train'), '--episodes', '1', '--max-steps', '50',
                   '--record-video', '--video-every', '2'])
            evaluate(['--model-dir', str(base/'train/models'), '--output-dir', str(base/'evaluation'),
                      '--eval-lane-demands', '800', '--eval-prs', '.2', '--seeds', '0',
                      '--max-steps', '50', '--record-video', '--video-every', '2'])
            for phase in ('train', 'evaluation'):
                clip = next((base/phase/'recordings').glob('*.mp4'))
                self.assertEqual(len(decode(clip)[1]), 25)
                metadata = json.loads(clip.with_suffix('.video.json').read_text())
                if phase == 'train':
                    self.assertGreater(metadata['learner_updates'], 0)


if __name__ == '__main__':
    unittest.main()
