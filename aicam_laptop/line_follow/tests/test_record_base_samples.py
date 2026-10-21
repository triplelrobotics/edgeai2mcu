import argparse
import csv
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

SPEC = importlib.util.spec_from_file_location(
    "record_base_samples", Path(__file__).resolve().parents[1] / "record_base_samples.py"
)
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(
            capture_dir=self.root / "var/capture",
            output_dir=self.root / "datasets/source_material/records",
            stream_url="http://camera/video_feed", motor_url="http://motor:80",
            save_fps=10.0, no_preview=True,
        )

    def keyboard_app(self):
        session = recorder.create_session(self.args)
        app = recorder.App(self.args, session)
        self.addCleanup(app.recorder.close)
        self.addCleanup(app.controller.session.close)
        app.controller.send_action = Mock(return_value=True)
        keys = SimpleNamespace(left="left", right="right", up="up", down="down", space="space")
        keyboard = SimpleNamespace(Key=keys, Listener=Mock(return_value=Mock()))
        mocked_pynput = patch.dict("sys.modules", {"pynput": SimpleNamespace(keyboard=keyboard)})
        mocked_pynput.start()
        self.addCleanup(mocked_pynput.stop)
        return app, keys

    def test_steering_can_start_motion_and_releases_to_straight(self):
        app, keys = self.keyboard_app()
        self.assertEqual(app.current_action, "STOP")
        for key, action in ((keys.left, "LEFT"), (keys.right, "RIGHT")):
            app.on_press(key)
            self.assertEqual(app.current_action, action)
            app.on_press(key)
            self.assertEqual(app.current_action, action)
            app.on_release(key)
            self.assertEqual(app.current_action, "STRAIGHT")

    def test_stop_release_keeps_stop_until_a_new_driving_event(self):
        for name in ("space", "down"):
            with self.subTest(key=name):
                app, keys = self.keyboard_app()
                app.on_press(keys.up)
                app.on_release(keys.up)
                app.on_press(keys.left)
                app.on_press(getattr(keys, name))
                self.assertEqual(app.current_action, "STOP")
                app.on_release(getattr(keys, name))
                self.assertEqual(app.current_action, "STOP")
                app.on_release(keys.left)
                self.assertEqual(app.current_action, "STRAIGHT")
                app.on_press(keys.right)
                self.assertEqual(app.current_action, "RIGHT")
                app.on_release(keys.right)
                self.assertEqual(app.current_action, "STRAIGHT")

    def test_stop_and_up_releases_do_not_send_commands(self):
        app, keys = self.keyboard_app()
        app.on_press(keys.up)
        app.on_press(keys.space)
        self.assertEqual(app.current_action, "STOP")
        app.controller.send_action.reset_mock()
        app.on_release(keys.space)
        self.assertEqual(app.current_action, "STOP")
        app.on_release(keys.up)
        self.assertEqual(app.current_action, "STOP")
        app.controller.send_action.assert_not_called()
        app.on_press(keys.up)
        self.assertEqual(app.current_action, "STRAIGHT")

    def test_stop_key_releases_send_no_driving_command(self):
        app, keys = self.keyboard_app()
        app.on_press(keys.space)
        app.on_press(keys.down)
        self.assertEqual(app.current_action, "STOP")
        app.on_release(keys.space)
        self.assertEqual(app.current_action, "STOP")
        app.on_release(keys.down)
        self.assertEqual(app.current_action, "STOP")
        app.on_press(keys.up)
        self.assertEqual(app.current_action, "STRAIGHT")

    def test_events_follow_web_control_without_combination_priority(self):
        app, keys = self.keyboard_app()
        app.on_press(keys.up)
        app.on_press(keys.left)
        self.assertEqual(app.current_action, "LEFT")
        app.on_press(keys.right)
        self.assertEqual(app.current_action, "RIGHT")
        app.on_release(keys.up)
        self.assertEqual(app.current_action, "RIGHT")
        app.on_release(keys.left)
        self.assertEqual(app.current_action, "STRAIGHT")
        app.on_press(keys.space)
        app.on_press(keys.up)
        self.assertEqual(app.current_action, "STRAIGHT")

    def test_repeated_press_is_ignored_until_release(self):
        app, keys = self.keyboard_app()
        for key in (keys.left, keys.right, keys.up, keys.down, keys.space):
            with self.subTest(key=key):
                app.on_press(key)
                timestamp = app.key_ts
                app.controller.send_action.reset_mock()
                app.on_press(key)
                app.controller.send_action.assert_not_called()
                self.assertEqual(app.key_ts, timestamp)
                app.on_release(key)
                app.controller.send_action.reset_mock()
                app.on_press(key)
                app.controller.send_action.assert_called_once()
                app.on_release(key)

    def test_repeated_up_cannot_override_stop(self):
        app, keys = self.keyboard_app()
        app.on_press(keys.up)
        app.on_press(keys.space)
        app.on_release(keys.space)
        app.on_press(keys.up)
        self.assertEqual(app.current_action, "STOP")
        app.on_release(keys.up)
        app.on_press(keys.up)
        self.assertEqual(app.current_action, "STRAIGHT")

    def test_press_and_release_record_event_timestamps(self):
        app, keys = self.keyboard_app()
        with patch.object(recorder.time, "time", side_effect=[1.0, 2.0, 3.0, 4.0]):
            for expected, callback in enumerate(
                (app.on_press, app.on_release, app.on_press, app.on_release), 1
            ):
                callback(keys.space)
                self.assertEqual(app.key_ts, float(expected))
                self.assertEqual(app.current_action, "STOP")

    def test_slow_image_write_does_not_block_keyboard_stop(self):
        app, keys = self.keyboard_app()
        app.on_press(keys.up)
        app.reader = Mock()
        frames = iter([np.zeros((224, 224, 3), dtype=np.uint8)])
        app.reader.read_latest.side_effect = lambda: next(frames, None)
        writing = threading.Event()
        release_write = threading.Event()
        stopped = threading.Event()
        errors = []

        def slow_save(*args):
            writing.set()
            if not release_write.wait(3):
                raise RuntimeError("Timed out waiting to release mock image write")
            return "mock.jpg"

        app.recorder.save = Mock(side_effect=slow_save)

        def capture():
            try:
                app.loop()
            except Exception as error:
                errors.append(error)

        def stop():
            app.on_press(keys.space)
            stopped.set()

        capture_thread = threading.Thread(target=capture)
        stop_thread = threading.Thread(target=stop)
        try:
            capture_thread.start()
            self.assertTrue(writing.wait(2))
            stop_thread.start()
            self.assertTrue(stopped.wait(1), "Keyboard STOP waited for image writing")
            self.assertEqual(app.current_action, "STOP")
            app.controller.send_action.assert_called_with("STOP")
        finally:
            app.running = False
            release_write.set()
            capture_thread.join(2)
            if stop_thread.ident is not None:
                stop_thread.join(2)
        self.assertFalse(capture_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(app.recorder.save.call_args.args[1], "STRAIGHT")

    def test_shutdown_stops_before_resource_cleanup_without_listener_join(self):
        app, keys = self.keyboard_app()
        from pynput import keyboard

        listener = keyboard.Listener.return_value
        app.reader = Mock()
        app.loop = Mock()
        order = []
        listener.stop.side_effect = lambda: order.append("listener_stop")
        app.reader.stop.side_effect = lambda: order.append("reader_stop")
        app.controller.session.close = Mock(side_effect=lambda: order.append("session_close"))

        def send(action, force=False):
            if force:
                order.append("forced_stop")
            return True

        app.controller.send_action.side_effect = send
        app.run()
        self.assertEqual(order, ["listener_stop", "forced_stop", "session_close", "reader_stop"])
        listener.join.assert_not_called()
        self.assertTrue(app.recorder.csv_file.closed)

    def test_q_still_exits_through_forced_stop_cleanup(self):
        app, keys = self.keyboard_app()
        app.reader = Mock()
        def quit_recording():
            app.on_press(keys.up)
            app.on_press(SimpleNamespace(char="q"))
        app.loop = quit_recording
        app.run()
        self.assertFalse(app.running)
        app.controller.send_action.assert_called_with("STOP", force=True)
        self.assertTrue(app.recorder.csv_file.closed)

    def test_unique_sessions_and_keep_preserves_data(self):
        session = recorder.create_session(self.args)
        other = recorder.create_session(self.args)
        self.assertNotEqual(session, other)
        capture = recorder.DatasetRecorder(str(session))
        capture.save(np.zeros((96, 96, 3), dtype=np.uint8), "LEFT", 1.0, 0.5)
        capture.close()
        destination = recorder.review_session(
            session, self.args.capture_dir, self.args.output_dir, "keep"
        )
        self.assertFalse(session.exists())
        self.assertTrue(other.exists())
        self.assertTrue((destination / "images/000001.jpg").exists())
        with (destination / "labels.csv").open() as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]["action"], "LEFT")
        self.assertEqual(json.loads((destination / "session.json").read_text())["status"], "kept")

    def test_discard_only_selected_session(self):
        session = recorder.create_session(self.args)
        other = recorder.create_session(self.args)
        recorder.review_session(session, self.args.capture_dir, self.args.output_dir, "discard")
        self.assertFalse(session.exists())
        self.assertTrue(other.exists())

    def test_keep_rejects_existing_destination(self):
        session = recorder.create_session(self.args)
        (self.args.output_dir / session.name).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            recorder.review_session(session, self.args.capture_dir, self.args.output_dir, "keep")
        self.assertTrue(session.exists())

    def test_discard_rejects_unrelated_directory(self):
        unrelated = self.root / "session_unrelated"
        unrelated.mkdir()
        with self.assertRaises(ValueError):
            recorder.review_session(unrelated, self.args.capture_dir, self.args.output_dir, "discard")
        self.assertTrue(unrelated.exists())

    def test_eof_leaves_pending(self):
        with patch("builtins.input", side_effect=EOFError):
            self.assertEqual(recorder.ask_review(), "pending")

    def test_interrupted_run_does_not_review_or_keep(self):
        arguments = ["--stream-url", self.args.stream_url, "--motor-url", self.args.motor_url,
                     "--capture-dir", str(self.args.capture_dir),
                     "--output-dir", str(self.args.output_dir)]
        with patch.object(recorder, "App") as app, patch.object(recorder, "ask_review") as review:
            app.return_value.run.side_effect = KeyboardInterrupt
            recorder.main(arguments)
            review.assert_not_called()
        sessions = list(self.args.capture_dir.iterdir())
        self.assertEqual(len(sessions), 1)
        self.assertEqual(json.loads((sessions[0] / "session.json").read_text())["status"], "pending")
        self.assertFalse(self.args.output_dir.exists())

    def test_fresh_stop_and_correct_control_parameter(self):
        controller = recorder.ESP32Controller(self.args.motor_url)
        controller.session = Mock()
        self.assertTrue(controller.send_action("STOP"))
        self.assertTrue(controller.send_action("STOP", force=True))
        self.assertEqual(controller.session.get.call_count, 2)
        self.assertEqual(controller.session.get.call_args.kwargs["params"], {"m": "STOP"})
        controller.session.get.side_effect = requests.ConnectionError("offline")
        self.assertFalse(controller.send_action("LEFT"))
        self.assertIsNone(controller.last_sent)

    def test_consecutive_stop_samples_are_skipped(self):
        app, keys = self.keyboard_app()
        app.reader = Mock()
        app.recorder.save = Mock(return_value="mock.jpg")
        actions = iter(["STOP", "STOP", "STRAIGHT", "LEFT", "STOP", "STOP", "RIGHT", "STOP", "STOP"])
        frames = []

        def read():
            action = next(actions, None)
            if action is None:
                app.running = False
                return None
            app.set_action(action)
            frame = np.full((96, 96, 3), len(frames), dtype=np.uint8)
            frames.append(frame)
            return frame

        app.reader.read_latest.side_effect = read
        with patch.object(recorder.time, "time", side_effect=range(100, 110)):
            app.loop()
        saved = app.recorder.save.call_args_list
        self.assertEqual([call.args[1] for call in saved],
                         ["STOP", "STRAIGHT", "LEFT", "STOP", "RIGHT", "STOP"])
        for call, index in zip(saved, (0, 2, 3, 4, 6, 7)):
            self.assertIs(call.args[0], frames[index])

    def test_stop_does_not_force_a_sample_before_the_next_capture_interval(self):
        app, keys = self.keyboard_app()
        app.reader = Mock()
        app.recorder.save = Mock(return_value="mock.jpg")
        actions = iter(["STRAIGHT", "STOP", None])

        def read():
            action = next(actions)
            if action is None:
                app.running = False
                return None
            app.set_action(action)
            return np.zeros((96, 96, 3), dtype=np.uint8)

        app.reader.read_latest.side_effect = read
        with patch.object(recorder.time, "time", side_effect=[100.0, 100.01, 100.02]):
            app.loop()
        app.recorder.save.assert_called_once()
        self.assertEqual(app.recorder.save.call_args.args[1], "STRAIGHT")

    def test_stream_outage_does_not_duplicate_old_frame(self):
        session = recorder.create_session(self.args)
        app = recorder.App(self.args, session)
        app.current_action = "STRAIGHT"
        app.reader = Mock()
        app.recorder.close()
        app.recorder = Mock()
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        def read():
            if app.latest_frame is None:
                return frame
            app.running = False
            return None
        app.reader.read_latest.side_effect = read
        app.loop()
        app.recorder.save.assert_called_once()
        app.controller.session.close()

    def test_recording_error_closes_files_and_forces_stop(self):
        session = recorder.create_session(self.args)
        app = recorder.App(self.args, session)
        app.controller = Mock()
        app.controller.send_action.return_value = True
        app.reader = Mock()
        listener = Mock()
        keyboard = SimpleNamespace(Listener=Mock(return_value=listener))
        app.loop = Mock(side_effect=RuntimeError("disk error"))
        with patch.dict("sys.modules", {"pynput": SimpleNamespace(keyboard=keyboard)}):
            with self.assertRaisesRegex(RuntimeError, "disk error"):
                app.run()
        app.controller.send_action.assert_any_call("STOP", force=True)
        app.controller.session.close.assert_called_once()
        listener.stop.assert_called_once()
        app.reader.stop.assert_called_once()
        self.assertTrue(app.recorder.csv_file.closed)
        metadata = json.loads((session / "session.json").read_text())
        self.assertEqual(metadata["status"], "pending")
        self.assertIn("ended_at", metadata)

    def test_failed_control_stops_recording_without_changing_label(self):
        session = recorder.create_session(self.args)
        app = recorder.App(self.args, session)
        self.addCleanup(app.recorder.close)
        self.addCleanup(app.controller.session.close)
        app.controller.send_action = Mock(return_value=False)
        with self.assertRaises(RuntimeError):
            app.set_action("LEFT")
        self.assertFalse(app.running)
        self.assertTrue(app.control_failed)
        self.assertEqual(app.current_action, "STOP")


if __name__ == "__main__":
    unittest.main()
