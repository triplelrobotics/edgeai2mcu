"""Record synchronized camera frames and control actions for line-follow training."""
import argparse
import csv
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests


# =========================
# User config
# =========================
LAPTOP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CAPTURE_DIR = LAPTOP_DIR / "var" / "capture"
DEFAULT_OUTPUT_DIR = LAPTOP_DIR / "datasets" / "source_material" / "records"
SAVE_FPS = 10.0
JPEG_QUALITY = 90
FRAME_WIDTH = 224   # saved width
FRAME_HEIGHT = 224  # saved height
REQUEST_TIMEOUT = 0.25


# =========================
# Internal state
# =========================
VALID_ACTIONS = {"LEFT", "RIGHT", "STRAIGHT", "STOP"}


def write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_session(args) -> Path:
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    session_dir = args.capture_dir / name
    session_dir.mkdir()
    write_metadata(session_dir / "session.json", {
        "schema_version": 1,
        "session": name,
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stream_url": args.stream_url,
        "motor_url": args.motor_url,
        "save_fps": args.save_fps,
        "image_size": [FRAME_WIDTH, FRAME_HEIGHT],
        "jpeg_quality": JPEG_QUALITY,
    })
    return session_dir


def review_session(session_dir: Path, capture_dir: Path, output_dir: Path,
                   decision: str) -> Path:
    if decision not in {"keep", "discard", "pending"}:
        raise ValueError("Decision must be keep, discard, or pending")
    session_dir = session_dir.resolve()
    if session_dir.parent != capture_dir.resolve() or not session_dir.name.startswith("session_"):
        raise ValueError("Review only a direct session_* child of --capture-dir")
    metadata_path = session_dir / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pending" or metadata.get("session") != session_dir.name:
        raise ValueError("Not a pending capture session")
    if decision == "discard":
        shutil.rmtree(session_dir)
        print(f"[Session] discarded: {session_dir}")
    elif decision == "keep":
        destination = output_dir.resolve() / session_dir.name
        if destination == session_dir or session_dir in destination.parents:
            raise ValueError("Kept session must be outside its pending directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        shutil.move(str(session_dir), str(destination))
        metadata["status"] = "kept"
        metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        write_metadata(destination / "session.json", metadata)
        print(f"[Session] kept: {destination}")
        return destination
    else:
        print(f"[Session] pending: {session_dir}")
    return session_dir


def ask_review() -> str:
    while True:
        try:
            choice = input("Keep / discard / pending [k/d/p, default p]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "pending"
        if choice in {"", "p", "pending"}:
            return "pending"
        if choice in {"k", "keep"}:
            return "keep"
        if choice in {"d", "discard"}:
            return "discard"


class ESP32Controller:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.last_sent: Optional[str] = None

    def send_action(self, action: str, force: bool = False) -> bool:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        # Keep normal deduplication; force a fresh STOP request during shutdown.
        if not force and action == self.last_sent:
            return True

        try:
            resp = self.session.get(
                f"{self.base_url}/cmd",
                params={"m": action},  # ESP32 /cmd expects "m".
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            self.last_sent = action
            return True
        except Exception as e:
            # A failed request must not count as the last successful command.
            self.last_sent = None
            print(f"[ESP32] send_action failed for {action}: {e}")
            return False


class MJPEGStreamReader:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def read_latest(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _push_latest(self, frame: np.ndarray) -> None:
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        self.frame_queue.put_nowait(frame)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            resp = None
            try:
                print(f"[Stream] connecting: {self.stream_url}")
                resp = requests.get(self.stream_url, stream=True, timeout=5)
                resp.raise_for_status()

                buffer = b""
                for chunk in resp.iter_content(chunk_size=4096):
                    if self.stop_event.is_set():
                        break
                    if not chunk:
                        continue

                    buffer += chunk
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9")

                    while start != -1 and end != -1 and end > start:
                        jpg = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self._push_latest(frame)

                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9")

            except Exception as e:
                print(f"[Stream] error: {e}; retrying...")
                # Allow shutdown to interrupt the retry delay.
                self.stop_event.wait(1.0)
            finally:
                # Release this streaming connection before retrying or exiting.
                if resp is not None:
                    resp.close()


class DatasetRecorder:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.labels_path = os.path.join(dataset_dir, "labels.csv")
        self.csv_file = open(self.labels_path, "x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)

        if os.path.getsize(self.labels_path) == 0:
            self.writer.writerow([
                "image",
                "action",
                "saved_ts",
                "action_ts",
            ])
            self.csv_file.flush()

        self.frame_id = self._discover_next_frame_id()

    def _discover_next_frame_id(self) -> int:
        existing = []
        for name in os.listdir(self.images_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() == ".jpg" and stem.isdigit():
                existing.append(int(stem))
        return (max(existing) + 1) if existing else 1

    def save(self, frame: np.ndarray, action: str, saved_ts: float, action_ts: float) -> str:
        filename = f"{self.frame_id:06d}.jpg"
        path = os.path.join(self.images_dir, filename)

        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"Failed to save image: {path}")

        self.writer.writerow([filename, action, f"{saved_ts:.6f}", f"{action_ts:.6f}"])
        self.csv_file.flush()
        self.frame_id += 1
        return filename

    def close(self) -> None:
        self.csv_file.close()


class App:
    def __init__(self, args, session_dir):
        self.args = args
        self.session_dir = session_dir
        self.controller = ESP32Controller(args.motor_url)
        self.reader = MJPEGStreamReader(args.stream_url)
        self.recorder = DatasetRecorder(str(session_dir))
        # Serialize motor commands with the final STOP; never hold during disk I/O.
        self.action_lock = threading.Lock()
        # Distinguish a control failure from a normal user exit.
        self.control_failed = False

        self.running = True
        self.current_action = "STOP"

        # Timestamp control key events, excluding repeated presses.
        self.key_ts = time.time()

        self.last_save_ts = 0.0
        self.last_saved_action = None
        self.latest_frame = None

        # Match the web control's keydown repeat filter:
        # commit@1491fd1, aicam_esp32/robot_lane_tracking/keyboard.py.
        self.pressed_keys = set()

    def set_action(self, action):
        with self.action_lock:
            if not self.running:
                # Ignore late commands once shutdown has begun; return None.
                return
            if not self.controller.send_action(action):
                # End capture; main() leaves the failed session pending.
                self.control_failed = True
                self.running = False
                raise RuntimeError("Motor request failed; session remains pending")
            if action != self.current_action:
                self.current_action = action
                print(f"[Action] {action}")

    def on_press(self, key):
        from pynput import keyboard
        if not self.running:
            # pynput interprets False as a request to stop listening.
            return False
        try:
            if key == keyboard.Key.left:
                action = "LEFT"
            elif key == keyboard.Key.right:
                action = "RIGHT"
            elif key == keyboard.Key.up:
                action = "STRAIGHT"
            elif key == keyboard.Key.down or key == keyboard.Key.space:
                action = "STOP"
            elif hasattr(key, "char") and key.char == "q":
                self.running = False
                return False
            else:
                # Ignore unrelated keys; None keeps the listener running.
                return

            if key in self.pressed_keys:
                return
            self.pressed_keys.add(key)
            self.key_ts = time.time()
            self.set_action(action)
        except Exception as e:
            print(f"[Keyboard press] {e}")

    def on_release(self, key):
        from pynput import keyboard
        if not self.running:
            # Prevent release events from sending commands during shutdown.
            return False
        try:
            if key not in (keyboard.Key.left, keyboard.Key.right, keyboard.Key.up,
                           keyboard.Key.down, keyboard.Key.space):
                return
            self.pressed_keys.discard(key)
            self.key_ts = time.time()
            # Only steering releases send a command, as in the web control.
            if key == keyboard.Key.left or key == keyboard.Key.right:
                self.set_action("STRAIGHT")
        except Exception as e:
            print(f"[Keyboard release] {e}")

    def draw_preview(self, frame):
        img = frame.copy()
        cv2.putText(img, f"action: {self.current_action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img, "q: quit", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img

    def loop(self):
        interval = 1.0 / self.args.save_fps

        while self.running:
            frame = self.reader.read_latest()
            if frame is not None:
                self.latest_frame = frame

            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if frame is not None and now - self.last_save_ts >= interval:
                if not self.running:
                    break
                # Copy labels before disk I/O; image writes must not block STOP.
                action, action_ts = self.current_action, self.key_ts
                # Skip consecutive STOP samples without interrupting preview/control.
                if action != "STOP" or self.last_saved_action != "STOP":
                    name = self.recorder.save(frame, action, now, action_ts)
                    print(f"[Saved] {name}, {action}")
                    self.last_save_ts = now
                    self.last_saved_action = action

            if not self.args.no_preview:
                preview = self.draw_preview(self.latest_frame)
                cv2.imshow("dataset recorder", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
                    break

            time.sleep(0.001)

    def run(self):
        # Initialization can fail before a listener or reader thread exists.
        listener = None
        try:
            from pynput import keyboard
            self.set_action("STOP")
            self.reader.start()
            listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()
            self.loop()
        finally:
            # Block late driving commands, then send STOP before other cleanup.
            self.running = False
            if listener is not None:
                # Do not delay STOP by waiting for listener.join().
                listener.stop()
            with self.action_lock:
                # Serialize behind any in-flight command, even if STOP was cached.
                if not self.controller.send_action("STOP", force=True):
                    print("[App] STOP request failed; check the robot directly")
                self.controller.session.close()
            if self.reader.thread.ident is not None:
                # Joining a thread that never started would raise an error.
                self.reader.stop()
            self.recorder.close()
            if not self.args.no_preview:
                cv2.destroyAllWindows()
            metadata_path = self.session_dir / "session.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(ended_at=datetime.now(timezone.utc).isoformat(),
                            saved_frames=self.recorder.frame_id - 1)
            write_metadata(metadata_path, metadata)
            print("[App] done")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-url", help="H618 MJPEG /video_feed URL")
    parser.add_argument("--motor-url", help="ESP32 base URL including its actual port")
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Parent directory for approved sessions")
    parser.add_argument("--save-fps", type=float, default=SAVE_FPS)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--review-session", type=Path, help="Review an existing pending session")
    parser.add_argument("--decision", choices=["keep", "discard", "pending"])
    args = parser.parse_args(argv)
    if args.review_session:
        review_session(args.review_session, args.capture_dir, args.output_dir,
                       args.decision or ask_review())
        return  # Review-only mode must not start a new capture.
    if args.decision:
        parser.error("--decision requires --review-session")
    if not args.stream_url or not args.motor_url:
        parser.error("--stream-url and --motor-url are required for capture")
    if not np.isfinite(args.save_fps) or args.save_fps <= 0:
        parser.error("--save-fps must be finite and greater than zero")
    session_dir = create_session(args)
    print(f"[Session] pending: {session_dir}")
    try:
        app = App(args, session_dir)
        app.run()
        if app.control_failed:
            raise RuntimeError("Motor communication failed during capture")
        if app.controller.last_sent != "STOP":
            raise RuntimeError("Recording ended without a confirmed STOP")
    except KeyboardInterrupt:
        print(f"[Session] interrupted; pending: {session_dir}")
        return  # Leave interrupted data pending; skip normal end-of-run review.
    except Exception:
        print(f"[Session] failed; pending: {session_dir}")
        raise
    review_session(session_dir, args.capture_dir, args.output_dir, ask_review())


if __name__ == "__main__":
    main()
