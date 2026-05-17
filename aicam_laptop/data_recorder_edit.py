# A simple tool to record synchronized video frames and control actions for training a self-driving model.
# This is a upgraded verison of the original data_recorder.py
# Use this one.
import csv
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests
from pynput import keyboard


# =========================
# User config
# =========================
ESP32_BASE_URL = "http://192.168.0.16:5000"
STREAM_URL = "http://192.168.0.100:5000/video_feed"  # replace with your H618 stream endpoint
DATASET_DIR = "dataset"
SAVE_FPS = 10.0
JPEG_QUALITY = 90
FRAME_WIDTH = 224   # saved width
FRAME_HEIGHT = 224  # saved height
SHOW_PREVIEW = True
REQUEST_TIMEOUT = 0.25


# =========================
# Internal state
# =========================
VALID_ACTIONS = {"LEFT", "RIGHT", "STRAIGHT", "STOP"}


@dataclass
class ActionState:
    current: str = "STOP"
    pressed_left: bool = False
    pressed_right: bool = False
    pressed_up: bool = False
    pressed_down: bool = False
    last_change_ts: float = 0.0


class ESP32Controller:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.last_sent: Optional[str] = None

    def send_action(self, action: str) -> bool:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if action == self.last_sent:
            return True

        try:
            resp = self.session.get(
                f"{self.base_url}/cmd",
                params={"act": action},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            self.last_sent = action
            return True
        except Exception as e:
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
                time.sleep(1.0)


class DatasetRecorder:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.labels_path = os.path.join(dataset_dir, "labels.csv")
        self.csv_file = open(self.labels_path, "a", newline="", encoding="utf-8")
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
    def __init__(self):
        self.controller = ESP32Controller(ESP32_BASE_URL)
        self.reader = MJPEGStreamReader(STREAM_URL)
        self.recorder = DatasetRecorder(DATASET_DIR)

        self.running = True
        self.current_action = "STOP"

        # 🔥 改这里：记录“每次按键事件”的时间戳（不是仅action变化）
        self.key_ts = time.time()

        self.last_save_ts = 0.0
        self.latest_frame = None

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

    def set_action(self, action):
        if action != self.current_action:
            self.current_action = action
            print(f"[Action] {action}")
        self.controller.send_action(action)

    def recompute_action(self):
        if self.down_pressed:
            return "STOP"
        if self.left_pressed and not self.right_pressed:
            return "LEFT"
        if self.right_pressed and not self.left_pressed:
            return "RIGHT"
        if self.up_pressed:
            return "STRAIGHT"
        if not self.left_pressed and not self.right_pressed:
            return "STRAIGHT"
        return "STOP"

    def on_press(self, key):
        try:
            if key == keyboard.Key.left:
                self.left_pressed = True
            elif key == keyboard.Key.right:
                self.right_pressed = True
            elif key == keyboard.Key.up:
                self.up_pressed = True
            elif key == keyboard.Key.down or key == keyboard.Key.space:
                self.down_pressed = True
            elif hasattr(key, "char") and key.char == "q":
                self.running = False
                return False
            else:
                return

            # 🔥 每次按键都更新时间戳
            self.key_ts = time.time()

            self.set_action(self.recompute_action())
        except Exception as e:
            print(f"[Keyboard press] {e}")

    def on_release(self, key):
        try:
            if key == keyboard.Key.left:
                self.left_pressed = False
            elif key == keyboard.Key.right:
                self.right_pressed = False
            elif key == keyboard.Key.up:
                self.up_pressed = False
            elif key == keyboard.Key.down or key == keyboard.Key.space:
                self.down_pressed = False
            else:
                return

            # 🔥 松键也更新时间戳
            self.key_ts = time.time()

            self.set_action(self.recompute_action())
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
        interval = 1.0 / SAVE_FPS

        while self.running:
            frame = self.reader.read_latest()
            if frame is not None:
                self.latest_frame = frame

            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - self.last_save_ts >= interval:
                try:
                    name = self.recorder.save(
                        self.latest_frame,
                        self.current_action,
                        now,
                        self.key_ts,
                    )
                    print(f"[Saved] {name}, {self.current_action}")
                    self.last_save_ts = now
                except Exception as e:
                    print(f"[Save] {e}")

            if SHOW_PREVIEW:
                preview = self.draw_preview(self.latest_frame)
                cv2.imshow("dataset recorder", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
                    break

            time.sleep(0.001)

    def run(self):
        self.reader.start()
        self.set_action("STOP")

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()

        try:
            self.loop()
        finally:
            self.controller.send_action("STOP")
            self.reader.stop()
            self.recorder.close()
            cv2.destroyAllWindows()
            listener.stop()
            print("[App] done")


if __name__ == "__main__":
    App().run()
