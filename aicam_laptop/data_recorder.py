import csv
import os
import time
import queue
import threading

import cv2
import numpy as np
import requests
from pynput import keyboard


# ====== 改这里 ======
ESP32_BASE_URL = "http://192.168.0.15:5000"
AICAM_STREAM_URL = "http://192.168.0.100:5000/video_feed"

# ESP32_BASE_URL = "http://172.20.10.3:5000"
# AICAM_STREAM_URL = "http://172.20.10.2:5000/video_feed"

DATASET_DIR = "dataset"
SAVE_FPS = 10.0
SAVE_WIDTH = 224
SAVE_HEIGHT = 224
JPEG_QUALITY = 90
SHOW_PREVIEW = True
# ====================


VALID_ACTIONS = {"LEFT", "RIGHT", "STRAIGHT", "STOP"}


class ESP32Controller:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.last_sent = None

    def send_action(self, action: str):
        if action not in VALID_ACTIONS:
            return

        if action == self.last_sent:
            return

        try:
            r = self.session.get(
                f"{self.base_url}/cmd",
                params={"m": action},
                timeout=0.3,
            )
            r.raise_for_status()
            self.last_sent = action
            print(f"[ESP32] {action}")
        except Exception as e:
            print(f"[ESP32] send failed: {e}")


class MJPEGReader:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def get_latest(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _put_latest(self, frame):
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        self.frame_queue.put_nowait(frame)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                print(f"[Stream] connecting {self.stream_url}")
                resp = requests.get(self.stream_url, stream=True, timeout=5)
                resp.raise_for_status()

                buf = b""
                for chunk in resp.iter_content(chunk_size=4096):
                    if self.stop_event.is_set():
                        break
                    if not chunk:
                        continue

                    buf += chunk
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9")

                    while start != -1 and end != -1 and end > start:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]

                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self._put_latest(frame)

                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9")

            except Exception as e:
                print(f"[Stream] error: {e}")
                time.sleep(1)


class Recorder:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.csv_path = os.path.join(dataset_dir, "labels.csv")
        self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)

        if os.path.getsize(self.csv_path) == 0:
            self.writer.writerow(["image", "action", "saved_ts", "action_ts"])
            self.csv_file.flush()

        self.frame_id = self._next_id()

    def _next_id(self):
        nums = []
        for name in os.listdir(self.images_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() == ".jpg" and stem.isdigit():
                nums.append(int(stem))
        return max(nums) + 1 if nums else 1

    def save(self, frame, action, saved_ts, action_ts):
        name = f"{self.frame_id:06d}.jpg"
        path = os.path.join(self.images_dir, name)

        frame = cv2.resize(frame, (SAVE_WIDTH, SAVE_HEIGHT), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("save image failed")

        self.writer.writerow([name, action, f"{saved_ts:.6f}", f"{action_ts:.6f}"])
        self.csv_file.flush()
        self.frame_id += 1
        return name

    def close(self):
        self.csv_file.close()


class App:
    def __init__(self):
        self.controller = ESP32Controller(ESP32_BASE_URL)
        self.reader = MJPEGReader(AICAM_STREAM_URL)
        self.recorder = Recorder(DATASET_DIR)

        self.running = True
        self.current_action = "STOP"
        self.action_ts = time.time()
        self.last_save_ts = 0.0
        self.latest_frame = None

        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

    def set_action(self, action):
        if action != self.current_action:
            self.current_action = action
            self.action_ts = time.time()
            print(f"[Action] {action}")
        self.controller.send_action(action)

    def recompute_action(self):
        # 对齐你原网页逻辑：
        # left/right 按下时转向
        # up 按下时 straight
        # down/space 时 stop
        # 松开 left/right 自动回 straight
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
            frame = self.reader.get_latest()
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
                        self.action_ts,
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