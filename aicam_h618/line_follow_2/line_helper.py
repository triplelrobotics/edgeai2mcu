import socket
import struct
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


MOTOR_ACTIONS = {"LEFT", "RIGHT", "STRAIGHT", "STOP"}


class StabilityTracker:
    """Keep a rolling window of predicted labels and return the majority label plus its ratio."""

    def __init__(self, window: int):
        self.history = deque(maxlen=window)

    def update(self, label: str) -> Tuple[str, float]:
        self.history.append(label)
        counts = Counter(self.history)
        stable_label, stable_count = counts.most_common(1)[0]
        return stable_label, stable_count / len(self.history)


class Esp32MotorServiceStub:
    """Local proxy for sending lane-follow commands to the ESP32 motor service."""

    def __init__(self, base_url: Optional[str], timeout: float, min_interval: float):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.min_interval = min_interval
        self.last_action: Optional[str] = None
        self.last_sent_ts = 0.0

    def send_lane(self, action: str, score: float, stability: float, force: bool = False) -> None:
        if not self.base_url:
            return
        if action not in MOTOR_ACTIONS:
            action = "STOP"

        now = time.time()
        if not force and action == self.last_action and now - self.last_sent_ts < self.min_interval:
            return

        try:
            resp = requests.get(
                f"{self.base_url}/lane",
                params={
                    "m": action,
                    "score": f"{score:.3f}",
                    "stability": f"{stability:.3f}",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self.last_action = action
            self.last_sent_ts = now
            print(f"[motor] sent {action} score={score:.3f} stability={stability:.2f}")
        except Exception as e:
            print(f"[motor] send failed: {e}")


class TpuInferenceServiceStub:
    """Local proxy for invoking the TPU inference service over a Unix socket."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def _recv_response_bytes(self, sock: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed while receiving")
            buf += chunk
        return buf

    def invoke(self, image_rgb_u8: np.ndarray) -> Tuple[Optional[List[dict]], float]:
        if image_rgb_u8.dtype != np.uint8:
            raise ValueError("image must be uint8")
        if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB")

        h, w = image_rgb_u8.shape[:2]
        img_bytes = image_rgb_u8.tobytes()

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
            header = struct.pack("<III", int(w), int(h), len(img_bytes))
            client.sendall(header + img_bytes)

            res_header = self._recv_response_bytes(client, 12)
            success, inf_ms, num_scores = struct.unpack("<IfI", res_header)
            if not success:
                return None, 0.0

            scores = []
            for _ in range(num_scores):
                data = self._recv_response_bytes(client, 8)
                class_id, score = struct.unpack("<If", data)
                scores.append({"id": int(class_id), "score": float(score)})

            scores.sort(key=lambda item: item["score"], reverse=True)
            return scores, float(inf_ms)
        except Exception as e:
            print(f"line tpu service error: {e}")
            return None, 0.0
        finally:
            client.close()


def load_label_map(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    idx, name = line.split(":", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    idx, name = parts
                labels[int(idx.strip())] = name.strip()
    except FileNotFoundError:
        pass
    return labels


def open_video_capture(camera_index: int, width: int, height: int, camera_fps: int):
    cam = cv2.VideoCapture(camera_index)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, camera_fps)
    return cam


def prepare_rgb_frame(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize first, then convert only the small model input from BGR to RGB."""

    resized_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)