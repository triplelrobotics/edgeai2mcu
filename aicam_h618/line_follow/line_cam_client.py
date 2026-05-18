import argparse
import os
import queue
import socket
import struct
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STREAM_URL = "http://127.0.0.1:5000/video_feed"
DEFAULT_LABELS_PATH = BASE_DIR / "labels.txt"
DEFAULT_DEBUG_DIR = BASE_DIR / "debug_frames"
SOCKET_PATH = "/tmp/line_tpu.sock"
DEBUG_FRAME_WIDTH = 224
DEBUG_FRAME_HEIGHT = 224
DEBUG_JPEG_QUALITY = 90


class MJPEGStreamReader:
    """Read an MJPEG /video_feed URL in a background thread and expose only the latest BGR frame."""

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
                print(f"[stream] connecting: {self.stream_url}")
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
                print(f"[stream] error: {e}; retrying...")
                time.sleep(1.0)


class StabilityTracker:
    """Keep a rolling window of predicted labels and return the majority label plus its ratio."""

    def __init__(self, window: int):
        self.history = deque(maxlen=window)

    def update(self, label: str) -> tuple[str, float]:
        self.history.append(label)
        counts = Counter(self.history)
        stable_label, stable_count = counts.most_common(1)[0]
        return stable_label, stable_count / len(self.history)


def recv_response_bytes(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from the TPU server response socket."""

    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf += chunk
    return buf


def load_labels(path: Path) -> Dict[int, str]:
    """Load class-id to label-name mappings from labels.txt for display."""

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


def remote_line_invoke(
    image_rgb_u8: np.ndarray,
    socket_path: str = SOCKET_PATH,
) -> Tuple[Optional[List[dict]], float]:
    """
    Send one RGB frame to the persistent EdgeTPU line-follow server.

    image_rgb_u8: HxWx3 uint8 RGB. The server handles model resize and quantization.
    return: (scores, inf_ms)
      scores: [{'id': int, 'score': float}, ...] sorted by score descending.
    """
    if image_rgb_u8.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    h, w = image_rgb_u8.shape[:2]
    img_bytes = image_rgb_u8.tobytes()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(socket_path)

        # request header: orig_w(I), orig_h(I), img_bytes_len(I)
        header = struct.pack("<III", int(w), int(h), len(img_bytes))
        client.sendall(header + img_bytes)

        # response header: success(I), inf_ms(f), num_scores(I)
        res_header = recv_response_bytes(client, 12)
        success, inf_ms, num_scores = struct.unpack("<IfI", res_header)
        if not success:
            return None, 0.0

        scores = []
        for _ in range(num_scores):
            data = recv_response_bytes(client, 8)
            class_id, score = struct.unpack("<If", data)
            scores.append({"id": int(class_id), "score": float(score)})

        scores.sort(key=lambda item: item["score"], reverse=True)
        return scores, float(inf_ms)

    except Exception as e:
        print(f"line tpu socket error: {e}")
        return None, 0.0
    finally:
        client.close()


def to_rgb_uint8(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV camera frame to HxWx3 uint8 RGB for model inference."""

    if frame_bgr.ndim == 2:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def draw_overlay(frame, raw_label, stable_label, stability, score, inf_ms, stream_fps, infer_fps, scores):
    """Draw the latest raw prediction, smoothed prediction, scores, and timing on a BGR preview frame."""

    y = 28
    color = (0, 255, 0) if stability >= 0.7 else (0, 220, 255)
    cv2.putText(frame, f"raw: {raw_label} {score:.2f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    y += 30
    cv2.putText(frame, f"stable: {stable_label} {stability * 100:.0f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    y += 30
    cv2.putText(frame, f"infer: {inf_ms:.1f}ms {infer_fps:.1f}Hz  stream: {stream_fps:.1f}FPS", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    y += 28
    summary = "  ".join(f"{item['label']}:{item['score']:.2f}" for item in scores[:3])
    cv2.putText(frame, summary, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)


def maybe_save_debug_frame(frame_bgr, debug_dir, save_pred, raw_label, stable_label, score, min_interval, last_save_ts):
    """Save a 224x224 JPEG like data_recorder_edit.py when raw prediction matches save_pred."""

    if not save_pred or raw_label != save_pred:
        return last_save_ts

    now = time.time()
    if now - last_save_ts < min_interval:
        return last_save_ts

    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((now % 1.0) * 1000)
    filename = f"{ts}_{ms:03d}_raw-{raw_label}_stable-{stable_label}_score-{score:.2f}.jpg"
    path = debug_dir / filename
    resized = cv2.resize(frame_bgr, (DEBUG_FRAME_WIDTH, DEBUG_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), DEBUG_JPEG_QUALITY])
    if ok:
        print(f"[debug] saved {path}")
        return now

    print(f"[debug] failed to save {path}")
    return last_save_ts


def main() -> None:
    """Run the camera-stream client loop: read frames, invoke TPU server, and preview predictions."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--socket", default=SOCKET_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--infer-interval", type=float, default=0.08)
    parser.add_argument("--stability-window", type=int, default=8)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-pred", choices=["LEFT", "RIGHT", "STRAIGHT"], default=None)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--save-interval", type=float, default=0.5)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    reader = MJPEGStreamReader(args.stream_url)
    tracker = StabilityTracker(args.stability_window)

    last_infer_ts = 0.0
    last_inf_ms = 0.0
    last_raw_label = "NONE"
    last_stable_label = "NONE"
    last_stability = 0.0
    last_score = 0.0
    last_scores = []

    stream_fps = 0.0
    infer_fps = 0.0
    frame_count = 0
    infer_count = 0
    fps_start = time.time()
    last_debug_save_ts = 0.0

    reader.start()
    try:
        preview_enabled = not args.no_preview
        while True:
            frame = reader.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            now = time.time()

            if now - last_infer_ts >= args.infer_interval:
                rgb = to_rgb_uint8(frame)
                scores, inf_ms = remote_line_invoke(rgb, socket_path=args.socket)
                last_infer_ts = now
                if scores:
                    infer_count += 1
                    last_inf_ms = inf_ms
                    for item in scores:
                        item["label"] = labels.get(item["id"], str(item["id"]))
                    best = scores[0]
                    last_raw_label = best["label"]
                    last_score = best["score"]
                    last_stable_label, last_stability = tracker.update(last_raw_label)
                    last_scores = scores
                    print(
                        f"[pred] raw={last_raw_label} score={last_score:.3f} "
                        f"stable={last_stable_label} stability={last_stability:.2f} "
                        f"infer={last_inf_ms:.1f}ms"
                    )
                    last_debug_save_ts = maybe_save_debug_frame(
                        frame,
                        args.debug_dir,
                        args.save_pred,
                        last_raw_label,
                        last_stable_label,
                        last_score,
                        args.save_interval,
                        last_debug_save_ts,
                    )

            if now - fps_start >= 1.0:
                elapsed = now - fps_start
                stream_fps = frame_count / elapsed
                infer_fps = infer_count / elapsed
                frame_count = 0
                infer_count = 0
                fps_start = now

            if preview_enabled:
                preview = frame.copy()
                draw_overlay(
                    preview,
                    last_raw_label,
                    last_stable_label,
                    last_stability,
                    last_score,
                    last_inf_ms,
                    stream_fps,
                    infer_fps,
                    last_scores,
                )
                try:
                    cv2.imshow("line follow inference", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    print(f"[preview] OpenCV GUI is unavailable; continuing without preview: {e}")
                    preview_enabled = False
                    continue

    finally:
        reader.stop()
        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
