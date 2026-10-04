import argparse
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests


DEFAULT_STREAM_URL = "http://192.168.0.100:5000/video_feed"
DEFAULT_OUTPUT_DIR = Path("dataset") / "hard_examples"
CLASS_NAMES = ("LEFT", "RIGHT", "STRAIGHT")
SAVE_KEYS = {
    ord("l"): "LEFT",
    ord("r"): "RIGHT",
    ord("s"): "STRAIGHT",
}
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
JPEG_QUALITY = 90


class MJPEGStreamReader:
    """Read an MJPEG /video_feed URL in a background thread."""

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


def save_frame(frame_bgr: np.ndarray, output_dir: Path, label: str) -> Path:
    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((now % 1.0) * 1000)
    path = out_dir / f"{ts}_{ms:03d}_{label}.jpg"

    resized = cv2.resize(frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")
    return path


def draw_preview(frame_bgr: np.ndarray, last_saved: str, counts: dict) -> np.ndarray:
    preview = frame_bgr.copy()
    lines = [
        "Hard example collector",
        "l=LEFT  r=RIGHT  s=STRAIGHT  q=quit",
        f"saved LEFT={counts['LEFT']} RIGHT={counts['RIGHT']} STRAIGHT={counts['STRAIGHT']}",
    ]
    if last_saved:
        lines.append(f"last: {last_saved}")

    y = 28
    for line in lines:
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        y += 28
    return preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    for label in CLASS_NAMES:
        (args.output_dir / label).mkdir(parents=True, exist_ok=True)

    reader = MJPEGStreamReader(args.stream_url)
    counts = {label: 0 for label in CLASS_NAMES}
    last_saved = ""

    print("[collector] focus the preview window, then press l/r/s to save the visible frame")
    print(f"[collector] output_dir={args.output_dir}")

    reader.start()
    try:
        latest_frame = None
        while True:
            frame = reader.read_latest()
            if frame is not None:
                latest_frame = frame

            if latest_frame is None:
                time.sleep(0.01)
                continue

            cv2.imshow("hard example collector", draw_preview(latest_frame, last_saved, counts))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in SAVE_KEYS:
                label = SAVE_KEYS[key]
                try:
                    path = save_frame(latest_frame, args.output_dir, label)
                    counts[label] += 1
                    last_saved = f"{label} {path.name}"
                    print(f"[saved] {label} {path}")
                except Exception as e:
                    print(f"[save] {e}")

            time.sleep(0.001)
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
