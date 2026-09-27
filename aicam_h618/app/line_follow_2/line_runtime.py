import argparse
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

from line_helper import (
    Esp32MotorServiceStub,
    StabilityTracker,
    TpuInferenceServiceStub,
    load_label_map,
    open_video_capture,
    prepare_rgb_frame,
)


BASE_DIR = Path(__file__).resolve().parent
LINE_FOLLOW_DIR = BASE_DIR.parent / "line_follow"
DEFAULT_LABELS_PATH = LINE_FOLLOW_DIR / "labels.txt"
DEFAULT_SOCKET_PATH = "/tmp/line_tpu.sock"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_ACCESS_IP = "192.168.50.20"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
DEFAULT_MODEL_WIDTH = 96
DEFAULT_MODEL_HEIGHT = 96
DEFAULT_PREVIEW_FPS = 30
DEFAULT_JPEG_QUALITY = 60


app = Flask(__name__)

state_lock = threading.Lock()
frame_condition = threading.Condition()
stop_event = threading.Event()

latest_frame_bgr: Optional[np.ndarray] = None
latest_jpeg: Optional[bytes] = None
latest_prediction = {
    "raw": "NONE",
    "stable": "NONE",
    "stability": 0.0,
    "score": 0.0,
    "tpu_ms": 0.0,
    "socket_ms": 0.0,
}

metrics = {
    "capture_fps": 0.0,
    "infer_fps": 0.0,
    "preview_fps": 0.0,
    "clients": 0,
    "camera_ok": False,
}


def capture_worker(args):
    """Continuously read the USB camera and publish the latest raw BGR frame."""

    global latest_frame_bgr

    cam = open_video_capture(args.camera, args.width, args.height, args.camera_fps)
    frame_count = 0
    fps_start = time.time()

    try:
        while not stop_event.is_set():
            ok, frame = cam.read()
            if not ok:
                metrics["camera_ok"] = False
                print("[camera] failed to read frame")
                time.sleep(0.05)
                continue

            with state_lock:
                latest_frame_bgr = frame
                metrics["camera_ok"] = True

            with frame_condition:
                frame_condition.notify_all()

            frame_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                metrics["capture_fps"] = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now
    finally:
        cam.release()


def inference_worker(args, labels: Dict[int, str], tpu_stub: TpuInferenceServiceStub, motor_stub: Esp32MotorServiceStub):
    """Run lane inference from the latest frame and send stable decisions to the ESP32."""

    tracker = StabilityTracker(args.stability_window)
    last_infer_ts = 0.0
    infer_count = 0
    fps_start = time.time()

    while not stop_event.is_set():
        with frame_condition:
            frame_condition.wait(timeout=0.2)

        now = time.time()
        if now - last_infer_ts < args.infer_interval:
            continue

        with state_lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()

        if frame is None:
            continue

        image_rgb = prepare_rgb_frame(frame, args.model_width, args.model_height)
        socket_start = time.perf_counter()
        scores, inf_ms = tpu_stub.invoke(image_rgb)
        socket_ms = (time.perf_counter() - socket_start) * 1000.0
        last_infer_ts = now

        if scores:
            infer_count += 1
            for item in scores:
                item["label"] = labels.get(item["id"], str(item["id"]))
            best = scores[0]
            raw_label = best["label"]
            score = best["score"]
            stable_label, stability = tracker.update(raw_label)
            motor_label = stable_label if args.motor_source == "stable" else raw_label
            motor_action = motor_label if stability >= args.motor_min_stability else "STOP"
            motor_stub.send_lane(motor_action, score, stability)

            with state_lock:
                latest_prediction.update(
                    {
                        "raw": raw_label,
                        "stable": stable_label,
                        "stability": stability,
                        "score": score,
                        "tpu_ms": inf_ms,
                        "socket_ms": socket_ms,
                    }
                )

            print(
                f"[pred] raw={raw_label} score={score:.3f} "
                f"stable={stable_label} stability={stability:.2f} "
                f"tpu={inf_ms:.1f}ms socket_total={socket_ms:.1f}ms"
            )

        if now - fps_start >= 1.0:
            metrics["infer_fps"] = infer_count / (now - fps_start)
            infer_count = 0
            fps_start = now


def preview_worker(args):
    """Encode the latest frame as MJPEG preview data for browser clients."""

    global latest_jpeg

    min_interval = 1.0 / max(args.preview_fps, 1)
    last_encode_ts = 0.0
    encode_count = 0
    fps_start = time.time()

    while not stop_event.is_set():
        with frame_condition:
            frame_condition.wait(timeout=0.2)

        now = time.time()
        if now - last_encode_ts < min_interval:
            continue

        with state_lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()

        if frame is None:
            continue

        if args.overlay:
            frame = draw_overlay(frame)

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)],
        )
        if not ret:
            continue

        with state_lock:
            latest_jpeg = buffer.tobytes()

        with frame_condition:
            frame_condition.notify_all()

        encode_count += 1
        last_encode_ts = now
        if now - fps_start >= 1.0:
            metrics["preview_fps"] = encode_count / (now - fps_start)
            encode_count = 0
            fps_start = now


def draw_overlay(frame):
    """Draw current prediction and runtime metrics on one preview frame."""

    with state_lock:
        pred = dict(latest_prediction)
        local_metrics = dict(metrics)

    cv2.putText(frame, f"raw: {pred['raw']} {pred['score']:.2f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"stable: {pred['stable']} {pred['stability'] * 100:.0f}%", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"cap {local_metrics['capture_fps']:.1f} infer {local_metrics['infer_fps']:.1f} preview {local_metrics['preview_fps']:.1f}",
                (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame


# Browser preview stream and Flask routes.
def generate_frames():
    """Yield the latest encoded JPEG as a multipart MJPEG stream."""

    metrics["clients"] += 1
    try:
        while not stop_event.is_set():
            with frame_condition:
                frame_condition.wait(timeout=1.0)
            with state_lock:
                jpeg = latest_jpeg

            if jpeg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
    finally:
        metrics["clients"] -= 1


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>line follow runtime</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
    .container { max-width: 880px; margin: 40px auto; background: white; padding: 20px;
                 border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    img { border: 2px solid #ddd; border-radius: 8px; max-width: 100%; }
  </style>
</head>
<body>
  <div class="container">
    <h1>line follow runtime</h1>
    <img src="/video_feed" alt="camera stream">
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the browser preview page."""

    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    """Serve the MJPEG camera preview stream."""

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    """Return current camera, inference, preview, and prediction state."""

    with state_lock:
        pred = dict(latest_prediction)
        local_metrics = dict(metrics)
    return (
        f"camera: {'OK' if local_metrics['camera_ok'] else 'NO'} | "
        f"capture: {local_metrics['capture_fps']:.1f} FPS | "
        f"infer: {local_metrics['infer_fps']:.1f} FPS | "
        f"preview: {local_metrics['preview_fps']:.1f} FPS | "
        f"clients: {local_metrics['clients']} | "
        f"raw={pred['raw']} stable={pred['stable']} score={pred['score']:.3f}"
    )


# Runtime entrypoint.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--access-ip", default=DEFAULT_ACCESS_IP)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--preview-fps", type=int, default=DEFAULT_PREVIEW_FPS)
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--model-width", type=int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--model-height", type=int, default=DEFAULT_MODEL_HEIGHT)
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--infer-interval", type=float, default=0.08)
    parser.add_argument("--stability-window", type=int, default=8)
    parser.add_argument("--motor-url", default=None, help="ESP32 base URL, for example http://192.168.50.123:5000")
    parser.add_argument("--motor-source", choices=["stable", "raw"], default="stable")
    parser.add_argument("--motor-min-stability", type=float, default=0.7)
    parser.add_argument("--motor-interval", type=float, default=0.15)
    parser.add_argument("--motor-timeout", type=float, default=0.25)
    parser.add_argument("--no-motor-stop-on-exit", action="store_true")
    args = parser.parse_args()

    labels = load_label_map(args.labels) if not args.no_inference else {}
    tpu_stub = TpuInferenceServiceStub(args.socket) if not args.no_inference else None
    motor_stub = Esp32MotorServiceStub(args.motor_url, args.motor_timeout, args.motor_interval)

    print("Start line runtime: camera capture, tpu inference, browser preview.")
    print(f"Camera: /dev/video{args.camera} {args.width}x{args.height}@{args.camera_fps}")
    if args.no_inference:
        print("Inference: disabled")
    else:
        print(f"Inference input: {args.model_width}x{args.model_height} RGB via {args.socket}")
    print(f"Preview: MJPEG {args.preview_fps} FPS quality={args.quality}")
    print(f"H618 preview: http://{args.access_ip}:{args.port}")
    print(f"Status:      http://{args.access_ip}:{args.port}/status")

    threads = [
        threading.Thread(target=capture_worker, args=(args,), daemon=True),
        threading.Thread(target=preview_worker, args=(args,), daemon=True),
    ]
    if not args.no_inference:
        threads.append(threading.Thread(target=inference_worker, args=(args, labels, tpu_stub, motor_stub), daemon=True))
    for thread in threads:
        thread.start()

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        stop_event.set()
        with frame_condition:
            frame_condition.notify_all()
        if args.motor_url and not args.no_motor_stop_on_exit:
            motor_stub.send_lane("STOP", 0.0, 0.0, force=True)


if __name__ == "__main__":
    main()
