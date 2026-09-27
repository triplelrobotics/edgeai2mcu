from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import numpy as np
from pathlib import Path

from tpu_detect_lib import remote_tpu_detect_invoke, MODEL_LIST

# ===== Config =====
MODEL_NAME = "ssd-mobilenet-v2"
MODEL_FN = MODEL_LIST[MODEL_NAME]["fn"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "test_data_detection"
LABEL_PATH = DATA_DIR / "coco_labels.txt"

THRESHOLD = 0.3
INFER_INTERVAL = 0.2  # ~5 FPS 推理. set to 0.0 if want faster infer speed

# ===== Globals =====
app = Flask(__name__)
camera = None
camera_lock = threading.Lock()

fps_counter = 0
fps_start_time = time.time()
current_fps = 0.0

last_infer_time = 0.0
last_dets = []
last_inf_ms = 0.0

def load_labels(path: str):
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # coco_labels.txt 常见格式： "0 person" / "1 bicycle" 或 "0:person"
                if ":" in line:
                    k, v = line.split(":", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        k, v = parts
                    else:
                        # fallback: treat as name-only
                        continue
                if k.strip().isdigit():
                    labels[int(k.strip())] = v.strip()
    except Exception as e:
        print(f"warn: cannot load labels from {path}: {e}")
    return labels

LABELS = load_labels(LABEL_PATH)

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # use 320 * 240 if want faster stream FPS.
        # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # camera.set(cv2.CAP_PROP_FPS, 60)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def to_rgb_uint8(frame):
    # 兼容灰度摄像头
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # 常见 USB 摄像头是 BGR
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def draw_dets(frame_bgr, dets):
    h, w = frame_bgr.shape[:2]
    for d in dets:
        xmin, ymin, xmax, ymax = d["bbox"]
        # bbox 可能是 float，做边界裁剪
        x1 = int(clamp(xmin, 0, w - 1))
        y1 = int(clamp(ymin, 0, h - 1))
        x2 = int(clamp(xmax, 0, w - 1))
        y2 = int(clamp(ymax, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = LABELS.get(d["id"], str(d["id"]))
        text = f"{name} {d['score']:.2f}"
        cv2.putText(frame_bgr, text, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def generate_frames():
    global fps_counter, fps_start_time, current_fps
    global last_infer_time, last_dets, last_inf_ms

    cam = get_camera()

    while True:
        with camera_lock:
            ok, frame = cam.read()

        if not ok:
            print("can't read camera stream")
            break

        # FPS 统计
        fps_counter += 1
        now = time.time()
        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now

        # 推理（限频）
        if now - last_infer_time > INFER_INTERVAL:
            rgb = to_rgb_uint8(frame)
            if rgb is not None:
                dets, inf_ms = remote_tpu_detect_invoke(MODEL_FN, rgb, THRESHOLD)
                if dets is not None:
                    last_dets = dets
                    last_inf_ms = inf_ms
                last_infer_time = now

        # 画 overlay
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Infer: {last_inf_ms:.1f}ms  thr:{THRESHOLD}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        draw_dets(frame, last_dets)

        # 编码输出
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # if Quality is higher, say 70, the video FPS is lower.
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>camera detection</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
    .container { max-width: 900px; margin: 50px auto; background: white; padding: 20px;
                 border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    img { border: 2px solid #ddd; border-radius: 8px; max-width: 100%; }
  </style>
</head>
<body>
  <div class="container">
    <h1>camera detection</h1>
    <img src="/video_feed" />
    <p>Real-time detection (EdgeTPU server-client). FPS + infer ms overlay.</p>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/status")
def status():
    cam = get_camera()
    opened = cam.isOpened() if cam else False
    return f"camera: {'OK' if opened else 'NO'} | FPS: {current_fps:.1f} | last_infer_ms: {last_inf_ms:.1f} | dets: {len(last_dets)}"

if __name__ == "__main__":
    try:
        print("Start camera detection server...")
        print("Open: http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
