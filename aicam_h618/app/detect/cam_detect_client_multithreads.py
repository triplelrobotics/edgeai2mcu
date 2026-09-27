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

# 摄像头参数（你可以按需改回 640x480）
CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_FPS = 30

# 推流参数
DISPLAY_TARGET_FPS = 22
JPEG_QUALITY = 45

# ===== Globals (Flask) =====
app = Flask(__name__)
camera = None
camera_lock = threading.Lock()

# ===== Shared State (multi-thread) =====
state_lock = threading.Lock()
frame_cond = threading.Condition(state_lock)
stop_event = threading.Event()

latest_frame_bgr = None        # 最新帧（用于显示/编码）
latest_frame_rgb = None        # 最新帧（用于推理）
frame_ts = 0.0                 # 最新帧时间戳

last_dets = []                 # 最近一次推理结果
last_inf_ms = 0.0              # 最近一次推理耗时（来自 TPU server）
det_ts = 0.0                   # 这次推理对应的帧时间戳

# 统计用（显示 fps / 推理 fps）
display_fps = 0.0
infer_fps = 0.0
_display_cnt = 0
_display_t0 = time.time()
_infer_cnt = 0
_infer_t0 = time.time()


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
                        continue
                if k.strip().isdigit():
                    labels[int(k.strip())] = v.strip()
    except Exception as e:
        print(f"warn: cannot load labels from {path}: {e}")
    return labels


LABELS = load_labels(str(LABEL_PATH))


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, CAM_FPS)

        # 可选：打印实际协商结果（只打印一次）
        print("CAP w,h,fps:",
              camera.get(cv2.CAP_PROP_FRAME_WIDTH),
              camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
              camera.get(cv2.CAP_PROP_FPS))
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


# ===== Threads =====
def capture_loop():
    """采集线程：尽可能快读摄像头，更新 latest_frame_*（覆盖旧帧，不排队）"""
    global latest_frame_bgr, latest_frame_rgb, frame_ts

    cam = get_camera()
    while not stop_event.is_set():
        with camera_lock:
            ok, frame = cam.read()

        if not ok:
            time.sleep(0.01)
            continue

        ts = time.time()
        rgb = to_rgb_uint8(frame)

        with frame_cond:
            latest_frame_bgr = frame
            latest_frame_rgb = rgb
            frame_ts = ts
            frame_cond.notify_all()


def infer_loop():
    """推理线程：永远推理最新帧（latest-frame），推理跑满 TPU，不排队"""
    global last_dets, last_inf_ms, det_ts
    global infer_fps, _infer_cnt, _infer_t0

    last_used_ts = 0.0

    while not stop_event.is_set():
        # 等待新帧（避免空转）
        with frame_cond:
            frame_cond.wait_for(lambda: stop_event.is_set() or frame_ts > last_used_ts)
            if stop_event.is_set():
                break

            rgb = None if latest_frame_rgb is None else latest_frame_rgb.copy()
            ts = frame_ts

        # DEBUG：确认推理帧分辨率
        if not hasattr(infer_loop, "_printed"):
            if rgb is not None:
                print("DEBUG infer rgb shape:", rgb.shape)
                infer_loop._printed = True

        if rgb is None or ts <= last_used_ts:
            continue
        last_used_ts = ts

        dets, inf_ms = remote_tpu_detect_invoke(MODEL_FN, rgb, THRESHOLD)
        if dets is None:
            dets = []

        with state_lock:
            last_dets = dets
            last_inf_ms = float(inf_ms)
            det_ts = ts

        # infer fps 统计
        now = time.time()
        _infer_cnt += 1
        if now - _infer_t0 >= 1.0:
            infer_fps = _infer_cnt / (now - _infer_t0)
            _infer_cnt = 0
            _infer_t0 = now


def generate_frames():
    """Flask 推流：高 FPS 输出最新帧 + 叠加最近一次推理结果（策略 A）"""
    global display_fps, _display_cnt, _display_t0

    frame_interval = 1.0 / float(DISPLAY_TARGET_FPS)
    next_t = time.time()

    while not stop_event.is_set():
        # 控制推流节拍（避免 JPEG 编码把 CPU 吃满）
        now = time.time()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += frame_interval

        with state_lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()
            dets = list(last_dets)
            inf_ms = last_inf_ms
            ifps = infer_fps
            f_ts = frame_ts
            d_ts = det_ts

        if frame is None:
            continue

        # display fps 统计（注意：多客户端时会把多个生成器都算进去，属于“总推流速率”）
        _display_cnt += 1
        now = time.time()
        if now - _display_t0 >= 1.0:
            display_fps = _display_cnt / (now - _display_t0)
            _display_cnt = 0
            _display_t0 = now

        # overlay
        cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Infer: {inf_ms:.1f}ms  thr:{THRESHOLD}  IFPS:{ifps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 仅用于观察“结果新鲜度”（策略 A 不做过滤）
        latency_ms = max(0.0, (f_ts - d_ts) * 1000.0) if d_ts > 0 else 0.0
        cv2.putText(frame, f"Latency: {latency_ms:.0f}ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        draw_dets(frame, dets)

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ===== Routes =====
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
    <p>Real-time detection (EdgeTPU server-client). Multi-thread: capture + infer + stream.</p>
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
    with state_lock:
        det_n = len(last_dets)
        inf_ms = last_inf_ms
        ifps = infer_fps
        dfps = display_fps
    return (f"camera: {'OK' if opened else 'NO'} | "
            f"display_fps: {dfps:.1f} | infer_fps: {ifps:.1f} | "
            f"last_infer_ms: {inf_ms:.1f} | dets: {det_n}")


if __name__ == "__main__":
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_inf = threading.Thread(target=infer_loop, daemon=True)

    try:
        print("Start camera detection client (multi-thread)...")
        print("Open: http://localhost:5000")
        t_cap.start()
        t_inf.start()
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        stop_event.set()
        if camera:
            camera.release()
        cv2.destroyAllWindows()
