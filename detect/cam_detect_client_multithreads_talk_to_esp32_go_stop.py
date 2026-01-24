#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import traceback
from collections import deque

import cv2
import numpy as np
import requests
from flask import Flask, Response

from tpu_detect_lib import remote_tpu_detect_invoke, MODEL_LIST

# =========================
# Config
# =========================
CAM_INDEX = 0
CAP_W, CAP_H = 320, 240
CAP_FPS = 30

# ESP32 AP side (you used 192.168.4.1 previously)
ESP32_BASE = "http://192.168.0.92:5000"

# Control loop behavior
CTRL_HZ = 15
GO_MIN_HOLD_S = 0.25
STOP_MIN_HOLD_S = 0.35
HTTP_TIMEOUT_S = 0.15

TARGET_ID = 0          # only use id==0 as "target"
TARGET_SCORE_MIN = 0.0 # set >0 if you want, e.g. 0.35

# Stream
JPEG_QUALITY = 80
STREAM_HZ_CAP = 20  # stream rate cap

# =========================
# Shared state
# =========================
stop_event = threading.Event()

# Latest camera frame (set by capture thread)
frame_cond = threading.Condition()
latest_frame_bgr = None
latest_frame_ts = 0.0

# Latest inference results (set by infer thread)
state_lock = threading.Lock()
last_dets = []          # list of detections (format depends on your TPU lib)
last_inf_ms = 0.0
infer_fps = 0.0
target_present = False  # computed in infer/control path
last_sent_cmd = "none"  # "go" / "stop" / "none"
last_ctrl_err = ""      # last requests error (optional)

# FPS smoothing
_inf_times = deque(maxlen=30)


# =========================
# Detection parsing helpers
# =========================
def _get_det_id_and_score(det):
    """
    Try to robustly extract (class_id, score) from a detection.
    Supports common formats:
      - dict: {"id": int, "score": float} or {"class_id": int, "conf": float}
      - tuple/list: (id, score, x1, y1, x2, y2) or (x1,y1,x2,y2,score,id) etc.
    If unknown, returns (None, None).
    """
    if det is None:
        return None, None

    # dict-like
    if isinstance(det, dict):
        cid = det.get("id", det.get("class_id", det.get("cls_id")))
        score = det.get("score", det.get("conf", det.get("confidence")))
        return cid, score

    # tuple/list-like
    if isinstance(det, (list, tuple)):
        # Most common: (id, score, ...)
        if len(det) >= 2 and isinstance(det[0], (int, np.integer)) and isinstance(det[1], (float, int, np.floating, np.integer)):
            return int(det[0]), float(det[1])

        # Another common: (..., score, id)
        if len(det) >= 2 and isinstance(det[-1], (int, np.integer)) and isinstance(det[-2], (float, int, np.floating, np.integer)):
            return int(det[-1]), float(det[-2])

    return None, None


def compute_target_present(dets) -> bool:
    """
    target is present if any detection has id==TARGET_ID and score>=TARGET_SCORE_MIN
    """
    for d in dets:
        cid, score = _get_det_id_and_score(d)
        if cid is None:
            continue
        if cid == TARGET_ID and (score is None or score >= TARGET_SCORE_MIN):
            return True
    return False


# =========================
# Threads
# =========================
def capture_loop():
    global latest_frame_bgr, latest_frame_ts

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    if not cap.isOpened():
        print("[CAP] Failed to open camera")
        stop_event.set()
        return

    print("[CAP] Camera opened")

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        ts = time.time()
        with frame_cond:
            latest_frame_bgr = frame
            latest_frame_ts = ts
            frame_cond.notify_all()

    cap.release()
    print("[CAP] Stopped")


def infer_loop():
    global last_dets, last_inf_ms, infer_fps, target_present

    print("[INF] Started")
    while not stop_event.is_set():
        # Wait for a fresh frame
        with frame_cond:
            if latest_frame_bgr is None:
                frame_cond.wait(timeout=0.2)
                continue
            frame = latest_frame_bgr.copy()

        # BGR -> RGB for TPU
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        try:
            dets, inf_ms = remote_tpu_detect_invoke(rgb)
        except Exception:
            # Don't crash the thread; just log and keep going
            traceback.print_exc()
            time.sleep(0.05)
            continue

        dt = (time.time() - t0)
        _inf_times.append(dt)
        fps = 1.0 / (sum(_inf_times) / max(1, len(_inf_times)))

        tp = compute_target_present(dets)

        with state_lock:
            last_dets = dets if dets is not None else []
            last_inf_ms = float(inf_ms) if inf_ms is not None else 0.0
            infer_fps = float(fps)
            target_present = bool(tp)

    print("[INF] Stopped")


def control_loop():
    """
    Debounced go/stop sender.
    Independent from video streaming (browser). Works as long as:
      - infer_loop updates target_present
      - Wi-Fi to ESP32 AP is connected
    """
    global last_sent_cmd, last_ctrl_err

    present_since = None
    absent_since = None
    last_sent = None  # "go" / "stop"

    print("[CTL] Started")
    while not stop_event.is_set():
        with state_lock:
            present = bool(target_present)

        now = time.time()
        if present:
            absent_since = None
            if present_since is None:
                present_since = now
        else:
            present_since = None
            if absent_since is None:
                absent_since = now

        want_go = present and (present_since is not None) and (now - present_since >= GO_MIN_HOLD_S)
        want_stop = (not present) and (absent_since is not None) and (now - absent_since >= STOP_MIN_HOLD_S)

        try:
            if want_go and last_sent != "go":
                requests.get(f"{ESP32_BASE}/go", timeout=HTTP_TIMEOUT_S)
                last_sent = "go"
                with state_lock:
                    last_sent_cmd = "go"
                    last_ctrl_err = ""
            elif want_stop and last_sent != "stop":
                requests.get(f"{ESP32_BASE}/stop", timeout=HTTP_TIMEOUT_S)
                last_sent = "stop"
                with state_lock:
                    last_sent_cmd = "stop"
                    last_ctrl_err = ""
        except Exception as e:
            # keep trying; don't block the loop
            with state_lock:
                last_ctrl_err = str(e)[:120]

        time.sleep(1.0 / max(1, CTRL_HZ))

    print("[CTL] Stopped")


# =========================
# Visualization / Streaming
# =========================
def draw_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """
    Keep it generic: draw 'target' status and id.
    (Not writing category name.)
    """
    with state_lock:
        tp = target_present
        infms = last_inf_ms
        fps = infer_fps
        sent = last_sent_cmd
        err = last_ctrl_err

    txt1 = f"target(id={TARGET_ID}): {'YES' if tp else 'NO'}"
    txt2 = f"infer_ms: {infms:.1f}  fps: {fps:.1f}  sent: {sent}"
    cv2.putText(img_bgr, txt1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, txt2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if err:
        cv2.putText(img_bgr, f"ctrl_err: {err}", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return img_bgr


def generate_frames():
    """
    Stream thread: does NOT do detection/control.
    Just reads latest_frame_bgr and overlays status.
    """
    last_emit = 0.0
    while not stop_event.is_set():
        # Cap stream rate so it doesn't burn CPU
        now = time.time()
        if now - last_emit < (1.0 / max(1, STREAM_HZ_CAP)):
            time.sleep(0.001)
            continue

        with frame_cond:
            if latest_frame_bgr is None:
                frame_cond.wait(timeout=0.2)
                continue
            frame = latest_frame_bgr.copy()

        frame = draw_overlay(frame)

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        last_emit = now
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


# =========================
# Flask app
# =========================
app = Flask(__name__)

@app.route("/")
def index():
    # Minimal page
    return (
        "<html><body>"
        "<h3>AI-cam stream</h3>"
        "<img src='/video_feed' />"
        "</body></html>"
    )

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# =========================
# Main
# =========================
def main():
    # Start threads
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_inf = threading.Thread(target=infer_loop, daemon=True)
    t_ctl = threading.Thread(target=control_loop, daemon=True)

    t_cap.start()
    t_inf.start()
    t_ctl.start()

    # Run Flask (visualization only)
    # IMPORTANT: bind 0.0.0.0 so you can view from other devices on that interface
    app.run(host="0.0.0.0", port=5000, threaded=True)

    # If Flask exits, stop threads
    stop_event.set()
    t_cap.join(timeout=1.0)
    t_inf.join(timeout=1.0)
    t_ctl.join(timeout=1.0)


if __name__ == "__main__":
    main()
