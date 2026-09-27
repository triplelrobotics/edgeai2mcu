import argparse
import re
import subprocess

from flask import Flask, Response, render_template_string
import cv2
import threading
import time


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

app = Flask(__name__)

camera = None
camera_lock = threading.Lock()

fps_counter = 0
fps_start_time = time.time()
current_fps = 0.0


def get_active_ip():
    """Return the active non-loopback IPv4 address, preferring RUNNING WiFi interfaces."""

    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        matches = re.findall(
            r"^(\S+):\s+flags=\d+<([^>]*)>[\s\S]*?\n\s+inet\s+(\d+\.\d+\.\d+\.\d+)",
            result.stdout,
            flags=re.MULTILINE,
        )
        running = [
            (name, ip) for name, flags, ip in matches
            if ip != "127.0.0.1" and "UP" in flags.split(",") and "RUNNING" in flags.split(",")
        ]
        if running:
            running.sort(key=lambda item: not item[0].lower().startswith(("wlan", "wifi", "wl")))
            return running[0][1]
    except Exception:
        pass

    raise RuntimeError("No UP,RUNNING non-loopback IPv4 address found. Use --advertise-ip to specify one.")


def get_camera():
    """Open the local USB camera once and return the shared OpenCV VideoCapture."""

    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 60)
    return camera


def generate_frames():
    """Yield JPEG frames as a multipart MJPEG stream for browser or line_cam_client consumers."""

    global fps_counter, fps_start_time, current_fps

    cam = get_camera()
    while True:
        with camera_lock:
            ok, frame = cam.read()

        if not ok:
            print("can't read camera stream")
            break

        fps_counter += 1
        now = time.time()
        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    """HTTP endpoint consumed by line_cam_client to receive camera frames."""

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>line camera stream</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
    .container { max-width: 800px; margin: 50px auto; background: white; padding: 20px;
                 border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    img { border: 2px solid #ddd; border-radius: 8px; max-width: 100%; }
  </style>
</head>
<body>
  <div class="container">
    <h1>line camera stream</h1>
    <img src="/video_feed" alt="camera stream">
    <p>Local camera MJPEG stream for line-follow inference.</p>
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve a simple browser preview of the local camera stream."""

    return render_template_string(HTML_TEMPLATE)


@app.route("/status")
def status():
    """Return camera-open state and current stream FPS."""

    cam = get_camera()
    opened = cam.isOpened() if cam else False
    return f"camera: {'OK' if opened else 'NO'} | FPS: {current_fps:.1f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--advertise-ip",
        default=None,
        help="IP printed for other devices to connect to, e.g. the WiFi IP 192.168.0.100.",
    )
    args = parser.parse_args()

    try:
        advertise_ip = args.advertise_ip or get_active_ip()
        print("Start line camera stream server...")
        print(f"Listening on: {args.host}:{args.port}")
        print(f"Local preview: http://127.0.0.1:{args.port}")
        print(f"Active preview: http://{advertise_ip}:{args.port}")
        print(f"Video feed:     http://{advertise_ip}:{args.port}/video_feed")
        print(f"Status:         http://{advertise_ip}:{args.port}/status")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
