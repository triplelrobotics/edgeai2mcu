# Line follow EdgeTPU inference

This folder is isolated from `classify`, `detect`, and `cam_only`.

Architecture:

- `line_stream_server.py` serves the local USB camera as an MJPEG `/video_feed` stream.
- `line_tpu_server.py` keeps the EdgeTPU delegate and TFLite interpreter alive in one background process.
- `line_cam_client.py` reads the MJPEG stream, sends frames to the TPU server, and shows prediction stability. It does not control the car.
- `labels.txt` maps model output indexes to action names.

Typical run on the H618:

```bash
cd aicam_h618/line_follow
python3 line_stream_server.py
```

If the printed active IP is not the WiFi IP you want clients to use, specify it:

```bash
python3 line_stream_server.py --advertise-ip 192.168.0.100
```

In another terminal:

```bash
cd aicam_h618/line_follow
python3 line_tpu_server.py
```

In a third terminal:

```bash
cd aicam_h618/line_follow
python3 line_cam_client.py --stream-url http://127.0.0.1:5000/video_feed
```

If the camera stream server runs on another host, replace the stream URL.

To send lane decisions to the ESP32 motor controller, pass the ESP32 server URL:

```bash
python3 line_cam_client.py --stream-url http://192.168.0.100:5000/video_feed --motor-url http://192.168.0.123:5000
```

Notes:

- The default model path is `tflite_models/model_int8_uint8_edgetpu.tflite`.
- The client displays raw prediction, smoothed prediction, confidence, inference latency, stream FPS, and stability.
- To collect frames for debugging a suspicious prediction, use for example:

```bash
python3 line_cam_client.py --stream-url http://192.168.0.100:5000/video_feed --no-preview --save-pred LEFT
```

Saved frames go to `debug_frames/` by default.
- If labels look swapped, edit `labels.txt` to match the Edge Impulse class order.
