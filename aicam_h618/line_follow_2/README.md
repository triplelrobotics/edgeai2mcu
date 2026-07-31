# Line Follow 2

This folder is a clean second version of the H618 line-follow runtime.

Assumed network:

```text
Laptop hotspot network: 192.168.50.0/24
Laptop hotspot IP:      192.168.50.1
H618 IP:                192.168.50.20
ESP32 car IP:           set this to its address on the same network
```

## File Roles

- `line_runtime.py`: main H618 line-follow runtime. It owns camera capture, serves browser preview/status, invokes TPU inference, and sends motor decisions.
- `line_tpu_service.py`: local TPU inference service. It owns the EdgeTPU interpreter and accepts Unix-socket inference requests.
- `line_helper.py`: shared non-runnable helpers, including `TpuInferenceServiceStub`, `Esp32MotorServiceStub`, label loading, stability smoothing, camera opening, and RGB frame preprocessing.

## Run

Start the TPU inference service:

```bash
cd aicam_h618/line_follow_2
python3 line_tpu_service.py
```

Start the line-follow runtime:

```bash
python3 line_runtime.py --motor-url http://192.168.50.123:5000 --overlay
```

Open from the laptop hotspot network:

```text
http://192.168.50.20:5000
http://192.168.50.20:5000/status
```

Runtime data flow:

```text
camera capture -> latest frame
  -> preview worker: JPEG/MJPEG browser stream
  -> inference worker: resize 96x96 -> BGR to RGB -> TpuInferenceServiceStub -> line_tpu_service.py
  -> Esp32MotorServiceStub -> ESP32 /lane
```

The inference path does not use HTTP MJPEG or JPEG decode.
