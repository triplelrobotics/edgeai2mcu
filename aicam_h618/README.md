# AICam H618

The H618 board captures camera frames, runs Coral USB Edge TPU inference, and
sends lane decisions to the [ESP32 controller](../aicam_esp32/README.md).
Training and compilation happen in the [laptop workflow](../aicam_laptop/README.md).

## Layout

| Local path | Device path | Purpose |
| --- | --- | --- |
| `app/` | `/workspace/aicam_coral/app/` | Synchronized code and deployment models |
| `var/` (ignored) | `/workspace/aicam_coral/var/` | Captures, caches, logs, and archives |
| `setup_device_layout.sh` | Run remotely through SSH | Create the directory layout |
| `compare_device_app.sh` | Run on the Mac | Compare app file paths and SHA-256 contents |

Device data directories include `var/captures/line_follow/`,
`var/cache/coral/{classify,detect,segment}/`, `var/log/`, and `var/run/`.
Creating directories does not automatically redirect every process's output:
console logs remain on stdout, and the line-follow socket currently defaults
to `/tmp/line_tpu.sock`.

## Deployment from the Mac

Run from the repository root, replacing `H618_IP` with the current address:

```bash
git lfs pull
ssh root@H618_IP 'sh -s' < aicam_h618/setup_device_layout.sh
sh aicam_h618/compare_device_app.sh root@H618_IP
```

The setup script creates directories; it does not upload or move files. The
comparison report lists file paths and hashes, excluding Python caches and
`.DS_Store`. Review expected differences before deployment.

For VS Code SFTP, copy [.vscode/sftp.example.json](../.vscode/sftp.example.json)
to `.vscode/sftp.json` and fill in the connection details. It maps
`aicam_h618/app/` to `/workspace/aicam_coral/app/`, with upload-on-save disabled
and sync deletion enabled. Review the target, run local-to-remote sync, and
compare again. Keep `var/` outside app synchronization.

The shell helpers and this README stay outside `app/`; they are workstation
tools/documentation and do not need uploading.

## Device prerequisites

Line-follow requires Python 3, OpenCV, NumPy, Flask, Requests, PyCoral, the Edge
TPU runtime, and camera/Coral USB access. H618 system and Python package versions
are not yet pinned in this repository. Its dependencies are separate from the
Apple Silicon laptop environment.

## Line-follow implementations

| Folder | Design |
| --- | --- |
| `app/line_follow/` | Original MJPEG camera server, TPU server, and prediction/control client |
| `app/line_follow_2/` | Second camera runtime with preview, inference, and motor workers, plus a TPU service |

Both versions use `app/line_follow/labels.txt` and its `tflite_models/` folder.
Their default model is `model_int8_uint8_edgetpu_run_20260607_133850.tflite`;
the unversioned model is the earlier deployment. See
[model lineage](../aicam_laptop/MODEL_LINEAGE.json).

Run one implementation at a time: they share port 5000, the camera, and the TPU
socket by default. Set `ESP32_BASE_URL` below to the actual ESP32 URL including
its verified port.

### Second implementation

In the first H618 SSH terminal:

```bash
cd /workspace/aicam_coral/app/line_follow_2
python3 line_tpu_service.py
```

In another H618 terminal:

```bash
cd /workspace/aicam_coral/app/line_follow_2
python3 line_runtime.py --access-ip H618_IP --overlay
```

Open `http://H618_IP:5000/` for preview or `/status` for status. Add
`--motor-url ESP32_BASE_URL` to enable motor requests. Inference receives camera
frames directly through a Unix socket, without MJPEG decoding. `line_helper.py`
contains camera, preprocessing, smoothing, and service-client helpers.

### Original implementation

Run each command in a separate H618 terminal, from
`/workspace/aicam_coral/app/line_follow/`:

```bash
python3 line_stream_server.py --advertise-ip H618_IP
python3 line_tpu_server.py
python3 line_cam_client.py --stream-url http://127.0.0.1:5000/video_feed
```

The client shows prediction, confidence, latency, FPS, and stability. Add
`--motor-url ESP32_BASE_URL` for lane control. To capture selected predictions:

```bash
python3 line_cam_client.py --no-preview --save-pred LEFT
```

Frames default to `/workspace/aicam_coral/var/captures/line_follow/`.
`--debug-dir` and `--save-interval` control destination and frequency. Review
captures after testing, transfer useful material to the laptop, and delete
device copies after verifying their backup or when no longer needed.

## Other demos

`app/` also contains `audio_detect`, `cam_only`, `classify`, `detect`,
`pose_estimate`, `segment`, and `misc`. They are independent demos/utilities.
Downloaded classification, detection, and segmentation test data belongs in
the corresponding `var/cache/coral/` directories.
