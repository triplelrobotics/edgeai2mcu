import argparse
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.utils.edgetpu import load_edgetpu_delegate, make_interpreter


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "tflite_models" / "model_int8_uint8_edgetpu.tflite"
DEFAULT_SOCKET_PATH = "/tmp/line_tpu.sock"
DEFAULT_DEVICE = "usb:0"
INPUT_MEAN = 127.5
INPUT_STD = 127.5

CONTEXT = {
    "delegate": None,
    "interpreter": None,
    "model_path": None,
}


def reset_tpu_hardware() -> None:
    try:
        print("line tpu server: usbreset ...")
        subprocess.run(["usbreset", "1a6e:0801"], capture_output=True)
        subprocess.run(["usbreset", "18d1:9302"], capture_output=True)
        time.sleep(2.0)
    except Exception as e:
        print(f"line tpu server: usbreset skipped: {e}")


def recv_request_bytes(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf += chunk
    return buf


def get_interpreter(model_path: Path, device: str):
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if CONTEXT["interpreter"] is not None and CONTEXT["model_path"] == model_path:
        return CONTEXT["interpreter"]

    if CONTEXT["delegate"] is None:
        print(f"line tpu server: loading EdgeTPU delegate on {device} ...")
        CONTEXT["delegate"] = load_edgetpu_delegate(options={"device": device})

    print(f"line tpu server: loading model {model_path}")
    interpreter = make_interpreter(str(model_path), device=device, delegate=CONTEXT["delegate"])
    interpreter.allocate_tensors()

    CONTEXT["interpreter"] = interpreter
    CONTEXT["model_path"] = model_path
    return interpreter


def resize_for_model(image_rgb: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    width, height = input_size
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def set_quantized_input(interpreter, image_rgb: np.ndarray) -> None:
    params = common.input_details(interpreter, "quantization_parameters")
    scale = params["scales"]
    zero_point = params["zero_points"]

    # Skip quantization: if the model's input quantization parameters match our preprocessing (i.e., scale * INPUT_STD ~1 and zero_point ~ INPUT_MEAN)
    if np.all(abs(scale * INPUT_STD - 1) < 1e-5) and np.all(abs(INPUT_MEAN - zero_point) < 1e-5):
        common.set_input(interpreter, image_rgb)
        return

    # Quantization preprocessing: first normalize float to [0, 1], then scale and shift back to uint8
    normalized = (image_rgb.astype(np.float32) - INPUT_MEAN) / (INPUT_STD * scale) + zero_point
    np.clip(normalized, 0, 255, out=normalized)
    common.set_input(interpreter, normalized.astype(np.uint8))


def get_output_scores(interpreter):
    output_detail = interpreter.get_output_details()[0]
    output = interpreter.tensor(output_detail["index"])()[0]
    output = np.asarray(output).reshape(-1)

    quant = output_detail.get("quantization", (0.0, 0))
    scale, zero_point = quant
    if scale:
        # inverse quantization: score = (q - zero_point) * scale, make the output to be float32 between 0 and 1.
        scores = (output.astype(np.float32) - float(zero_point)) * float(scale)
    else:
        scores = output.astype(np.float32)

    return [(idx, float(score)) for idx, score in enumerate(scores)]


def handle_request(conn: socket.socket, interpreter) -> None:
    # request header: orig_w(I), orig_h(I), img_bytes_len(I)
    header = recv_request_bytes(conn, 12)
    orig_w, orig_h, img_len = struct.unpack("<III", header)
    if orig_w * orig_h * 3 != img_len:
        raise ValueError(f"Bad frame dims: w={orig_w} h={orig_h} img_len={img_len}")

    img_raw = recv_request_bytes(conn, img_len)
    image = np.frombuffer(img_raw, dtype=np.uint8).reshape((orig_h, orig_w, 3))

    input_size = common.input_size(interpreter)
    input_image = resize_for_model(image, input_size)
    set_quantized_input(interpreter, input_image)

    start = time.perf_counter()
    interpreter.invoke()
    inf_ms = (time.perf_counter() - start) * 1000.0

    scores = get_output_scores(interpreter)
    res_header = struct.pack("<IfI", 1, float(inf_ms), len(scores))
    res_body = b"".join(struct.pack("<If", int(idx), float(score)) for idx, score in scores)
    conn.sendall(res_header + res_body)


def start_server(model_path: Path, socket_path: str, device: str, no_reset: bool) -> None:
    if not no_reset:
        reset_tpu_hardware()

    interpreter = get_interpreter(model_path, device)
    print(f"line tpu server: input_size={common.input_size(interpreter)}")

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(5)
    os.chmod(socket_path, 0o666)

    print(f"line tpu server started: {socket_path} pid={os.getpid()}")

    try:
        while True:
            conn, _ = server.accept()
            try:
                handle_request(conn, interpreter)
            except Exception as e:
                print(f"line tpu server error: {e}")
                try:
                    conn.sendall(struct.pack("<I", 0))
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    start_server(args.model, args.socket, args.device, no_reset=not args.reset)


if __name__ == "__main__":
    main()
