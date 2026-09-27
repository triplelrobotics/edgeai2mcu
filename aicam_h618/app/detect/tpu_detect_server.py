import os
import socket
import struct
import time
import subprocess
import numpy as np
from pathlib import Path
import cv2

from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter, load_edgetpu_delegate

# --- 服务配置 ---
SOCKET_PATH = "/tmp/tpu_detect.sock"
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "test_data_detection"
DEFAULT_MODEL_FN = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" # 默认用你说的 ssd-mobilenet-v2
DEVICE = "usb:0"  # 需要的话改成 usb 或 usb:1

# 全局上下文：保持这些变量存活，libedgetpu.so 及其硬件句柄就不会释放
CONTEXT = {
    "delegate": None,
    "interpreter": None,
    "current_model_fn": None,
}

# 核心修复逻辑：强制重置 USB 总线，防止 Bus ID 从 2 掉到 5
def reset_tpu_hardware():
    """可选：跟你分类 server 一样的硬重置逻辑，提升稳定性"""
    try:
        print("TPU detect server: usbreset ...")
        subprocess.run(["usbreset", "1a6e:0801"], capture_output=True)
        subprocess.run(["usbreset", "18d1:9302"], capture_output=True)
        time.sleep(2)
    except Exception as e:
        print(f"usbreset skipped: {e}")

def get_interpreter(model_fn: str):
    model_path = DATA_FOLDER / model_fn
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if CONTEXT["current_model_fn"] == model_fn and CONTEXT["interpreter"] is not None:
        return CONTEXT["interpreter"]

    print(f"TPU detect server: loading model {model_fn}")
    if CONTEXT["delegate"] is None:
        print("TPU detect server: loading edgetpu delegate ...")
        CONTEXT["delegate"] = load_edgetpu_delegate(options={"device": DEVICE})

    # 重新绑定解释器
    interpreter = make_interpreter(str(model_path), device=DEVICE, delegate=CONTEXT["delegate"])
    interpreter.allocate_tensors()

    CONTEXT["interpreter"] = interpreter
    CONTEXT["current_model_fn"] = model_fn
    return interpreter

def recv_all(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf += chunk
    return buf

def start_server():
    reset_tpu_hardware()
    # 预热 delegate（可选）
    if CONTEXT["delegate"] is None:
        CONTEXT["delegate"] = load_edgetpu_delegate(options={"device": DEVICE})

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    os.chmod(SOCKET_PATH, 0o666)

    print(f"TPU detect server started: {SOCKET_PATH}  pid={os.getpid()}")

    try:
        while True:
            conn, _ = server.accept()
            try:
                # A. 接收协议头: fn_len(I), threshold(f), orig_w(I), orig_h(I), img_bytes_len(I)
                header = recv_all(conn, 20)
                # fn_len, threshold, orig_w, orig_h, img_len = struct.unpack("IfIII", header)
                fn_len, threshold, orig_w, orig_h, img_len = struct.unpack("<IfIII", header)
                if orig_w * orig_h * 3 != img_len:
                    raise ValueError(f"Bad header dims: w={orig_w} h={orig_h} img_len={img_len}")



                # B. 接收模型文件名和图片原始字节
                model_fn = recv_all(conn, fn_len).decode()
                img_raw = recv_all(conn, img_len)

                if not model_fn:
                    model_fn = DEFAULT_MODEL_FN

                # C. 推理准备
                interpreter = get_interpreter(model_fn)
                image = np.frombuffer(img_raw, dtype=np.uint8).reshape((orig_h, orig_w, 3)) # orig image (RGB uint8)

                # D. 预处理. resize on server side + get scale for bbox mapping
                # scale = common.set_resized_input(interpreter, image.shape[:2], lambda size: image)
                _, scale = common.set_resized_input(interpreter, (orig_w, orig_h), lambda size: cv2.resize(image, size))
                
                # E. 执行推理
                start = time.perf_counter()
                interpreter.invoke()
                inf_ms = (time.perf_counter() - start) * 1000.0

                # F. 获取并打包结果
                objs = detect.get_objects(interpreter, threshold, scale)
                # res_header = struct.pack("IfI", 1, float(inf_ms), len(objs)) # response header: success(I), inf_ms(f), num(I)
                res_header = struct.pack("<IfI", 1, float(inf_ms), len(objs))

                res_body = b""
                # per det: id(I), score(f), xmin(f), ymin(f), xmax(f), ymax(f)
                for o in objs:
                    b = o.bbox
                    # res_body += struct.pack("Ifffff", int(o.id), float(o.score), float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax))
                    res_body += struct.pack("<Ifffff", int(o.id), float(o.score), float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax))


                conn.sendall(res_header + res_body)

            except Exception as e:
                print(f"TPU detect server error: {e}")
                # send minimal failure
                try:
                    conn.sendall(struct.pack("I", 0))
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        server.close()

if __name__ == "__main__":
    start_server()
