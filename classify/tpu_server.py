import os
import socket
import struct
import numpy as np
import subprocess
import time
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter, load_edgetpu_delegate

# --- 服务配置 ---
SOCKET_PATH = "/tmp/tpu_service.sock"
DEVICE = 'usb:0'
DATA_FOLDER = 'test_data_classification'
INPUT_MEAN = 128.0 # 对应原代码参数
INPUT_STD = 128.0  # 对应原代码参数

# 1. 核心修复逻辑：强制重置 USB 总线，防止 Bus ID 从 2 掉到 5
def reset_tpu_hardware():
    print("正在执行总线硬重置，确保 EHCI (480M) 握手...")
    # 同时重置引导模式和运行模式下的 ID
    subprocess.run(["usbreset", "1a6e:0801"], capture_output=True)
    subprocess.run(["usbreset", "18d1:9302"], capture_output=True)
    time.sleep(2) 

# 2. 全局上下文：保持这些变量存活，libedgetpu.so 及其硬件句柄就不会释放
CONTEXT = {
    'delegate': None,
    'interpreter': None,
    'current_model_fn': None
}

def get_interpreter(model_fn):
    model_path = os.path.join(DATA_FOLDER, model_fn)
    if CONTEXT['current_model_fn'] == model_fn and CONTEXT['interpreter']:
        return CONTEXT['interpreter']

    print(f"服务端：正在切换/加载模型 {model_fn} ...")
    if CONTEXT['delegate'] is None:
        # 只加载一次 Delegate
        CONTEXT['delegate'] = load_edgetpu_delegate(options={'device': DEVICE})
    
    # 重新绑定解释器
    CONTEXT['interpreter'] = make_interpreter(model_path, device=DEVICE, delegate=CONTEXT['delegate'])
    CONTEXT['interpreter'].allocate_tensors()
    CONTEXT['current_model_fn'] = model_fn
    return CONTEXT['interpreter']

def start_server():
    reset_tpu_hardware()
    if CONTEXT['delegate'] is None:
        # 只加载一次 Delegate
        print("loading edgetpu delegate ...")
        CONTEXT['delegate'] = load_edgetpu_delegate(options={'device': DEVICE})
    
    if os.path.exists(SOCKET_PATH): os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    os.chmod(SOCKET_PATH, 0o666)
    print(f"TPU 推理常驻服务已启动。硬件句柄已锁定，当前 PID: {os.getpid()}")

    try:
        while True:
            conn, _ = server.accept()
            try:
                # A. 接收协议头: fn_len(I), top_k(I), threshold(f), img_bytes_len(I)
                header = conn.recv(16)
                if not header: continue
                fn_len, top_k, threshold, img_bytes_len = struct.unpack('IIfI', header)
                
                # B. 接收模型文件名和图片原始字节
                model_fn = conn.recv(fn_len).decode()
                img_raw = b''
                while len(img_raw) < img_bytes_len:
                    chunk = conn.recv(img_bytes_len - len(img_raw))
                    if not chunk: break
                    img_raw += chunk
                
                # C. 推理准备
                interpreter = get_interpreter(model_fn)
                width, height = common.input_size(interpreter)
                image_np = np.frombuffer(img_raw, dtype=np.uint8).reshape((height, width, 3))
                
                # D. 核心：量化预处理（从你原代码迁移而来）
                params = common.input_details(interpreter, 'quantization_parameters')
                scale = params['scales']
                zero_point = params['zero_points']
                
                # 检查是否满足跳过预处理的条件
                if np.all(abs(scale * INPUT_STD - 1) < 1e-5) and np.all(abs(INPUT_MEAN - zero_point) < 1e-5):
                    common.set_input(interpreter, image_np)
                else:
                    # q = (input - mean) / (std * scale) + zero_point
                    normalized_input = (image_np.astype(np.float32) - INPUT_MEAN) / (INPUT_STD * scale) + zero_point
                    np.clip(normalized_input, 0, 255, out=normalized_input)
                    common.set_input(interpreter, normalized_input.astype(np.uint8))
                
                # E. 执行推理
                start = time.perf_counter()
                interpreter.invoke()
                inf_time = (time.perf_counter() - start) * 1000
                
                # F. 获取并打包结果
                classes = classify.get_classes(interpreter, top_k, threshold)
                res_header = struct.pack('IfI', 1, inf_time, len(classes))
                res_body = b''
                for c in classes:
                    res_body += struct.pack('If', c.id, c.score)
                conn.sendall(res_header + res_body)

            except Exception as e:
                print(f"服务端异常: {e}")
                conn.sendall(struct.pack('I', 0)) # 发送失败标志
            finally:
                conn.close()
    finally:
        server.close()

if __name__ == "__main__":
    start_server()
