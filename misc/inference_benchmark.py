import time
import timeit
import numpy as np
import sys
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import load_edgetpu_delegate

DEVICE = 'usb'
PRELOADED_DELEGATE = load_edgetpu_delegate(options={'device': DEVICE})

def run_official_style_benchmark(model_path, iterations=200):
    print(f"--- 官方风格基准测试 (2025版) ---")
    print(f"Python 版本: {sys.version}")
    print(f"模型路径: {model_path}")

    # 1. 加载委托并创建解释器
    try:
        # 默认尝试加载 Edge TPU 委托
        interpreter = make_interpreter(*model_path.split('@'), device=DEVICE, delegate=PRELOADED_DELEGATE)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 2. 获取输入张量指针 (实现零拷贝技巧)
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    
    # 官方技巧：通过 tensor() 获取缓冲区引用
    input_tensor = interpreter.tensor(input_index)
    
    # 3. 预填充随机数据（循环外完成，不计入时间）
    np.random.seed(12345)
    random_data = np.random.randint(0, 256, size=input_tensor().shape, dtype=np.uint8)
    input_tensor()[0] = random_data[0] # 直接填充到缓冲区

    # 4. 预热 (Warmup) - 确保硬件进入活跃状态
    print("正在进行预热...")
    for _ in range(10):
        interpreter.invoke()

    # 5. 核心测试逻辑 (模仿 timeit.timeit)
    print(f"正在进行 {iterations} 次迭代...")
    
    # 我们测量多次 invoke 的总时间
    timer = timeit.Timer(lambda: interpreter.invoke())
    total_time = timer.timeit(number=iterations)
    
    avg_latency = (total_time / iterations) * 1000 # 转换为 ms

    print("\n" + "="*30)
    print(f"平均推理延迟: {avg_latency:.2f} ms")
    print(f"估算 FPS: {1000/avg_latency:.1f}")
    print("="*30)
    print("注意：此数据仅包含 interpreter.invoke()，不含图像预处理。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 my_benchmark.py <你的edgetpu.tflite模型路径>")
    else:
        run_official_style_benchmark(sys.argv[1])
