import argparse
import time
import urllib.request
import os
import socket
import struct
import numpy as np
from PIL import Image
from pycoral.utils.dataset import read_label_file

# --- 核心配置：必须与服务端 MODEL_LIST 对应 ---
SOCKET_PATH = "/tmp/tpu_service.sock"
MODEL_LIST = {
    'efficientnet-edgetpu-l': {'fn': 'efficientnet-edgetpu-L_quant_edgetpu.tflite', 'im_sz': 300, 'labl': 'imagenet'},
    'efficientnet-edgetpu-m': {'fn': 'efficientnet-edgetpu-M_quant_edgetpu.tflite', 'im_sz': 240, 'labl': 'imagenet'},
    'efficientnet-edgetpu-s': {'fn': 'efficientnet-edgetpu-S_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'inception-v1': {'fn': 'inception_v1_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'inception-v2': {'fn': 'inception_v2_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'inception-v3': {'fn': 'inception_v3_299_quant_edgetpu.tflite', 'im_sz': 299, 'labl': 'imagenet'},
    'inception-v4': {'fn': 'inception_v4_299_quant_edgetpu.tflite', 'im_sz': 299, 'labl': 'imagenet'}, 
    'mobilenet-v1-ss': {'fn': 'mobilenet_v1_0.25_128_quant_edgetpu.tflite', 'im_sz': 128, 'labl': 'imagenet'},
    'mobilenet-v1-s': {'fn': 'mobilenet_v1_0.5_160_quant_edgetpu.tflite', 'im_sz': 160, 'labl': 'imagenet'},
    'mobilenet-v1-m': {'fn': 'mobilenet_v1_0.75_192_quant_edgetpu.tflite', 'im_sz': 192, 'labl': 'imagenet'},
    'mobilenet-v1-l': {'fn': 'mobilenet_v1_1.0_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'mobilenet-v1-l-tf2': {'fn': 'tf2_mobilenet_v1_1.0_224_ptq_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'mobilenet-v2-inet-birds': {'fn': 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'inat_bird'},
    'mobilenet-v2-inet-insects': {'fn': 'mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'inat_insect'},
    'mobilenet-v2-inet-plants': {'fn': 'mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'inat_plant'},
    'mobilenet-v2': {'fn': 'mobilenet_v2_1.0_224_quant_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'mobilenet-v2-tf2': {'fn': 'tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
    'mobilenet-v3-tf2': {'fn': 'tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite', 'im_sz': 224,  'labl': 'imagenet'},
    'resnet-50': {'fn': 'tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite', 'im_sz': 224, 'labl': 'imagenet'},
}

def remote_tpu_invoke(model_fn, image_np, top_k, threshold):
    """
    通过 Unix Socket 向常驻服务端发起推理请求
    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_PATH)
        img_bytes = image_np.tobytes()
        fn_bytes = model_fn.encode()
        
        # 协议头: 文件名长度(I), top_k(I), threshold(f), 图片字节长度(I)
        header = struct.pack('IIfI', len(fn_bytes), top_k, threshold, len(img_bytes))
        client.sendall(header + fn_bytes + img_bytes)
        
        # 接收响应: 成功标志(I), 推理耗时(f), 结果数量(I)
        res_header = client.recv(12)
        success, inf_ms, num_res = struct.unpack('IfI', res_header)
        
        if not success:
            return None, 0
        
        results = []
        for _ in range(num_res):
            data = client.recv(8)
            c_id, score = struct.unpack('If', data)
            results.append({'id': c_id, 'score': score})
            
        return results, inf_ms
    except Exception as e:
        print(f"Socket 连接异常: {e}")
        return None, 0
    finally:
        client.close()

def prepare_data(models_to_test, data_folder):
    """保持原有的数据准备逻辑"""
    for model_name in models_to_test:
        model_filename = MODEL_LIST[model_name]['fn']
        label_filename = MODEL_LIST[model_name]['labl'] + '_labels.txt'
        img_filename = 'parrot.jpg' if label_filename == 'imagenet_labels.txt' else 'dragonfly.bmp' if label_filename == 'inat_insect_labels.txt' else 'sunflower.bmp'
        
        for fi, fn in enumerate([model_filename, label_filename, img_filename]):
            file_path = os.path.join(data_folder, fn)
            if not os.path.exists(file_path):
                print(f"正在准备下载 {fn}...")
                url = f"raw.githubusercontent.com{fn}"
                urllib.request.urlretrieve(url, file_path)
    print("所有测试数据已就绪。")

def run_inference(models_to_test, data_folder, top_k, threshold, count):
    """重构后的推理逻辑：将 invoke 替换为 remote 调用"""
    for model_name in models_to_test:
        m_cfg = MODEL_LIST[model_name]
        label_filepath = os.path.join(data_folder, m_cfg['labl'] + '_labels.txt')
        img_filename = 'parrot.jpg' if m_cfg['labl'] == 'imagenet' else 'dragonfly.bmp' if m_cfg['labl'] == 'inat_insect' else 'sunflower.bmp'
        img_filepath = os.path.join(data_folder, img_filename)

        labels = read_label_file(label_filepath)
        
        # 预处理 (在客户端 CPU 上完成)
        image = Image.open(img_filepath).convert('RGB').resize((m_cfg['im_sz'], m_cfg['im_sz']), Image.LANCZOS)
        img_np = np.asarray(image, dtype=np.uint8)

        print(f'\n--- 模型测试: {model_name} (H618 Client -> TPU Server) ---')
        
        for i in range(count):
            # 发起远程调用
            classes, inf_time = remote_tpu_invoke(m_cfg['fn'], img_np, top_k, threshold)
            
            if classes is not None:
                print(f'第 {i+1} 次推理耗时: {inf_time:.1f}ms')
                if i == count - 1: # 只在最后一次打印详细结果
                    print('------- 最终结果 --------')
                    for c in classes:
                        print('%s: %.5f' % (labels.get(c['id'], c['id']), c['score']))
            else:
                print("推理请求失败，请确保服务端正在运行且没有掉线。")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model_name', required=True, help='模型名称或输入 all')
    parser.add_argument('-d', '--data_folder', default='test_data_classification')
    parser.add_argument('-k', '--top_k', type=int, default=1)
    parser.add_argument('-t', '--threshold', type=float, default=0.0)
    parser.add_argument('-c', '--count', type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)

    models_to_test = list(MODEL_LIST.keys()) if args.model_name == 'all' else [args.model_name]
    
    # 步骤 1: 准备本地图片和标签 (Server 端也需要模型文件在对应目录)
    prepare_data(models_to_test, args.data_folder)
    
    # 步骤 2: 执行远程推理
    run_inference(models_to_test, args.data_folder, args.top_k, args.threshold, args.count)

if __name__ == '__main__':
    main()
