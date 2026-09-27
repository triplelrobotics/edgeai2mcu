import socket
import struct
import numpy as np

SOCKET_PATH = "/tmp/tpu_detect.sock"

# 只放你现在要用的（够用且最稳）
# warning: efficientdet-lite3x is not compatible with usb.
MODEL_LIST = {'ssd-mobilenet-v1':{'fn': 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite', 'im_sz': 300, 'labl': 'coco'},
                'ssd-mobilenet-v2': {'fn': 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite', 'im_sz': 300, 'labl': 'coco'},
                'ssd-mobilenet-v2-tf2': {'fn': 'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite', 'im_sz': 300, 'labl': 'coco'},
                'ssd-mobilenet-v2-faces': {'fn': 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite', 'im_sz': 320, 'labl': 'coco'},
                'ssdlite-mobileDet': {'fn': 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', 'im_sz': 320, 'labl': 'coco'},
                'efficientdet-lite0': {'fn': 'efficientdet_lite0_320_ptq_edgetpu.tflite', 'im_sz': 320, 'labl': 'coco'},
                'efficientdet-lite1': {'fn': 'efficientdet_lite1_384_ptq_edgetpu.tflite', 'im_sz': 384, 'labl': 'coco'},
                'efficientdet-lite2': {'fn': 'efficientdet_lite2_448_ptq_edgetpu.tflite', 'im_sz': 448, 'labl': 'coco'},
                'efficientdet-lite3': {'fn': 'efficientdet_lite3_512_ptq_edgetpu.tflite', 'im_sz': 512, 'labl': 'coco'},
                'ssd-fpn-mobilenet-v1-tf2': {'fn': 'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite', 'im_sz': 640, 'labl': 'coco'}}

def recv_all(client: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = client.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving")
        buf += chunk
    return buf

def remote_tpu_detect_invoke(model_fn: str, image_rgb_u8: np.ndarray, threshold: float = 0.3):
    """
    image_rgb_u8: HxWx3 uint8 RGB (原始分辨率即可，不要提前resize)
    return: (detections, inf_ms)
      detections: [{'id':int,'score':float,'bbox':(xmin,ymin,xmax,ymax)}]
    """
    if image_rgb_u8.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    h, w = image_rgb_u8.shape[:2]
    img_bytes = image_rgb_u8.tobytes()
    fn_bytes = (model_fn or "").encode()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_PATH)

        # header: fn_len(I), threshold(f), orig_w(I), orig_h(I), img_bytes_len(I)
        # header = struct.pack("IfIII", len(fn_bytes), float(threshold), int(w), int(h), len(img_bytes))
        header = struct.pack("<IfIII", len(fn_bytes), float(threshold), int(w), int(h), len(img_bytes))

        client.sendall(header + fn_bytes + img_bytes)

        # response header: success(I), inf_ms(f), num(I)
        res_header = recv_all(client, 12)
        # success, inf_ms, num = struct.unpack("IfI", res_header)
        success, inf_ms, num = struct.unpack("<IfI", res_header)

        if not success:
            return None, 0.0

        dets = []
        # per det: id(I), score(f), xmin(f), ymin(f), xmax(f), ymax(f) => 24 bytes
        for _ in range(num):
            data = recv_all(client, 24)
            # cid, score, xmin, ymin, xmax, ymax = struct.unpack("Ifffff", data)
            cid, score, xmin, ymin, xmax, ymax = struct.unpack("<Ifffff", data)

            dets.append({"id": int(cid), "score": float(score), "bbox": (xmin, ymin, xmax, ymax)})

        return dets, float(inf_ms)

    except Exception as e:
        print(f"detect client socket error: {e}")
        return None, 0.0
    finally:
        client.close()
