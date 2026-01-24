from flask import Flask, Response, render_template_string
import cv2
import threading
import time

app = Flask(__name__)

# 全局变量
camera = None
camera_lock = threading.Lock()
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # 使用默认摄像头
        # --- 核心修改点：强制启用 MJPG 格式 ---
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 设置摄像头参数（可选）
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 60)
    return camera

def generate_frames():
    global fps_counter, fps_start_time, current_fps
    camera = get_camera()
    while True:
        with camera_lock:
            success, frame = camera.read()
        
        if not success:
            # print("无法读取摄像头")
            print("can't read camera stream")
            break
        else:
            # 计算FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:  # 每秒更新一次FPS
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # 在画面上显示FPS
            cv2.putText(frame, f'FPS: {current_fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # time.sleep(0.033)  # 约30FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 简单的HTML页面
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>camera stream</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            text-align: center; 
            background-color: #f0f0f0;
        }
        .container { 
            max-width: 800px; 
            margin: 50px auto; 
            background: white; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        img { 
            border: 2px solid #ddd; 
            border-radius: 8px; 
            max-width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>camera stream</h1>
        <img src="/video_feed" alt="camera stream">
        <p>real time video stream - FPS on the left corner</p>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    camera = get_camera()
    is_opened = camera.isOpened() if camera else False
    return f"摄像头状态: {'已连接' if is_opened else '未连接'} | 当前FPS: {current_fps:.1f}"

if __name__ == '__main__':
    try:
        print("启动摄像头流媒体服务...")
        print("访问地址: http://localhost:5000 (如果使用端口转发)")
        print("或访问: http://[远程主机IP]:5000")
        print("按 Ctrl+C 停止服务")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
