from flask import Flask, Response, render_template_string
import os
# 设置无头模式，避免GUI依赖
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import threading
import time

app = Flask(__name__)

# 全局变量
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(1)  # 使用默认摄像头
        # 设置摄像头参数（可选）
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    camera = get_camera()
    while True:
        with camera_lock:
            success, frame = camera.read()
        
        if not success:
            print("无法读取摄像头")
            break
        else:
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.033)  # 约30FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 简单的HTML页面
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>摄像头流</title>
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
        <h1>远程摄像头流</h1>
        <img src="/video_feed" alt="摄像头流">
        <p>实时摄像头画面</p>
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
    return f"摄像头状态: {'已连接' if is_opened else '未连接'}"

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