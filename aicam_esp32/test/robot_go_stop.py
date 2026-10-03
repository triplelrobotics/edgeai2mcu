# robot_go_stop.py  (CircuitPython)
import os
import time
import board
import pwmio
import wifi
import socketpool

from asyncio import create_task, gather, run, sleep as async_sleep
from adafruit_httpserver import Server, Request, Response, GET
from adafruit_motor import motor as Motor


# --------- Config ----------
AP_SSID = "mywifiAP"
AP_PASSWORD = "password123"

# Motor driver pins (your original mapping)
PIN_BIN1 = board.IO4   # right
PIN_BIN2 = board.IO5
PIN_AIN1 = board.IO6   # left
PIN_AIN2 = board.IO7

PWM_FREQ = 200
# throttle sign: keep your original convention
LEFT_SIGN = -1
RIGHT_SIGN = 1

DRIVE_THROTTLE = 0.11   # tune for your car (0.2~0.6 typical)
POLL_DT = 0.02          # 20ms


# --------- Global state ----------
drive_state = "stop"  # "go" or "stop"


def _setup_motors():
    bin1 = pwmio.PWMOut(PIN_BIN1, frequency=PWM_FREQ)
    bin2 = pwmio.PWMOut(PIN_BIN2, frequency=PWM_FREQ)
    ain1 = pwmio.PWMOut(PIN_AIN1, frequency=PWM_FREQ)
    ain2 = pwmio.PWMOut(PIN_AIN2, frequency=PWM_FREQ)

    motor_l = Motor.DCMotor(ain1, ain2)
    motor_r = Motor.DCMotor(bin1, bin2)
    return motor_l, motor_r


def _set_motors(motor_l, motor_r, state: str):
    if state == "go":
        motor_l.throttle = LEFT_SIGN * DRIVE_THROTTLE
        motor_r.throttle = RIGHT_SIGN * DRIVE_THROTTLE
    else:
        # coast stop (None) is usually gentler than brake (0)
        motor_l.throttle = None
        motor_r.throttle = None


# def _start_ap():
#     print("wifi enabled?", wifi.radio.enabled)
#     wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
#     print("ap active?", wifi.radio.ap_active)
#     print("IP:", wifi.radio.ipv4_address_ap)
#     print("GW:", wifi.radio.ipv4_gateway_ap)

def _wifi_up():
    """
    Prefer STA (connect to existing Wi-Fi). If it fails, fallback to AP.
    Returns: (mode, bind_ip_str, display_ip_str)
      mode: "sta" or "ap"
      bind_ip_str: server.bind address
      display_ip_str: the IP you should open in browser
    """
    ssid = os.getenv("CIRCUITPY_WIFI_SSID")
    pwd  = os.getenv("CIRCUITPY_WIFI_PASSWORD")
    hostname = os.getenv("HOSTNAME") or "esp32s3"

    try:
        wifi.radio.hostname = hostname
    except Exception:
        pass

    # 1) Try STA
    if ssid:
        try:
            print("wifi enabled?", wifi.radio.enabled)
            print(f"[WiFi] Connecting STA to {ssid} ...")
            wifi.radio.connect(ssid, pwd)
            ip = wifi.radio.ipv4_address
            gw = wifi.radio.ipv4_gateway
            print("[WiFi] STA connected")
            print("IP:", ip)
            print("GW:", gw)
            return "sta", str(ip), str(ip)
        except Exception as e:
            print("[WiFi] STA connect failed:", repr(e))

    # 2) Fallback AP (your original behavior)
    print("wifi enabled?", wifi.radio.enabled)
    wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
    print("ap active?", wifi.radio.ap_active)
    print("IP:", wifi.radio.ipv4_address_ap)
    print("GW:", wifi.radio.ipv4_gateway_ap)
    return "ap", str(wifi.radio.ipv4_address_ap), str(wifi.radio.ipv4_gateway_ap)



def run_server():
    global drive_state

    # _start_ap()
    # pool = socketpool.SocketPool(wifi.radio)
    # server = Server(pool, "/", debug=False)

    mode, bind_ip, open_ip = _wifi_up()
    pool = socketpool.SocketPool(wifi.radio)
    server = Server(pool, "/", debug=False)

    motor_l, motor_r = _setup_motors()
    _set_motors(motor_l, motor_r, "stop")

    @server.route("/", GET)
    def index(request: Request):
        body = (
            "<html><body>"
            "<h2>Robot Control</h2>"
            "<p>Use: <a href='/go'>/go</a> | <a href='/stop'>/stop</a> | <a href='/state'>/state</a></p>"
            "</body></html>"
        )
        return Response(request, body=body, content_type="text/html")

    @server.route("/go", GET)
    def go(request: Request):
        global drive_state
        drive_state = "go"
        return Response(request, body="OK go\n", content_type="text/plain")

    @server.route("/stop", GET)
    def stop(request: Request):
        global drive_state
        drive_state = "stop"
        return Response(request, body="OK stop\n", content_type="text/plain")

    @server.route("/state", GET)
    def state(request: Request):
        return Response(request, body=f"{drive_state}\n", content_type="text/plain")

    @server.route("/set", GET)
    def set_params(request: Request):
        global DRIVE_THROTTLE
        # Query string: /set?throttle=0.08
        qs = request.query_params  # adafruit_httpserver exposes parsed query
        if "throttle" in qs:
            try:
                v = float(qs["throttle"])
                # clamp for safety
                if v < 0: v = 0.0
                if v > 1: v = 1.0
                DRIVE_THROTTLE = v
                return Response(request, body=f"OK throttle={DRIVE_THROTTLE}\n", content_type="text/plain")
            except Exception:
                return Response(request, body="ERR bad throttle\n", content_type="text/plain", status=400)

        return Response(request, body="Usage: /set?throttle=0.08\n", content_type="text/plain", status=400)

    @server.route("/status", GET)
    def status(request: Request):
        # return a simple text/json-ish payload to avoid json lib
        try:
            ip = str(wifi.radio.ipv4_address) if wifi.radio.connected else str(wifi.radio.ipv4_address_ap)
        except Exception:
            ip = "unknown"
        body = (
            '{"drive_state":"' + drive_state + '",'
            '"throttle":' + str(DRIVE_THROTTLE) + ','
            '"ip":"' + ip + '"}\n'
        )
        return Response(request, body=body, content_type="application/json")

    # Bind to AP IP
    # server.start(str(wifi.radio.ipv4_address_ap))
    # print("server started at http://%s:5000" % wifi.radio.ipv4_gateway_ap)
    # print("try /go /stop")

    server.start(bind_ip)
    print(f"server started ({mode}) at http://{open_ip}:5000")
    print("try /go /stop /state")

    async def handle_http():
        while True:
            server.poll()
            await async_sleep(0)

    async def drive_loop():
        last = None
        while True:
            if drive_state != last:
                print("drive_state ->", drive_state)
                last = drive_state
            _set_motors(motor_l, motor_r, drive_state)
            await async_sleep(POLL_DT)

    async def main():
        await gather(
            create_task(handle_http()),
            create_task(drive_loop()),
        )

    run(main())
