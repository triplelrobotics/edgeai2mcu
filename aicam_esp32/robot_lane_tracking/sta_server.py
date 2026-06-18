import os
import wifi
import socketpool
from adafruit_httpserver import Server


def _wifi_up():
    ssid = os.getenv("CIRCUITPY_WIFI_SSID")
    pwd = os.getenv("CIRCUITPY_WIFI_PASSWORD")
    hostname = os.getenv("HOSTNAME") or "esp32s3"

    if not ssid:
        raise RuntimeError("CIRCUITPY_WIFI_SSID not found in settings.toml")

    try:
        wifi.radio.hostname = hostname
    except Exception:
        pass

    print("[WiFi] Connecting STA to %s ..." % ssid)
    wifi.radio.connect(ssid, pwd)

    ip = wifi.radio.ipv4_address
    gw = wifi.radio.ipv4_gateway

    print("[WiFi] STA connected")
    print("IP:", ip)
    print("GW:", gw)

    return str(ip)


def start_server():
    bind_ip = _wifi_up()
    pool = socketpool.SocketPool(wifi.radio)
    server = Server(pool, "/", debug=False)
    server.start(bind_ip)
    return bind_ip, server