# robot_go_stop_print.py  (CircuitPython)
import wifi
import socketpool
from asyncio import create_task, gather, run, sleep as async_sleep
from adafruit_httpserver import Server, Request, Response, GET

AP_SSID = "mywifiAP"
AP_PASSWORD = "password123"

drive_state = "stop"  # "go" or "stop"

def _start_ap():
    print("wifi enabled?", wifi.radio.enabled)
    wifi.radio.start_ap(ssid=AP_SSID, password=AP_PASSWORD)
    print("ap active?", wifi.radio.ap_active)
    print("AP IP:", wifi.radio.ipv4_address_ap)
    print("AP GW:", wifi.radio.ipv4_gateway_ap)

def run_server():
    global drive_state

    _start_ap()
    pool = socketpool.SocketPool(wifi.radio)
    server = Server(pool, "/", debug=False)

    @server.route("/", GET)
    def index(request: Request):
        body = (
            "<html><body>"
            "<h2>Robot Control (PRINT ONLY)</h2>"
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

    server.start(str(wifi.radio.ipv4_address_ap))
    print("server started. try: http://%s:5000/go" % wifi.radio.ipv4_gateway_ap)

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
            await async_sleep(0.05)

    async def main():
        await gather(
            create_task(handle_http()),
            create_task(drive_loop()),
        )

    run(main())
