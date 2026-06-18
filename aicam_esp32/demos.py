# 在这里写上你的代码 :-)
import time
import neopixel
import board
import digitalio
import busio
import wifi # import wifi module
import socketpool
import pwmio
import math

from adafruit_httpserver import Server, Request, Response, Websocket, GET
import json
from asyncio import create_task, gather, run, sleep as async_sleep
from adafruit_motor import motor as Motor

class demos(object):
    def __init__(self):
        pass
    
    def demo_led(self):

        pwm = pwmio.PWMOut(board.IO1, frequency=50)  # output on LED pin with default of 500Hz

        while True:
            for cycle in range(0, 65535):  # Cycles through the full PWM range from 0 to 65535
                pwm.duty_cycle = cycle  # Cycles the LED pin duty cycle through the range of values

            for cycle in range(65534, 0, -1):  # Cycles through the PWM range backwards from 65534 to 0
                pwm.duty_cycle = cycle  # Cycles the LED pin duty cycle through the range of values
        
    def demo_neopixel(self):
        def _cycle_neopixel(pixel, wait):
            for r in range(255):
                pixel[0] = (r, 0, 0)
                time.sleep(wait)
            for r in range(255, 0, -1):
                pixel[0] = (r, 0, 0)
                time.sleep(wait)
            for g in range(255):
                pixel[0] = (0, g, 0)
                time.sleep(wait)
            for g in range(255, 0, -1):
                pixel[0] = (0, g, 0)
                time.sleep(wait)
            for b in range(255):
                pixel[0] = (0, 0, b)
                time.sleep(wait)
            for b in range(255, 0, -1):
                pixel[0] = (0, 0, b)
                time.sleep(wait)
        print("==============================")
        print("ESP32-S3-DevKitC-1/CircuitPython NeoPixel exercise")
        print("to control onboard RGB NeoPixel")
        print("neopixel version: " + neopixel.__version__)
        print("board.NEOPIXEL: ", board.NEOPIXEL)
        print()
        # Create the NeoPixel object
        pixel = neopixel.NeoPixel(board.IO48, 1, pixel_order=neopixel.GRB)
        while True:
            print("I'm blinking ...")
            pixel[0] = (0, 0, 0)
            time.sleep(2.0)

            _cycle_neopixel(pixel, 0.005)

            pixel[0] = (0, 0, 0)
            time.sleep(2.0)
        print("- bye -\n")

    def demo_badMotorCtrl(self):
        # right wheel
        BN1 = digitalio.DigitalInOut(board.IO4)
        BN1.direction = digitalio.Direction.OUTPUT

        BN2 = digitalio.DigitalInOut(board.IO5)
        BN2.direction = digitalio.Direction.OUTPUT

        # left wheel
        AN1 = digitalio.DigitalInOut(board.IO6)
        AN1.direction = digitalio.Direction.OUTPUT

        AN2 = digitalio.DigitalInOut(board.IO7)
        AN2.direction = digitalio.Direction.OUTPUT

        BN1.value = 1
        BN2.value = 1
        AN1.value = 1
        AN2.value = 1
        

        # test the actual volt on the motor
        while True:
            BN1.value = 1
            BN2.value = 0
            AN1.value = 0
            AN2.value = 1
        
        
        # # Bn1:Bn2=1:0 (forward, right wheel); An1:An2=0:1 (forward, left wheel)
        # while True:
        #     time.sleep(30)
        #     BN1.value = 1
        #     BN2.value = 0
        #     AN1.value = 0
        #     AN2.value = 1
        #     time.sleep(3)   # foward run
        #     AN1.value = 1
        #     AN2.value = 1
        #     time.sleep(0.5) # Left wheel stop, turn left 
        #     AN1.value = 0
        #     AN2.value = 1   # (! dont do this. sudden stop and reverse maybe hurtful)
        #     time.sleep(3)   # reverse forward run
        #     AN1.value = 1
        #     AN2.value = 1
        #     BN1.value = 1
        #     BN2.value = 1
        #     BN2.value = 1

    def demo_serialComm(self, uart_id, uart_func):
        if uart_id == 0:
            uart = busio.UART(board.TX, board.RX, baudrate=9600, timeout=1) # uart0
        elif uart_id == 1:
            uart = busio.UART(board.IO17, board.IO18, baudrate=9600, timeout=1) # uart1
        else:
            raise ValueError("Invalid UART ID ...")

        if uart_func == "uart_read":
            print("waiting for the data ...")
            while True:
                try:
                    received_data = uart.readline()  # Read the data from UART
                    print("received:", received_data)    # Decode the received bytes to string
                except Exception as e:
                    print("Error:", e)

        elif uart_func == "uart_write":
            while True:
                try:
                    uart.write(b'Hello') # Send a string over UART
                    print("data sent!")
                    time.sleep(1)  # Wait for a second before sending the next message
                except Exception as e:
                    print("Error:", e)

        elif uart_func == "uart_loopback":
            while True:
                try:
                    uart.write(b'Hello') # Send a string over UART
                    time.sleep(1)  # Wait for a second before sending the next message
                    received_data = uart.readline()  # Read the data from UART
                    print("Received:", received_data)    # Decode the received bytes to string
                except Exception as e:
                    print("Error:", e)

    def demo_wifiAP(self):
        ap_ssid = "mywifiAP"   # set access point credentials
        ap_password = "password123"
        
        print("if the wifi radio is enabled? ", wifi.radio.enabled)

        wifi.radio.start_ap(ssid=ap_ssid, password=ap_password) # configure access point, ssid, password, channel, authmode, and max_connections

        print("if running as an access point? ", wifi.radio.ap_active)
        print("My IP address is", wifi.radio.ipv4_address_ap) # print IP address
        print("My Mac address is", wifi.radio.mac_address_ap)
        print("ipv4 gateway addr: ", wifi.radio.ipv4_gateway_ap)

        pool = socketpool.SocketPool(wifi.radio)
        server = Server(pool, "/", debug=True)
        global websocket
        websocket = None

        @server.route("/", GET)  # no method param == GET
        def base(request: Request):
            """
            Serve a default static plain text message.
            """
            # return Response(request, "Hello from the CircuitPython HTTP Server!")
            # my_str = f"<html><body><h1> Hello! Current time.monotonic is {time.monotonic()}</h1></body></html>"
            my_str = """<html>
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <script src="joystick.min.js"></script>
                                <link rel="stylesheet" type="text/css" href="style.css">
                            </head>
                            <body>
                                <div class="control-panel">
                                    <h1> Virtual Joystick for Robot Control </h1>
                                    <div id="iWantItHere"></div>
                                    <div class="whole">
                                        <div class="data">
                                            <div class="title">State</div>
                                            <div class="content" id="state">still</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Displacemenet vector</div>
                                            <div class="content" id="vector">x:0, y:0</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Displacemenet value</div>
                                            <div class="content" id="value">0</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Angle (degrees)</div>
                                            <div class="content" id="degrees">-</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Angle (radians)</div>
                                            <div class="content" id="radians">-</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Direction</div>
                                            <div class="content" id="direction">-</div>
                                        </div>
                                    </div>
                                </div>
                               

                                <script type="text/javascript">
                                    document.body.addEventListener('touchmove', function(e){e.preventDefault();}, {passive:false});  // turn off the auto-refreshing of browser window
                                    let ws = new WebSocket('ws://' + location.host + '/connect-websocket');
                                    ws.onopen = function() {
                                        alert("websocket connection opened!");
                                    }
                                    ws.onclose = function() {
                                        alert('WebSocket connection closed');
                                    }
                                    ws.onmessage = function (evt) { 
                                        var received_msg = evt.data;
                                        alert("Client side: msg sent from server is received by client");
                                        
                                    };

                                    const joy = new Joystick(
                                        document.querySelector("#iWantItHere"),
                                        {
                                            scale: 2,
                                            color: "rgb(255, 0, 255)",
                                            strokeColor: "rgb(0, 0, 0)"
                                        });
                                    var joy_stat = {state: "still", x:0, y:0, value:0, deg:0, rad:0, dire:""};
                                    
                                    joy.on("start", () => {
                                        document.querySelector("#state").innerHTML = "moving!";
                                        joy_stat.state = "moving";
                                    });

                                    joy.on("end", () => {
                                        document.querySelector("#state").innerHTML = "still";
                                        joy_stat.state = "still";
                                    });

                                    joy.on("change", () => {
                                        joyx = joy.directionVector().x;
                                        joyy = -joy.directionVector().y;  // fit it to Cartesian coordinate system
                                        joyvalue = joy.displacementValue();
                                        joyrads = -joy.directionAngleRads();  // fit it to Cartesian coordinate system
                                        joydegs = -joy.directionAngleDegs(); // fit it to Cartesian coordinate system
                                        joydire = joy.direction();
                                        document.querySelector("#vector").innerHTML = joyx.toFixed(4) + ', ' + joyy.toFixed(4);
                                        document.querySelector("#value").innerHTML = joyvalue.toFixed(4);
                                        document.querySelector("#radians").innerHTML = joyrads.toFixed(4) || '-';
                                        document.querySelector("#degrees").innerHTML = joydegs.toFixed(4) || '-';
                                        document.querySelector("#direction").innerHTML = joydire || '-';
                                        joy_stat.x = joyx;
                                        joy_stat.y = joyy;
                                        joy_stat.value = joyvalue;
                                        joy_stat.deg = joydegs;
                                        joy_stat.rad = joyrads;
                                        joy_stat.dire = joydire;
                                        ws.send(JSON.stringify(joy_stat));
                                    });
                                    
                                </script>
                            </body>
                        </html>"""
            return Response(request, body=my_str, content_type="text/html")
        
        @server.route("/connect-websocket", GET)
        def connect_client(request: Request):
            global websocket  # pylint: disable=global-statement
            if websocket is not None:
                websocket.close()  # Close any existing connection
            websocket = Websocket(request)
            return websocket

        server.start(str(wifi.radio.ipv4_address_ap))  # instead of serve_forever()
        print("can code go to here?")


        async def handle_http_requests():
            while True:
                server.poll()
                await async_sleep(0)

        async def handle_websocket_requests():
            while True:
                if websocket is not None:
                    if (data := websocket.receive(fail_silently=True)) is not None:
                        print(json.loads(data))
                await async_sleep(0)
        
        async def send_websocket_messages():
            while True:
                if websocket is not None:
                    websocket.send_message("Server side: send msg to client", fail_silently=True)
                await async_sleep(1)

        async def main():
            await gather(
                create_task(handle_http_requests()),
                create_task(handle_websocket_requests()),
                create_task(send_websocket_messages()),
            )


        run(main()) 

    
    def demo_readVoltage(self):
        pass



    
    def demo_joystickMotorCtrl(self):
        # 1. Access joystick info
        ap_ssid = "mywifiAP"   # set access point credentials
        ap_password = "password123"
        
        print("if the wifi radio is enabled? ", wifi.radio.enabled)

        wifi.radio.start_ap(ssid=ap_ssid, password=ap_password) # configure access point, ssid, password, channel, authmode, and max_connections

        print("if running as an access point? ", wifi.radio.ap_active)
        print("My IP address is", wifi.radio.ipv4_address_ap) # print IP address
        print("My Mac address is", wifi.radio.mac_address_ap)
        print("ipv4 gateway addr: ", wifi.radio.ipv4_gateway_ap)

        pool = socketpool.SocketPool(wifi.radio)
        server = Server(pool, "/", debug=True)
        global websocket
        websocket = None

        @server.route("/", GET)  # no method param == GET
        def base(request: Request):
            """
            Serve a default static plain text message.
            """
            # return Response(request, "Hello from the CircuitPython HTTP Server!")
            # my_str = f"<html><body><h1> Hello! Current time.monotonic is {time.monotonic()}</h1></body></html>"
            my_str = """<html>
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <script src="joystick.min.js"></script>
                                <link rel="stylesheet" type="text/css" href="style.css">
                            </head>
                            <body>
                                <div class="control-panel">
                                    <h1> Virtual Joystick Robot Control Client </h1>
                                    <div id="iWantItHere"></div>
                                    <div class="whole">
                                        <div class="data">
                                            <div class="title">State</div>
                                            <div class="content" id="state">still</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Displacemenet vector</div>
                                            <div class="content" id="vector">x:0, y:0</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Displacemenet value</div>
                                            <div class="content" id="value">0</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Angle (degrees)</div>
                                            <div class="content" id="degrees">-</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Angle (radians)</div>
                                            <div class="content" id="radians">-</div>
                                        </div>
                                        <div class="data">
                                            <div class="title">Direction</div>
                                            <div class="content" id="direction">-</div>
                                        </div>
                                    </div>
                                </div>
                               

                                <script type="text/javascript">
                                    document.body.addEventListener('touchmove', function(e){e.preventDefault();}, {passive:false});  // turn off the auto-refreshing of browser window
                                    let ws = new WebSocket('ws://' + location.host + '/connect-websocket');
                                    ws.onopen = function() {
                                        alert("websocket connection opened!");
                                    }
                                    ws.onclose = function() {
                                        alert('WebSocket connection closed');
                                    }
                                    ws.onmessage = function (evt) { 
                                        var received_msg = evt.data;
                                        alert("Client side: msg sent from server is received by client");
                                        
                                    };

                                    const joy = new Joystick(
                                        document.querySelector("#iWantItHere"),
                                        {
                                            scale: 2,
                                            color: "rgb(255, 0, 255)",
                                            strokeColor: "rgb(0, 0, 0)"
                                        });
                                    var joy_stat = {state: "still", x:0, y:0, value:0, deg:0, rad:0, dire:""};
                                    
                                    joy.on("start", () => {
                                        document.querySelector("#state").innerHTML = "moving!";
                                        joy_stat.state = "moving";
                                    });

                                    joy.on("end", () => {
                                        document.querySelector("#state").innerHTML = "still";
                                        joy_stat.state = "still";
                                    });

                                    joy.on("change", () => {
                                        joyx = joy.directionVector().x;
                                        joyy = -joy.directionVector().y;  // fit it to Cartesian coordinate system
                                        joyvalue = joy.displacementValue();
                                        joyrads = -joy.directionAngleRads();  // fit it to Cartesian coordinate system
                                        joydegs = -joy.directionAngleDegs(); // fit it to Cartesian coordinate system
                                        joydire = joy.direction();
                                        document.querySelector("#vector").innerHTML = joyx.toFixed(4) + ', ' + joyy.toFixed(4);
                                        document.querySelector("#value").innerHTML = joyvalue.toFixed(4);
                                        document.querySelector("#radians").innerHTML = joyrads.toFixed(4) || '-';
                                        document.querySelector("#degrees").innerHTML = joydegs.toFixed(4) || '-';
                                        document.querySelector("#direction").innerHTML = joydire || '-';
                                        joy_stat.x = joyx;
                                        joy_stat.y = joyy;
                                        joy_stat.value = joyvalue;
                                        joy_stat.deg = joydegs;
                                        joy_stat.rad = joyrads;
                                        joy_stat.dire = joydire;
                                        });

                                    setInterval(wssend, 20); // 50hz remote joystick signal
                                    function wssend(){
                                        ws.send(JSON.stringify(joy_stat));
                                    }
                                </script>
                            </body>
                        </html>"""
            return Response(request, body=my_str, content_type="text/html")
        
        @server.route("/connect-websocket", GET)
        def connect_client(request: Request):
            global websocket  # pylint: disable=global-statement
            if websocket is not None:
                websocket.close()  # Close any existing connection
            websocket = Websocket(request)
            return websocket

        server.start(str(wifi.radio.ipv4_address_ap))  # instead of serve_forever()
        print("can code go to here?")

        global jsinfo
        jsinfo = None


        async def handle_http_requests():
            while True:
                server.poll()
                await async_sleep(0)

        async def handle_websocket_requests():
            global jsinfo
            while True:
                if websocket is not None:
                    if (data := websocket.receive(fail_silently=False)) is not None:
                        jsinfo = json.loads(data)
                        # print(jsinfo)  # remove print() to avoid some websocket delay error when operations too fast
                await async_sleep(0)
        
        async def send_websocket_messages():
            while True:
                if websocket is not None:
                    websocket.send_message("Server side: send msg to client", fail_silently=True)
                await async_sleep(1)

        # 2. Set up motors
        mtdrv_bin1 = pwmio.PWMOut(board.IO4, frequency=200)
        mtdrv_bin2 = pwmio.PWMOut(board.IO5, frequency=200)
        mtdrv_ain1 = pwmio.PWMOut(board.IO6, frequency=200)
        mtdrv_ain2 = pwmio.PWMOut(board.IO7, frequency=200)
        

        motor_l = Motor.DCMotor(mtdrv_ain1, mtdrv_ain2)
        motor_r = Motor.DCMotor(mtdrv_bin1, mtdrv_bin2)

        OP_DURATION = 5
        DEBUG = False
        FULL_THROTTLE = 0.5 # 12V->6V on my setup
        PART_THROTTLE = 0.5*0.4 # ~40% of my FULL THROTTLE. this cannot be too large otherwise hard to control via joystick
        
        CTRL_MT_ITVL = 0.01 # 10ms, 100hz
        PWM_INC = (1.0 - 0.0) / (0.1 / CTRL_MT_ITVL)
        

        async def motor_spin():
            global jsinfo
            cur_thr = [0, 0]
            while True:
                if jsinfo is not None:
                    jx, jy,  jdeg, jval = jsinfo['x'], jsinfo['y'], jsinfo['deg'], jsinfo['value']
                    # print("jx, jy, jdeg, jval", jx, jy,  jdeg, jval)
                    mtlout, mtrout = joystick_to_motor(jx, jy,  jdeg, jval)
                    # print("mtlout: {}, mtrout: {}".format(mtlout, mtrout))
                    # print("curlout: {}, currout: {}".format(cur_thr[0], cur_thr[1]))
                    apply_mt(mode="ramp", mt_l=motor_l, mt_r=motor_r, targ_thr=[mtlout, mtrout], cur_thr=cur_thr, pwm_inc=PWM_INC)  # cur_thr the list as argument can be maintained globally within while loop
                await async_sleep(CTRL_MT_ITVL)
                

        def basic_ops():
            # Drive forward at full throttle
            motor_l.throttle = -FULL_THROTTLE  # TR: in my setup, *(-1) -> forward.
            if DEBUG: 
                print_motor_status(motor_l)
            time.sleep(OP_DURATION)
            
            # Coast to a stop
            motor_l.throttle = None
            if DEBUG: 
                print_motor_status(motor_l)
            time.sleep(OP_DURATION)
    
            # Drive backwards at 50% throttle
            motor_l.throttle = FULL_THROTTLE * 0.5
            if DEBUG: 
                print_motor_status(motor_l)
            time.sleep(OP_DURATION)

            # Brake to a stop
            motor_l.throttle = 0
            if DEBUG: 
                print_motor_status(motor_l)
            time.sleep(OP_DURATION)

        def print_motor_status(motor):
            if motor == motor_l:
                motor_name = "Left Motor"
            elif motor == motor_r:
                motor_name = "Right Motor"
            else:
                motor_name = "Unknown"
            print(f"Motor {motor_name} throttle is set to {motor.throttle/FULL_THROTTLE}.")

        def joystick_to_motor(jx, jy, jdeg, jval):
            if jy > 0:
                if jx > 0:
                    lout = jval
                    rout = 0 + (jval - 0) * jdeg / 90.0 #  first legend
                elif jx < 0:
                    lout = 0 + (jval - 0) * (180 - jdeg) / 90.0
                    rout = jval
                else:
                    lout = jval
                    rout = jval
            elif jy < 0:
                if jx > 0:
                    lout = -jval
                    rout = -jval + (0 - (-jval)) * math.fabs(jdeg - (-90.0)) / 90.0
                elif jx < 0:
                    lout = -jval + (0 - (-jval)) * math.fabs(jdeg - (-90.0)) / 90.0
                    rout = -jval
                else:
                    lout = -jval
                    rout = -jval
            else:
                if jx > 0:
                    lout = jval
                    rout = 0
                elif jx < 0:
                    lout = 0
                    rout = jval
                else:
                    lout = 0
                    rout = 0
            return lout, rout
        
        
        def apply_mt(mode, mt_l, mt_r, targ_thr, cur_thr=None, pwm_inc=None):
            mt_lout = targ_thr[0]
            mt_rout = targ_thr[1]
            if mode == "instant":
                mt_l.throttle = -1 * PART_THROTTLE * mt_lout  # could be FULL_THROTTLE
                mt_r.throttle = 1 * PART_THROTTLE * mt_rout
            elif mode == "ramp":
                assert cur_thr is not None and pwm_inc is not None
               
                if targ_thr[0] > cur_thr[0]:  # left motor
                    cur_thr[0] += pwm_inc
                elif targ_thr[0] < cur_thr[0]:
                    cur_thr[0] -= pwm_inc
                else:
                    pass
                
                if targ_thr[1] > cur_thr[1]:  # right motor
                    cur_thr[1] += pwm_inc
                elif targ_thr[1] < cur_thr[1]:
                    cur_thr[1] -= pwm_inc
                else:
                    pass
                
                if abs(cur_thr[0]) < pwm_inc and targ_thr[0] == 0: cur_thr[0] = 0  # supress cur_thr to be exactly 0 when it is usually not after substracting pwm_inc,
                if abs(cur_thr[1]) < pwm_inc and targ_thr[1] == 0: cur_thr[1] = 0  # while maintain the gradual slow-down.

                if cur_thr[0] >= 1: # left motor max
                    cur_thr[0] = 1
                elif cur_thr[0] <= -1:
                    cur_thr[0] = -1

                if cur_thr[1] >= 1: # right motor max
                    cur_thr[1] = 1
                elif cur_thr[1] <= -1:
                    cur_thr[1] = -1
                
                mt_l.throttle = -1 * PART_THROTTLE * cur_thr[0] 
                mt_r.throttle = 1 * PART_THROTTLE * cur_thr[1]
 
            else:
                raise ValueError("invalid mt mode!!")
            
            


        async def main():
            await gather(
                create_task(handle_http_requests()),
                create_task(handle_websocket_requests()),
                create_task(send_websocket_messages()),
                create_task(motor_spin()),
            )


        run(main())



if __name__ == "__main__":

    d = demos()
    # d.demo_neopixel()
    d.demo_serialComm(uart_id=0, uart_func="uart_read")


