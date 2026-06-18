import asyncio
from . import sta_server
from . import keyboard
from . import motor_control


def run():
    bind_ip, server = sta_server.start_server()
    motor = motor_control.MotorController()
    keyboard.register_routes(server, motor)

    print("server started at http://%s:5000" % bind_ip)
    print("open page, click once, then use arrow keys")

    async def handle_http():
        while True:
            server.poll()
            await asyncio.sleep(0)

    async def drive_loop():
        last = None
        while True:
            if motor.get_action() != last:
                last = motor.get_action()
                print("drive_state ->", last)
            motor.apply()
            await asyncio.sleep(0.02)

    async def main():
        await asyncio.gather(
            asyncio.create_task(handle_http()),
            asyncio.create_task(drive_loop()),
        )

    asyncio.run(main())