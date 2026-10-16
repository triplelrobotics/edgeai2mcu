"""Legacy IO5 voltage probe.
- Standalone: yes on the board; no other project module or H618 needed.
- Purpose: alternate HIGH/LOW every 10 seconds for meter/logic probing.
- Start: execution or import immediately loops; needs CircuitPython board/pwmio.
- Setup: disconnect motor power; IO5 is a motor-driver input.
- Status: not recently verified; HIGH is 100% duty, with no exception cleanup.
"""

import time
import board
import pwmio

PIN = board.IO5   # 👉 你要测试的引脚（可改成 IO4/IO5/IO6）

print("Starting GPIO test on:", PIN)

p = pwmio.PWMOut(PIN, frequency=1000)

while True:
    print("HIGH (3.3V)")
    p.duty_cycle = 65535
    time.sleep(10)

    print("LOW (0V)")
    p.duty_cycle = 0
    time.sleep(10)
