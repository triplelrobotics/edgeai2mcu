# gpio_test.py / 直接粘到 code.py

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