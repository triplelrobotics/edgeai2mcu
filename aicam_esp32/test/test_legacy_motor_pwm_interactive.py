"""Legacy keyboard-stepped PWM / DCMotor test.
- Standalone: yes on the board with serial input; no H618 needed.
- Start: execution or import allocates pins; Space/Enter runs each step, q exits.
- Requires: CircuitPython board/pwmio/supervisor, serial input, adafruit_motor.
- Settings: throttle 1.0 (full duty), 2 seconds per step; adjust and lift wheels first.
- Status: not recently verified; old wheel/direction labels differ from today's reference.
- Limits: no exception cleanup; prefer test_wheel_control.py.
"""

import time
import board
import pwmio
import supervisor
import sys
from adafruit_motor import motor as Motor

# ── 引脚定义 ──────────────────────────────────────────────
PIN_BIN1 = board.IO4   # 右轮
PIN_BIN2 = board.IO5
PIN_AIN1 = board.IO6   # 左轮
PIN_AIN2 = board.IO7

PWM_FREQ      = 200
TEST_THROTTLE = 1.0  # 0.4
HOLD_SEC      = 2.0
DUTY          = int(65535 * TEST_THROTTLE)

# ── 工具函数 ──────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

def wait_for_go():
    """阻塞直到收到空格或回车，收到 q 则退出程序"""
    print("  → 按空格/回车启动，q 退出")
    while True:
        if supervisor.runtime.serial_bytes_available:
            ch = sys.stdin.read(1)
            if ch == "q":
                return False
            if ch in (" ", "\r", "\n"):
                return True

def run_timed(start_fn, stop_fn, sec=HOLD_SEC):
    """启动电机，运行 sec 秒后自动停止，期间收到 q 可提前退出，返回 False 表示要退出"""
    start_fn()
    deadline = time.monotonic() + sec
    while time.monotonic() < deadline:
        if supervisor.runtime.serial_bytes_available:
            ch = sys.stdin.read(1)
            if ch == "q":
                stop_fn()
                return False
        time.sleep(0.05)
    stop_fn()
    return True

# ── 阶段 1：原始 PWM 级别 ─────────────────────────────────
section("PHASE 1: Raw PWM pin test (bypass DCMotor)")

raw_ain1 = pwmio.PWMOut(PIN_AIN1, frequency=PWM_FREQ)
raw_ain2 = pwmio.PWMOut(PIN_AIN2, frequency=PWM_FREQ)
raw_bin1 = pwmio.PWMOut(PIN_BIN1, frequency=PWM_FREQ)
raw_bin2 = pwmio.PWMOut(PIN_BIN2, frequency=PWM_FREQ)

all_raw = [raw_ain1, raw_ain2, raw_bin1, raw_bin2]

def stop_all_raw():
    for p in all_raw:
        p.duty_cycle = 0

raw_steps = [
    ([raw_ain1], [raw_ain2], "LEFT  FWD  — AIN1=HIGH AIN2=LOW  → 左轮应正转"),
    ([raw_ain2], [raw_ain1], "LEFT  REV  — AIN1=LOW  AIN2=HIGH → 左轮应反转"),
    ([raw_bin1], [raw_bin2], "RIGHT FWD  — BIN1=HIGH BIN2=LOW  → 右轮应正转"),
    ([raw_bin2], [raw_bin1], "RIGHT REV  — BIN1=LOW  BIN2=HIGH → 右轮应反转"),
]

for hi_pins, lo_pins, desc in raw_steps:
    print(f"\n{desc}")
    if not wait_for_go():
        stop_all_raw()
        print("已停止，程序结束")
        raise SystemExit

    def start_raw():
        for p in hi_pins:
            p.duty_cycle = DUTY
        for p in lo_pins:
            p.duty_cycle = 0

    if not run_timed(start_raw, stop_all_raw):
        print("已停止，程序结束")
        raise SystemExit

stop_all_raw()
raw_ain1.deinit()
raw_ain2.deinit()
raw_bin1.deinit()
raw_bin2.deinit()

# ── 阶段 2：DCMotor API ───────────────────────────────────
section("PHASE 2: DCMotor API test")

ain1 = pwmio.PWMOut(PIN_AIN1, frequency=PWM_FREQ)
ain2 = pwmio.PWMOut(PIN_AIN2, frequency=PWM_FREQ)
bin1 = pwmio.PWMOut(PIN_BIN1, frequency=PWM_FREQ)
bin2 = pwmio.PWMOut(PIN_BIN2, frequency=PWM_FREQ)

motor_l = Motor.DCMotor(ain1, ain2)
motor_r = Motor.DCMotor(bin1, bin2)

def stop_motors():
    motor_l.throttle = None
    motor_r.throttle = None

motor_steps = [
    ( TEST_THROTTLE,  TEST_THROTTLE, "STRAIGHT  — 两轮同向"),
    (-TEST_THROTTLE, -TEST_THROTTLE, "REVERSE   — 两轮同向反转"),
    ( TEST_THROTTLE,  0.0,           "ONLY LEFT — 仅左轮"),
    ( 0.0,            TEST_THROTTLE, "ONLY RIGHT— 仅右轮"),
    ( TEST_THROTTLE, -TEST_THROTTLE, "SPIN CW   — 原地顺时针"),
    (-TEST_THROTTLE,  TEST_THROTTLE, "SPIN CCW  — 原地逆时针"),
]

for l_thr, r_thr, desc in motor_steps:
    print(f"\n{desc}  (L={l_thr:+.2f}, R={r_thr:+.2f})")
    if not wait_for_go():
        stop_motors()
        print("已停止，程序结束")
        raise SystemExit

    def start_motors():
        motor_l.throttle = l_thr
        motor_r.throttle = r_thr

    if not run_timed(start_motors, stop_motors):
        print("已停止，程序结束")
        raise SystemExit

stop_motors()
section("TEST COMPLETE — 所有步骤完成")
