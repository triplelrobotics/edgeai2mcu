"""
motor_test.py  —  逐步测试两路电机的每个引脚
运行后观察串口输出，按提示观察轮子是否转动
"""

import time
import board
import pwmio
from adafruit_motor import motor as Motor

# ── 引脚定义（与主程序保持一致）──────────────────────────
PIN_BIN1 = board.IO4   # 右轮
PIN_BIN2 = board.IO5
PIN_AIN1 = board.IO6   # 左轮
PIN_AIN2 = board.IO7

PWM_FREQ  = 200
TEST_THROTTLE = 0.2    # 测试用油门，不要太大
HOLD_SEC  = 2.0        # 每个状态保持秒数

# ── 工具函数 ──────────────────────────────────────────────
def wait(sec=HOLD_SEC):
    time.sleep(sec)

def section(title):
    print("\n" + "="*40)
    print(title)
    print("="*40)

# ── 阶段 1：原始 PWM 级别，绕过 DCMotor ──────────────────
# 直接控制 duty_cycle，确认引脚本身是否工作
# duty_cycle 范围 0 ~ 65535
section("PHASE 1: Raw PWM pin test (bypass DCMotor)")

raw_ain1 = pwmio.PWMOut(PIN_AIN1, frequency=PWM_FREQ)
raw_ain2 = pwmio.PWMOut(PIN_AIN2, frequency=PWM_FREQ)
raw_bin1 = pwmio.PWMOut(PIN_BIN1, frequency=PWM_FREQ)
raw_bin2 = pwmio.PWMOut(PIN_BIN2, frequency=PWM_FREQ)

DUTY = int(65535 * TEST_THROTTLE)

# 左轮正转：AIN1 高，AIN2 低
print("LEFT  FWD  — AIN1=HIGH AIN2=LOW  → 左轮应正转")
raw_ain1.duty_cycle = DUTY
raw_ain2.duty_cycle = 0
wait()

# 左轮反转：AIN1 低，AIN2 高
print("LEFT  REV  — AIN1=LOW  AIN2=HIGH → 左轮应反转")
raw_ain1.duty_cycle = 0
raw_ain2.duty_cycle = DUTY
wait()

# 左轮停止
print("LEFT  STOP")
raw_ain1.duty_cycle = 0
raw_ain2.duty_cycle = 0
wait(0.5)

# 右轮正转：BIN1 高，BIN2 低
print("RIGHT FWD  — BIN1=HIGH BIN2=LOW  → 右轮应正转")
raw_bin1.duty_cycle = DUTY
raw_bin2.duty_cycle = 0
wait()

# 右轮反转：BIN1 低，BIN2 高
print("RIGHT REV  — BIN1=LOW  BIN2=HIGH → 右轮应反转")
raw_bin1.duty_cycle = 0
raw_bin2.duty_cycle = DUTY
wait()

# 全停
print("ALL   STOP")
raw_ain1.duty_cycle = 0
raw_ain2.duty_cycle = 0
raw_bin1.duty_cycle = 0
raw_bin2.duty_cycle = 0

# 必须 deinit，否则下面无法重新初始化同一引脚
raw_ain1.deinit()
raw_ain2.deinit()
raw_bin1.deinit()
raw_bin2.deinit()
wait(0.5)

# ── 阶段 2：通过 DCMotor API 测试 ────────────────────────
section("PHASE 2: DCMotor API test")

ain1 = pwmio.PWMOut(PIN_AIN1, frequency=PWM_FREQ)
ain2 = pwmio.PWMOut(PIN_AIN2, frequency=PWM_FREQ)
bin1 = pwmio.PWMOut(PIN_BIN1, frequency=PWM_FREQ)
bin2 = pwmio.PWMOut(PIN_BIN2, frequency=PWM_FREQ)

motor_l = Motor.DCMotor(ain1, ain2)
motor_r = Motor.DCMotor(bin1, bin2)

steps = [
    # (左油门, 右油门, 描述)
    ( TEST_THROTTLE,  TEST_THROTTLE, "STRAIGHT  — 两轮同向"),
    (-TEST_THROTTLE, -TEST_THROTTLE, "REVERSE   — 两轮同向反转"),
    ( TEST_THROTTLE,  0.0,           "ONLY LEFT — 仅左轮"),
    ( 0.0,            TEST_THROTTLE, "ONLY RIGHT— 仅右轮"),
    ( TEST_THROTTLE, -TEST_THROTTLE, "SPIN CW   — 原地顺时针"),
    (-TEST_THROTTLE,  TEST_THROTTLE, "SPIN CCW  — 原地逆时针"),
]

for l_thr, r_thr, desc in steps:
    print(f"{desc}  (L={l_thr:+.2f}, R={r_thr:+.2f})")
    motor_l.throttle = l_thr
    motor_r.throttle = r_thr
    wait()

# 全停
print("STOP")
motor_l.throttle = None
motor_r.throttle = None

section("TEST COMPLETE — check serial log against observations")