import board
import pwmio
import json
from adafruit_motor import motor as Motor

PIN_BIN1 = board.IO4   # right
PIN_BIN2 = board.IO5
PIN_AIN1 = board.IO6   # left
PIN_AIN2 = board.IO7

PWM_FREQ = 200
LEFT_SIGN = 1    # -1
RIGHT_SIGN = -1  # 1

CONFIG_PATH = "robot_lane_tracking/config.json"

# DEFAULT_BASE = 0.13
# DEFAULT_TURN_DELTA = 0.07
# DEFAULT_MIN_EFFECTIVE = 0.05

DEFAULT_BASE = 0.14
DEFAULT_TURN_DELTA = 0.06
DEFAULT_MIN_EFFECTIVE = 0.08


def clamp(x, lo=0.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class MotorController:
    def __init__(self):
        # left
        ain1 = pwmio.PWMOut(PIN_AIN1, frequency=PWM_FREQ)
        ain2 = pwmio.PWMOut(PIN_AIN2, frequency=PWM_FREQ)

        # right
        bin1 = pwmio.PWMOut(PIN_BIN1, frequency=PWM_FREQ)
        bin2 = pwmio.PWMOut(PIN_BIN2, frequency=PWM_FREQ)

        self.motor_l = Motor.DCMotor(ain1, ain2)
        self.motor_r = Motor.DCMotor(bin1, bin2)

        self.state = "STOP"
        self.base = DEFAULT_BASE
        self.turn_delta = DEFAULT_TURN_DELTA
        self.min_effective = DEFAULT_MIN_EFFECTIVE
        self.estopped = False

        self.load_config()

    def set_action(self, action):
        if action in ("LEFT", "STRAIGHT", "RIGHT", "STOP"):
            self.state = action

    def set_lane_action(self, action):
        if not self.estopped:
            self.set_action(action)

    def estop(self):
        self.estopped = True
        self.state = "STOP"

    def arm(self):
        self.estopped = False
        self.state = "STOP"

    def is_estopped(self):
        return self.estopped

    def get_action(self):
        return self.state

    def set_params(self, base=None, turn_delta=None, min_effective=None):
        if base is not None:
            self.base = clamp(base)
        if turn_delta is not None:
            self.turn_delta = clamp(turn_delta)
        if min_effective is not None:
            self.min_effective = clamp(min_effective)

    def get_params(self):
        return {
            "base": self.base,
            "turn_delta": self.turn_delta,
            "min_effective": self.min_effective,
        }

    def load_config(self):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            self.set_params(
                base=cfg.get("base"),
                turn_delta=cfg.get("turn_delta"),
                min_effective=cfg.get("min_effective"),
            )
            print("Loaded config:", self.get_params())
        except Exception as e:
            print("Using default motor config:", e)

    def save_config(self):
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.get_params(), f)
        print("Saved config:", self.get_params())

    def _nz(self, v):
        if v <= 0.0:
            return 0.0
        return clamp(max(v, self.min_effective))

    def _pair(self):
        b = self.base
        d = self.turn_delta

        if self.state == "LEFT":
            # return self._nz(b - d), self._nz(b)
            return self._nz(b), self._nz(b - d)
        if self.state == "RIGHT":
            # return self._nz(b), self._nz(b - d)
            return self._nz(b - d), self._nz(b)
        if self.state == "STRAIGHT":
            return self._nz(b), self._nz(b)

        return None, None

    def apply(self):
        left, right = self._pair()

        if left is None:
            self.motor_l.throttle = None
            self.motor_r.throttle = None
            return

        self.motor_l.throttle = LEFT_SIGN * left
        self.motor_r.throttle = RIGHT_SIGN * right
