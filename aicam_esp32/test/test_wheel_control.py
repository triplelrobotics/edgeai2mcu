"""
Interactive wheel hardware check; run with wheels lifted off the ground.
Pure wheel control without robot_lane_tracking.motor_control lib included.
"""

import sys
import time

import board
import pwmio
import supervisor
from adafruit_motor import motor as Motor

# New vehicle reference for this test; application mapping is not changed yet.
LEFT_PINS = (board.IO4, board.IO5)   # BIN1/BIN2
RIGHT_PINS = (board.IO6, board.IO7)  # AIN1/AIN2
PWM_FREQ = 200
LEFT_FORWARD_SIGN = -1  # BIN positive throttle was observed to move backward.
RIGHT_FORWARD_SIGN = 1  # AIN negative throttle was observed to move backward.


STEPS = (
    ("Left forward; right stopped", 1, 0),
    ("Left backward; right stopped", -1, 0),
    ("Right forward; left stopped", 0, 1),
    ("Right backward; left stopped", 0, -1),
    ("Both forward", 1, 1),
    ("Both backward", -1, -1),
    ("Clockwise pivot: left forward, right backward", 1, -1),
    ("Counterclockwise pivot: left backward, right forward", -1, 1),
)


def stop_outputs(outputs):
    # Attempt every pin even if one output fails to reset.
    for output in outputs:
        try:
            output.duty_cycle = 0
        except Exception as exc:
            print("Could not reset PWM; disconnect motor power:", exc)


def pulse(left_motor, right_motor, outputs, left, right, duration):
    try:
        left_motor.throttle = LEFT_FORWARD_SIGN * left if left else None
        right_motor.throttle = RIGHT_FORWARD_SIGN * right if right else None
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if supervisor.runtime.serial_bytes_available:
                key = sys.stdin.read(1)
                if key.lower() == "q":
                    return False
            time.sleep(0.01)
        return True
    finally:
        stop_outputs(outputs)


def run(throttle=0.14, duration=0.75):
    if not 0 < throttle <= 0.30:
        raise ValueError("Use throttle above 0 and at most 0.30")
    if not 0 < duration <= 2.0:
        raise ValueError("Use duration above 0 and at most 2 seconds")

    print("Lift ALL wheels off the ground. Keep motor power disconnect accessible.")
    print("Stop code.py first. This test does not start Wi-Fi or HTTP control.")
    print("Left: BIN IO4/IO5; right: AIN IO6/IO7; forward signs:",
          LEFT_FORWARD_SIGN, RIGHT_FORWARD_SIGN)
    print("Forward faces the vehicle front with BIN on the left and AIN on the right.")
    print("Confirm polarity on hardware; this test does not change application mapping.")
    print("Throttle:", throttle, "Duration:", duration, "seconds per step")
    print("During a pulse: q or Ctrl+C stops the test (best effort).")
    if input("Type ready to initialize outputs; anything else cancels: ").strip().lower() != "ready":
        return

    outputs = []
    try:
        for pin in LEFT_PINS + RIGHT_PINS:
            outputs.append(pwmio.PWMOut(pin, frequency=PWM_FREQ, duty_cycle=0))
        stop_outputs(outputs)
        if input("All wheels must be STOPPED. Type yes to confirm: ").strip().lower() != "yes":
            print("Stopped-state check failed or cancelled. Do not continue.")
            return

        left_motor = Motor.DCMotor(outputs[0], outputs[1])
        right_motor = Motor.DCMotor(outputs[2], outputs[3])
        passed = 0
        for description, left, right in STEPS:
            print("\nStep", passed + 1, "of", len(STEPS), ":", description)
            command = input("Type run to pulse; anything else ends the test: ").strip().lower()
            if command != "run":
                print("Test cancelled; completed checks:", passed)
                return
            if not pulse(left_motor, right_motor, outputs,
                         left * throttle, right * throttle, duration):
                print("Test interrupted; completed checks:", passed)
                return
            answer = input("Correct wheel(s), direction AND stopped afterward? yes/no: ").strip().lower()
            if answer != "yes":
                print("Check failed or unconfirmed:", description)
                print("Check pin alignment, wiring and direction signs before retrying.")
                return
            passed += 1
        print("All", passed, "wheel checks manually confirmed.")
    finally:
        stop_outputs(outputs)
        for output in outputs:
            try:
                output.deinit()
            except Exception as exc:
                print("Could not release PWM:", exc)
        print("PWM reset attempted and outputs released.")


if __name__ == "__main__":
    run()
