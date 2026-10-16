"""
Interactive application motor check; run with all wheels lifted.
Mainly test the MotorController from robot_lane_tracking.motor_control.
"""

import sys
import time

import supervisor

from robot_lane_tracking.motor_control import MotorController


STEPS = (
    ("STOP", "Both wheels remain stopped"),
    ("STRAIGHT", "Both wheels forward at the same requested throttle"),
    ("LEFT", "Both wheels forward; physical left BIN slower than right AIN"),
    ("RIGHT", "Both wheels forward; physical right AIN slower than left BIN"),
)


def stop(controller):
    controller.set_action("STOP")
    # Attempt both motors even if stopping one raises an exception.
    error = None
    for motor in (controller.motor_r, controller.motor_l):
        try:
            motor.throttle = None
        except Exception as exc:
            print("Could not stop motor; disconnect motor power:", exc)
            error = exc
    if error is not None:
        raise error


def pulse(controller, action, duration):
    try:
        controller.set_action(action)
        controller.apply()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if supervisor.runtime.serial_bytes_available:
                if sys.stdin.read(1).lower() == "q":
                    return False
            time.sleep(0.01)
        return True
    finally:
        stop(controller)


def run(throttle=0.14, turn_delta=0.06, min_effective=0.08, duration=0.75):
    if not 0 < throttle <= 0.30:
        raise ValueError("Use throttle above 0 and at most 0.30")
    if not 0 < turn_delta < throttle:
        raise ValueError("Use turn_delta above 0 and below throttle")
    if not 0 <= min_effective < throttle:
        raise ValueError("Use min_effective at least 0 and below throttle")
    if not 0 < duration <= 2.0:
        raise ValueError("Use duration above 0 and at most 2 seconds")

    print("MotorController hardware check: lift ALL wheels off the ground.")
    print("Keep motor power disconnect accessible. Stop any other motor program.")
    print("Physical left: BIN IO4/IO5; physical right: AIN IO6/IO7.")
    print("Forward uses the same vehicle reference as test_wheel_control.py.")
    print("This checks the application controller, without Wi-Fi/HTTP control.")
    print("Saved speed settings are overridden for this test and are not saved.")
    print("Throttle:", throttle, "Turn delta:", turn_delta,
          "Minimum:", min_effective, "Seconds per step:", duration)
    print("During a pulse: q or Ctrl+C ends the test (best effort).")
    if input("Type ready to initialize motors; anything else cancels: ").strip().lower() != "ready":
        return

    controller = None
    try:
        controller = MotorController()
        stop(controller)
        controller.set_params(base=throttle, turn_delta=turn_delta,
                              min_effective=min_effective)
        if input("Both wheels must be STOPPED. Type yes to confirm: ").strip().lower() != "yes":
            print("Stopped-state check failed or cancelled. Do not continue.")
            return

        for number, (action, description) in enumerate(STEPS, 1):
            print("\nStep", number, "of", len(STEPS), ":", action)
            print(description)
            if input("Type run to test; anything else ends the test: ").strip().lower() != "run":
                print("Test cancelled.")
                return
            if not pulse(controller, action, duration):
                print("Test interrupted.")
                return
            if input("Correct direction/speed AND stopped afterward? yes/no: ").strip().lower() != "yes":
                print("Check failed or unconfirmed:", action)
                return
        print("All", len(STEPS), "application motor checks manually confirmed.")
    finally:
        if controller is not None:
            stop(controller)
        print("Test ended. Soft reload before rerunning to release allocated pins.")


if __name__ == "__main__":
    run()
