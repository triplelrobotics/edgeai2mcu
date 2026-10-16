"""Named startup modes for code.py.
- Import only the selected application; importing this module starts nothing.
- Test scripts run as __main__ in their own namespace.
- Add reviewed entry points here and document them in README.md.
"""

def _run_script(path):
    with open(path, "r") as source:
        script = source.read()
    exec(script, {"__name__": "__main__", "__file__": path})


def _run_lane_tracking():
    from robot_lane_tracking.robot_app import run as run_lane_tracking

    run_lane_tracking()


def _run_motor_check():
    _run_script("test/test_motor_controller.py")


def _run_wheel_check():
    _run_script("test/test_wheel_control.py")


_MODES = {
    "lane_tracking": _run_lane_tracking,
    "motor_check": _run_motor_check,
    "wheel_check": _run_wheel_check,
}


def run(mode):
    start = _MODES.get(mode)
    if start is None:
        raise ValueError("Unknown startup mode: %s. Use %s." % (mode, ", ".join(_MODES)))
    start()
