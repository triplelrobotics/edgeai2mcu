"""Legacy browser joystick launcher.
- Standalone: no; requires test_legacy_small_demos.py, browser assets and a client browser.
- Start: execute as __main__ to launch demo_joystickMotorCtrl().
- Requires: /test/test_legacy_small_demos.py and all its dependencies.
- Setup: serve JS/CSS at the HTTP root; connect the client to the demo AP.
- Status: not recently verified; review old mapping, lift wheels; no disconnect timeout.
"""

import sys

# exec(open(...).read()) from the device root does not add /test to sys.path.
if "/test" not in sys.path:
    sys.path.append("/test")
from test_legacy_small_demos import demos


if __name__ == "__main__":
    d = demos()
    # d.demo_neopixel()
    # d.demo_serialComm(uart_id=0, uart_func="uart_read")
    # d.demo_wifiAP()
    # pass
    # d.demo_badMotorCtrl()
    # d.demo_led()
    d.demo_joystickMotorCtrl()
