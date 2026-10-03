from demos import demos


if __name__ == "__main__":
    d = demos()
    # d.demo_neopixel()
    # d.demo_serialComm(uart_id=0, uart_func="uart_read")
    # d.demo_wifiAP()
    # pass
    # d.demo_badMotorCtrl()
    # d.demo_led()
    d.demo_joystickMotorCtrl()