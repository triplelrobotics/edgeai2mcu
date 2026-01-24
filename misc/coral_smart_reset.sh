#!/bin/bash


# looks like the 'fail to load delegate' issue has something to do with the usb controller type. For bus 004 and 003, 
# both are xHCI controller (specially for usb2.0), the Coral TPU usually can load delegate successfully after running any model inference once
# (ID 1a6e:089a Global Unichip Corp -> ID 18d1:9302 Google Inc). 
# But for bus 001, it is EHCI controller (for usb3.0), the Coral TPU usually fails.
# In case of any loading issue, this script will try to reset the coral usb interface in various ways.

echo "========================================"
echo "    Coral TPU USB Smart Reset Tool"
echo "========================================"

# Check device status function
check_device_status() {
    # Check if Google device or correct device ID exists
    local google_device=$(lsusb | grep -E "Google|18d1:9302")
    local global_unichip_corp=$(lsusb | grep -E "1a6e:089a")
    if [ -n "$google_device" ]; then
        local bus_num=$((10#$(echo "$google_device" | awk '{print $2}')))
        local controller_info=$(lsusb -t | grep -E "Bus 0*${bus_num}" | head -1)
        if [[ $controller_info =~ ehci ]]; then
            echo "✓ Found Google device on EHCI controller, Coral TPU working normally"
            return 0
        else
            echo "✗ Google device found but not on EHCI controller (Bus $bus_number), proceeding to reset"
            return 1
        fi
    elif [ -n "$global_unichip_corp" ]; then
        local bus_num=$((10#$(echo "$global_unichip_corp" | awk '{print $2}')))
        local controller_info=$(lsusb -t | grep -E "Bus 0*${bus_num}" | head -1)
        if [[ $controller_info =~ ehci ]]; then
            echo "✓ Found Coral TPU initial state (1a6e) on EHCI controller, proceeding to load delegate"
            return 2
        else
            echo "✗ Coral TPU initial state (1a6e) found but not on EHCI controller (Bus $bus_number), proceeding to reset"
            return 3
        fi
    else
        echo "✗ No Coral TPU device found"
        return 4
    fi
}

# Test load delegate
test_load_delegate() {
    local attempt="$1"
    echo "Attempt $attempt: Testing Edge TPU delegate loading..."
    
    python3 -c "
import sys
try:
    from pycoral.utils.edgetpu import load_edgetpu_delegate
    DEVICE = 'usb'
    PRELOADED_DELEGATE = load_edgetpu_delegate(options={'device': DEVICE})
    print('SUCCESS: Edge TPU delegate loaded successfully')
    sys.exit(0)
except ImportError:
    print('ERROR: pycoral not installed, please install with: pip3 install pycoral')
    sys.exit(2)
except ValueError as e:
    if 'Failed to load delegate from libedgetpu.so.1.0' in str(e):
        print('FAIL: Failed to load delegate from libedgetpu.so.1.0')
    else:
        print(f'FAIL: ValueError: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Delegate loading failed: {e}')
    sys.exit(1)
" 2>/dev/null
    
    return $?
}


# Authorized reset method
reset_authorized() {
    echo "Executing authorized reset..."
    
    for dev in /sys/bus/usb/devices/*/; do
        if [ -f "$dev/idVendor" ] && [[ "$(cat $dev/idVendor 2>/dev/null)" =~ ^(1a6e|18d1)$ ]]; then
            echo "  Disabling device authorization..."
            echo 0 > "$dev/authorized"
            sleep 2
            echo "  Re-enabling device authorization..."
            echo 1 > "$dev/authorized"
            sleep 3
            return 0
        fi
    done
    return 1
}

# Trigger udev rescan
trigger_udev_rescan() {
    echo "  Triggering udev rules rescan..."
    sudo udevadm trigger --subsystem-match=usb >/dev/null 2>&1
    sudo udevadm settle >/dev/null 2>&1
    sleep 2
}

# Unbind/bind reset method
reset_unbind_bind() {
    echo "Executing unbind/bind reset..."
    
    for dev in /sys/bus/usb/devices/*/; do
        if [ -f "$dev/idVendor" ] && [[ "$(cat $dev/idVendor 2>/dev/null)" =~ ^(1a6e|18d1)$ ]]; then
            DEVICE_NAME=$(basename "$dev")
            echo "  Unbinding USB driver..."
            echo "$DEVICE_NAME" | sudo tee /sys/bus/usb/drivers/usb/unbind >/dev/null 2>&1
            sleep 2
            echo "  Rebinding USB driver..."
            echo "$DEVICE_NAME" | sudo tee /sys/bus/usb/drivers/usb/bind >/dev/null 2>&1
            sleep 3
            
            # Trigger udev rescan after driver rebinding
            trigger_udev_rescan
            return 0
        fi
    done
    return 1
}

# Usbreset reset method
reset_usbreset() {
    echo "Executing usbreset hardware reset..."EHCI controller
    
    if ! command -v usbreset >/dev/null 2>&1; then
        echo "✗ usbreset command not found"
        return 1
    fi
    
    local bus_device=$(lsusb | grep -E "(1a6e:|18d1:)" | head -1)
    if [ -n "$bus_device" ]; then
        local bus_num=$(echo "$bus_device" | awk '{print $2}')
        local dev_num=$(echo "$bus_device" | awk '{print $4}' | sed 's/://')
        local device_short="$bus_num/$dev_num"
        
        echo "  Resetting device: $device_short"
        sudo usbreset "$device_short" >/dev/null 2>&1
        sleep 5
        
        # Trigger udev rescan after hardware reset (most important)
        trigger_udev_rescan
        return 0
    fi
    return 1
}

iterative_reset(){
    reset_attempts=("reset_authorized" "reset_unbind_bind" "reset_usbreset")
    attempt_descriptions=("first attempt (reset_authorized)" "second attempt (reset_unbind_bind)" "third attempt (reset_usbreset)")

    # 循环尝试不同的重置方法
    for i in "${!reset_attempts[@]}"; do
        method="${reset_attempts[$i]}"
        description="${attempt_descriptions[$i]}"

        echo "Executing $description..."
        "$method" # 调用对应的重置函数

        check_device_status
        status=$? # 更新状态

        if [ $status -eq 0 ]; then
            echo "Coral TPU successfully returned to normal after $description."
            return 0 # 成功，退出循环
        fi
    done
    echo "Coral TPU still not back to normal."
    echo "Manual intervention might be required."
    return 1 # 如果所有尝试都失败了，返回1
}

# Main program logic
main() {
    echo
    echo "Checking current device status..."
    
    # First check device status
    check_device_status
    status=$?
    
    if [ $status -eq 0 ]; then
        echo "Coral TPU is already working normally, no operations needed"
        exit 0
    elif [ $status -eq 4 ]; then
        echo "No Coral TPU device detected"
        echo
        echo "Suggestions:"
        echo "1. Check USB connection"
        echo "2. Try different USB ports"
        echo "3. Check device power"
        exit 1
    elif [ $status -eq 1 ]; then
        echo "Coral TPU is in Google state (18d1), but not on EHCI controller, need reset"
        iterative_reset && exit 0 || exit 1
    elif [ $status -eq 3 ]; then
        echo "Coral TPU is in initial state (1a6e), but not on EHCI controller, need reset"
        iterative_reset && exit 0 || exit 1
    elif [ $status -eq 2 ]; then
        echo "Coral TPU is in initial state (1a6e), and on EHCI controller. good! proceeding to load delegate"
        if test_load_delegate; then
            echo "✓ Coral TPU initial loading delegate successfully, working normally"
            exit 0
        else
            echo "✗ Failed to load Edge TPU delegate, manual intervention required"
            exit 1
        fi
    fi
    # All methods failed
    echo
    echo "========================================"
    echo "✗ All reset methods failed"
    echo "========================================"
    echo
    echo "Manual operation suggestions:"
    echo "1. Physically unplug and replug the USB device"
    echo "2. Try connecting to a USB 2.0 port (EHCI controller)"
    echo "3. Check if USB power supply is sufficient"
    echo "4. Restart the system"
    
    exit 1
}

# just for testing, comment out in production
# main(){
#     reset_authorized
#     reset_unbind_bind
#     reset_usbreset
#     show_controller_info
# }

# Execute main program
main "$@"