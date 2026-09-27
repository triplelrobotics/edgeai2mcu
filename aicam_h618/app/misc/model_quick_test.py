from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import load_edgetpu_delegate

DEVICE = 'usb:0'
PRELOADED_DELEGATE = load_edgetpu_delegate(options={'device': DEVICE})
print("preloaded delegate for device", DEVICE)

interpreter = make_interpreter("tflite_learn_992353_5_edgetpu.tflite", device=DEVICE, delegate=PRELOADED_DELEGATE)
interpreter.allocate_tensors()

print("TPU model loaded")