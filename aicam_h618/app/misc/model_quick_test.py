from pathlib import Path

from pycoral.utils.edgetpu import load_edgetpu_delegate, make_interpreter


DEVICE = "usb:0"
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "line_follow"
    / "tflite_models"
    / "model_int8_uint8_edgetpu_run_20260607_133850.tflite"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

PRELOADED_DELEGATE = load_edgetpu_delegate(options={"device": DEVICE})
print("preloaded delegate for device", DEVICE)

interpreter = make_interpreter(
    str(MODEL_PATH),
    device=DEVICE,
    delegate=PRELOADED_DELEGATE,
)
interpreter.allocate_tensors()

print("TPU model loaded:", MODEL_PATH)
