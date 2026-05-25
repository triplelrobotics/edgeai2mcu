import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    tfmot = None

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel("ERROR")

CLASS_NAMES = ["LEFT", "RIGHT", "STRAIGHT"]
IMAGE_SIZE = (96, 96)
BASE_DIR = Path(__file__).resolve().parent
LATEST_RUN_PATH = BASE_DIR / "trained_line_models" / "latest_run.txt"


def default_model_dir():
    """Return the latest timestamped training run, falling back to the legacy output directory."""

    if LATEST_RUN_PATH.exists():
        latest = Path(LATEST_RUN_PATH.read_text(encoding="utf-8").strip())
        if latest.exists():
            return latest
    return BASE_DIR / "trained_line_models" / "mobilenetv2_96_a035_extpre"


def default_keras_model_name(model_dir: Path):
    """Prefer QAT SavedModel checkpoints, then QAT .keras, then float32 .keras."""

    for name in ("best_qat", "best_qat.keras", "best_float32.keras"):
        if (model_dir / name).exists():
            return name
    return "best_float32.keras"


def load_keras_model(model_path: Path):
    """Load either a plain Keras model or a QAT model with quantize wrappers."""

    if tfmot is None:
        return tf.keras.models.load_model(model_path)
    with tfmot.quantization.keras.quantize_scope():
        return tf.keras.models.load_model(model_path)


def representative_dataset(data_dir: Path, count: int, seed: int):
    """Yield float32 training samples for full-integer quantization calibration."""

    image_paths = []
    for label in CLASS_NAMES:
        image_paths.extend(sorted((data_dir / "train" / label).glob("*")))
    rng = np.random.default_rng(seed)
    rng.shuffle(image_paths)

    for idx, image_path in enumerate(image_paths):
        if idx >= count:
            break
        image = preprocess_image(load_rgb_image(image_path))[None, ...]
        yield [tf.cast(image, tf.float32)]


def convert_float32(model, out_path: Path):
    """Export a Keras model to float32 TFLite."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def convert_int8(model, out_path: Path, data_dir: Path, rep_count: int, seed: int):
    """Export a full-integer uint8 TFLite model suitable for EdgeTPU compilation."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(data_dir, rep_count, seed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def load_rgb_image(image_path: Path):
    """Load one image with TensorFlow ops so export/eval does not require Pillow."""

    data = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="area")
    return image.numpy().astype(np.float32)


def preprocess_image(image: np.ndarray):
    """Apply the same external MobileNetV2 preprocessing used during training."""

    return tf.keras.applications.mobilenet_v2.preprocess_input(image.copy())


def preprocess_for_tflite(image_path: Path, input_details):
    """Load one image and encode it for the TFLite model input tensor."""

    arr = preprocess_image(load_rgb_image(image_path))
    detail = input_details[0]
    dtype = detail["dtype"]
    if dtype == np.float32:
        return arr[None, ...].astype(np.float32)

    scale, zero_point = detail["quantization"]
    quantized = arr / scale + zero_point
    quantized = np.clip(np.round(quantized), np.iinfo(dtype).min, np.iinfo(dtype).max)
    return quantized.astype(dtype)[None, ...]


def read_output(interpreter):
    """Read and dequantize one TFLite output tensor."""

    detail = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(detail["index"])[0]
    scale, zero_point = detail["quantization"]
    if scale:
        return (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def evaluate_tflite(model_path: Path, data_dir: Path):
    """Evaluate one TFLite model on class-folder test data."""

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int32)
    total = 0
    correct = 0
    for true_id, label in enumerate(CLASS_NAMES):
        for image_path in sorted((data_dir / "test" / label).glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            x = preprocess_for_tflite(image_path, input_details)
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()
            scores = read_output(interpreter)
            pred_id = int(np.argmax(scores))
            matrix[true_id, pred_id] += 1
            total += 1
            correct += int(pred_id == true_id)

    return {
        "model": str(model_path),
        "input": {
            "shape": input_details[0]["shape"].tolist(),
            "dtype": str(input_details[0]["dtype"]),
            "quantization": input_details[0]["quantization"],
        },
        "output": {
            "shape": output_details[0]["shape"].tolist(),
            "dtype": str(output_details[0]["dtype"]),
            "quantization": output_details[0]["quantization"],
        },
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/line_training_ready"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--keras-model", default=None)
    parser.add_argument("--rep-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = default_model_dir()
    print(f"model_dir: {args.model_dir}")

    if args.keras_model is None:
        args.keras_model = default_keras_model_name(args.model_dir)

    model_path = args.model_dir / args.keras_model
    model = load_keras_model(model_path)

    float_path = args.model_dir / "model_float32.tflite"
    int8_path = args.model_dir / "model_int8_uint8.tflite"

    convert_float32(model, float_path)
    convert_int8(model, int8_path, args.data_dir, args.rep_count, args.seed)

    metrics = {
        "class_names": CLASS_NAMES,
        "float32": evaluate_tflite(float_path, args.data_dir),
        "int8_uint8": evaluate_tflite(int8_path, args.data_dir),
    }
    with open(args.model_dir / "metrics_tflite.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"wrote {float_path}")
    print(f"wrote {int8_path}")
    print(f"wrote {args.model_dir / 'metrics_tflite.json'}")


if __name__ == "__main__":
    main()
