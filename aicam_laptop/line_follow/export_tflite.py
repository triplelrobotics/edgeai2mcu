import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from prepare_dataset import check_model_classes, read_classes, validate_split_classes, write_json

try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    tfmot = None

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel("ERROR")

CLASS_NAMES = ["LEFT", "RIGHT", "STRAIGHT"]
IMAGE_SIZE = (96, 96)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "trained_models" / "line_follow"
LATEST_RUN_PATH = MODELS_DIR / "latest_run.txt"


@contextmanager
def export_stage(name):
    started = time.perf_counter()
    print(f"{name}...", flush=True)
    yield
    print(f"{name} done ({time.perf_counter() - started:.1f}s)", flush=True)


def read_run_pointer(pointer_path: Path) -> Path:
    """Resolve a run pointer relative to the file that stores it."""

    value = Path(pointer_path.read_text(encoding="utf-8").strip())
    return value if value.is_absolute() else pointer_path.parent / value


def default_model_dir():
    """Return the latest timestamped training run, falling back to the legacy output directory."""

    if LATEST_RUN_PATH.exists():
        latest = read_run_pointer(LATEST_RUN_PATH)
        if latest.exists():
            return latest
    return MODELS_DIR / "mobilenetv2_96_a035_extpre"


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

    total = min(count, len(image_paths))
    print(f"[Calibration] Using {total} samples", flush=True)
    for idx, image_path in enumerate(image_paths):
        if idx >= count:
            break
        image = preprocess_image(load_rgb_image(image_path))[None, ...]
        yield [tf.cast(image, tf.float32)]
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"[Calibration] {idx + 1}/{total}", flush=True)


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
    test_count = sum(1 for label in CLASS_NAMES for path in (data_dir / "test" / label).glob("*")
                     if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
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
            if total % 25 == 0 or total == test_count:
                print(f"[Evaluation] {model_path.name}: {total}/{test_count}", flush=True)

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
    global CLASS_NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/line_follow_v3_base_plus_hard_right_dedup_20261003"),
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--keras-model", default=None)
    parser.add_argument("--rep-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", metavar="CLASS",
                        help="Training classes in output order, e.g. RIGHT STRAIGHT; default: all dataset classes")
    args = parser.parse_args()
    CLASS_NAMES = read_classes(args.data_dir, args.classes)
    print("Selected classes:", CLASS_NAMES, flush=True)
    for split in ("train", "test"):
        validate_split_classes(args.data_dir, split, CLASS_NAMES)

    if args.model_dir is None:
        args.model_dir = default_model_dir()
    check_model_classes(args.model_dir, CLASS_NAMES)
    print(f"model_dir: {args.model_dir}", flush=True)

    if args.keras_model is None:
        args.keras_model = default_keras_model_name(args.model_dir)

    model_path = args.model_dir / args.keras_model
    with export_stage(f"[1/6] Loading {model_path.name}"):
        model = load_keras_model(model_path)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model output does not match dataset classes")
    write_json(args.model_dir / "labels.json", CLASS_NAMES)

    float_path = args.model_dir / "model_float32.tflite"
    int8_path = args.model_dir / "model_int8_uint8.tflite"

    print("TensorFlow conversion may run silently for several minutes within a stage.", flush=True)
    with export_stage("[2/6] Converting float32 TFLite"):
        convert_float32(model, float_path)
    with export_stage("[3/6] Converting uint8 TFLite"):
        convert_int8(model, int8_path, args.data_dir, args.rep_count, args.seed)
    with export_stage("[4/6] Evaluating float32 TFLite"):
        float_metrics = evaluate_tflite(float_path, args.data_dir)
    with export_stage("[5/6] Evaluating uint8 TFLite"):
        int8_metrics = evaluate_tflite(int8_path, args.data_dir)

    metrics = {"class_names": CLASS_NAMES, "float32": float_metrics, "int8_uint8": int8_metrics}
    with export_stage("[6/6] Saving metrics"):
        with open(args.model_dir / "metrics_tflite.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(f"Test accuracy: float32={float_metrics['accuracy']:.2%}, uint8={int8_metrics['accuracy']:.2%}", flush=True)
    print(f"wrote {float_path}", flush=True)
    print(f"wrote {int8_path}", flush=True)
    print(f"wrote {args.model_dir / 'metrics_tflite.json'}", flush=True)


if __name__ == "__main__":
    main()
