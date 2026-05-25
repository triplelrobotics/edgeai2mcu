import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

try:
    import tensorflow_model_optimization as tfmot
except ImportError as e:
    raise SystemExit(
        "tensorflow_model_optimization is required for QAT. Install it with:\n"
        "python -m pip install tensorflow-model-optimization==0.7.5"
    ) from e

tf.get_logger().setLevel("ERROR")

CLASS_NAMES = ["LEFT", "RIGHT", "STRAIGHT"]
IMAGE_SIZE = (96, 96)
BASE_DIR = Path(__file__).resolve().parent
LATEST_RUN_PATH = BASE_DIR / "trained_line_models" / "latest_run.txt"
LATEST_FLOAT_RUN_PATH = BASE_DIR / "trained_line_models" / "latest_float_run.txt"


def latest_or_legacy_float_dir():
    """Find the newest float32 model directory to use as the QAT starting point."""

    if LATEST_FLOAT_RUN_PATH.exists():
        latest = Path(LATEST_FLOAT_RUN_PATH.read_text(encoding="utf-8").strip())
        if (latest / "best_float32.keras").exists():
            return latest
    legacy = BASE_DIR / "trained_line_models" / "mobilenetv2_96_a035_extpre"
    if (legacy / "best_float32.keras").exists():
        return legacy
    raise FileNotFoundError("Cannot find best_float32.keras. Run train_float32.py first.")


def default_qat_dir():
    """Create a timestamped QAT output directory."""

    return BASE_DIR / "trained_line_models" / "mobilenetv2_96_a035_extpre_qat" / time.strftime("run_%Y%m%d_%H%M%S")


def make_dataset(data_dir: Path, split: str, batch_size: int, shuffle: bool, seed: int):
    """Load one class-folder split as RGB float32 batches with fixed class order."""

    return tf.keras.utils.image_dataset_from_directory(
        data_dir / split,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed,
    )


def preprocess_batch(images, labels):
    """Convert RGB 0..255 images to MobileNetV2's expected -1..1 float input."""

    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels


def class_weights_from_dataset(dataset):
    """Compute inverse-frequency class weights from one categorical dataset."""

    counts = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for _, y in dataset.unbatch():
        counts[int(tf.argmax(y).numpy())] += 1
    total = counts.sum()
    weights = total / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    return {idx: float(weight) for idx, weight in enumerate(weights)}


def confusion_matrix(model, dataset):
    """Return a small confusion matrix for one dataset."""

    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int32)
    for x, y in dataset:
        pred = model.predict(x, verbose=0)
        true_ids = np.argmax(y.numpy(), axis=1)
        pred_ids = np.argmax(pred, axis=1)
        for true_id, pred_id in zip(true_ids, pred_ids):
            matrix[true_id, pred_id] += 1
    return matrix


def write_metrics(path: Path, model, datasets):
    """Write final val/test accuracy and confusion matrices to JSON."""

    metrics = {"class_names": CLASS_NAMES}
    for split, dataset in datasets.items():
        loss, acc = model.evaluate(dataset, verbose=0)
        metrics[split] = {
            "loss": float(loss),
            "accuracy": float(acc),
            "confusion_matrix": confusion_matrix(model, dataset).tolist(),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def make_qat_model(float_model_path: Path):
    """Load a float32 Keras model, flatten its MobileNetV2 backbone, then wrap it for QAT."""

    float_model = tf.keras.models.load_model(float_model_path)
    flat_model = make_flat_model_from_nested(float_model)
    return tfmot.quantization.keras.quantize_model(flat_model)


def find_nested_backbone(float_model):
    """Find the nested MobileNetV2 model inside the transfer-learning model."""

    for layer in float_model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("Could not find nested MobileNetV2 backbone in float model.")


def get_layer_weights(model, layer_name: str):
    """Return weights from a named layer in the source model."""

    return model.get_layer(layer_name).get_weights()


def make_flat_model_from_nested(float_model):
    """Rebuild the model without nesting MobileNetV2 so tfmot can quantize it."""

    nested_backbone = find_nested_backbone(float_model)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        alpha=0.35,
        include_top=False,
        weights=None,
    )
    backbone.set_weights(nested_backbone.get_weights())

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(0.1, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="predictions")(x)

    flat_model = tf.keras.Model(backbone.input, outputs, name="line_mobilenetv2_96_a035_flat")
    flat_model.get_layer("classifier_dense").set_weights(get_layer_weights(float_model, "classifier_dense"))
    flat_model.get_layer("predictions").set_weights(get_layer_weights(float_model, "predictions"))
    return flat_model


def configure_trainable_layers(model, mode: str, last_n: int):
    """Choose how much of the QAT model is trainable."""

    if mode == "all":
        for layer in model.layers:
            layer.trainable = True
        return

    for layer in model.layers:
        layer.trainable = False

    if mode == "head":
        trainable_names = ("classifier_dense", "classifier_dropout", "predictions")
        for layer in model.layers:
            if any(name in layer.name for name in trainable_names):
                layer.trainable = True
        return

    if mode == "last-n":
        trainable_seen = 0
        for layer in reversed(model.layers):
            if trainable_seen < last_n:
                layer.trainable = True
                trainable_seen += 1
        return

    raise ValueError(f"Unknown trainable mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/line_training_ready"))
    parser.add_argument("--float-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--trainable", choices=["head", "last-n", "all"], default="head")
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--use-class-weight", action="store_true")
    args = parser.parse_args()

    if args.float_model_dir is None:
        args.float_model_dir = latest_or_legacy_float_dir()
    if args.output_dir is None:
        args.output_dir = default_qat_dir()

    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_PATH.write_text(str(args.output_dir.resolve()), encoding="utf-8")

    print(f"float_model_dir: {args.float_model_dir}")
    print(f"output_dir: {args.output_dir}")

    train_ds = make_dataset(args.data_dir, "train", args.batch_size, True, args.seed)
    val_ds = make_dataset(args.data_dir, "val", args.batch_size, False, args.seed)
    test_ds = make_dataset(args.data_dir, "test", args.batch_size, False, args.seed)

    train_ds = train_ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    weight_ds = make_dataset(args.data_dir, "train", args.batch_size, False, args.seed)
    class_weight = class_weights_from_dataset(weight_ds)
    print("class_weight:", class_weight)

    qat_model = make_qat_model(args.float_model_dir / "best_float32.keras")
    configure_trainable_layers(qat_model, args.trainable, args.last_n)
    print(f"qat trainable mode: {args.trainable}")
    print(f"trainable layers: {sum(int(layer.trainable) for layer in qat_model.layers)} / {len(qat_model.layers)}")
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_qat",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "qat_training_log.csv", append=False),
    ]

    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight if args.use_class_weight else None,
    )

    qat_model.save(args.output_dir / "final_qat")
    write_metrics(args.output_dir / "metrics_qat_floatgraph.json", qat_model, {"val": val_ds, "test": test_ds})
    with open(args.output_dir / "labels.txt", "w", encoding="utf-8") as f:
        for idx, label in enumerate(CLASS_NAMES):
            f.write(f"{idx} {label}\n")


if __name__ == "__main__":
    main()
