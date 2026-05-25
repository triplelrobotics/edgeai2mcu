import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

CLASS_NAMES = ["LEFT", "RIGHT", "STRAIGHT"]
IMAGE_SIZE = (96, 96)
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "trained_line_models" / "mobilenetv2_96_a035_extpre"
LATEST_RUN_PATH = BASE_DIR / "trained_line_models" / "latest_run.txt"
LATEST_FLOAT_RUN_PATH = BASE_DIR / "trained_line_models" / "latest_float_run.txt"


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


def build_model(dropout: float, dense_units: int, alpha: float, train_backbone: bool):
    """Build MobileNetV2 transfer-learning model matching the Edge Impulse baseline."""

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = train_backbone

    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="line_mobilenetv2_96_a035")
    return model, backbone


def make_augmentation():
    """Create train-time-only image augmentation; no horizontal flip because LEFT/RIGHT labels would swap."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def preprocess_batch(images, labels):
    """Convert RGB 0..255 images to MobileNetV2's expected -1..1 float input."""

    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels


def set_finetune_layers(backbone, train_last_layers: int):
    """Unfreeze the last train_last_layers non-BatchNorm backbone layers for fine-tuning."""

    backbone.trainable = True
    trainable_seen = 0
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            continue
        if trainable_seen < train_last_layers:
            layer.trainable = True
            trainable_seen += 1
        else:
            layer.trainable = False


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


def default_run_dir():
    """Create a timestamped output directory for one training run."""

    return RUNS_DIR / time.strftime("run_%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/line_training_ready"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-epochs", type=int, default=25)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--finetune-layers", type=int, default=35)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--dense-units", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = default_run_dir()

    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_PATH.write_text(str(args.output_dir.resolve()), encoding="utf-8")
    LATEST_FLOAT_RUN_PATH.write_text(str(args.output_dir.resolve()), encoding="utf-8")
    print(f"output_dir: {args.output_dir}")

    train_ds = make_dataset(args.data_dir, "train", args.batch_size, True, args.seed)
    val_ds = make_dataset(args.data_dir, "val", args.batch_size, False, args.seed)
    test_ds = make_dataset(args.data_dir, "test", args.batch_size, False, args.seed)

    augmentation = make_augmentation()
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    weight_ds = make_dataset(args.data_dir, "train", args.batch_size, False, args.seed)
    class_weight = class_weights_from_dataset(weight_ds)
    print("class_weight:", class_weight)

    model, backbone = build_model(args.dropout, args.dense_units, args.alpha, train_backbone=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_float32.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_log.csv", append=False),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    set_finetune_layers(backbone, args.finetune_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs + args.finetune_epochs,
        initial_epoch=args.head_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.save(args.output_dir / "final_float32.keras")
    write_metrics(args.output_dir / "metrics_float32.json", model, {"val": val_ds, "test": test_ds})
    with open(args.output_dir / "labels.txt", "w", encoding="utf-8") as f:
        for idx, label in enumerate(CLASS_NAMES):
            f.write(f"{idx} {label}\n")


if __name__ == "__main__":
    main()
