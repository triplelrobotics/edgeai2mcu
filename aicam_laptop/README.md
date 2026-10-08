# AICam laptop workflow

`aicam_laptop/` contains the workstation side of the line-follow workflow: data
collection, dataset preparation, model training, TFLite export, and Coral Edge
TPU compilation. Edge Impulse is not part of the active workflow.

## Layout

```text
aicam_laptop/
├── datasets/                # Source material and versioned datasets (DVC)
├── line_follow/             # Frequently used collection and model scripts
│   └── scripts/             # Occasional dataset reconstruction scripts
├── trained_models/          # Generated training outputs (DVC, Git-ignored)
│   └── line_follow/         # Line-follow training runs and latest-run pointers
├── tools/edge_tpu_compiler/ # Platform-specific Coral compiler wrapper
└── var/archive/             # Ignored retired scripts and local history
```

Dataset lineage is documented in [DATASETS.md](DATASETS.md). The relationship
between training runs and deployed models is recorded in
[MODEL_LINEAGE.json](MODEL_LINEAGE.json). See the [H618 workflow](../aicam_h618/README.md)
and [ESP32 workflow](../aicam_esp32/README.md) for device deployment.

Run the commands below from `aicam_laptop/`.

## Restore the Python environment

The checked-in Conda environment targets Apple Silicon Macs and preserves the
TensorFlow/Keras 2.10 model format used by the existing training runs:

```bash
conda env create -f environment.yml
conda activate edgeai2mcu-laptop
python -m pip check
```

The environment was verified with Python 3.10.22, TensorFlow 2.10.0, Keras
2.10.0, and TensorFlow Model Optimization 0.7.5. It is the Apple Silicon
equivalent of the historical Windows environment, which used Python 3.10,
TensorFlow CPU 2.10.1, and Keras 2.10.0. DVC remains installed separately with
`pipx`; it is not a training dependency.

This file pins direct dependencies; pip installs transitive dependencies
automatically. It is not a complete package lock. To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate edgeai2mcu-laptop
```

## Restore artifacts

`datasets/` and `trained_models/` are tracked by DVC. After a
fresh clone and environment setup, restore them with:

```bash
dvc pull
dvc status
```

This requires the repository's DVC remote to be configured and accessible.
Git stores the small `.dvc` metadata files; the artifact contents live in the
DVC remote and local cache.

## Collect data

Use `record_base_samples.py` for continuous synchronized frames and
control labels:

Set `ESP32_BASE_URL`, `STREAM_URL`, and `DATASET_DIR` in the script before
collection. The default output is `datasets/` relative to the working directory.
On macOS, keyboard capture may require Accessibility permission. The recorder
currently sends `/cmd?act=...`, while the ESP32 handler reads `/cmd?m=...`;
align that interface before collecting control labels.

```bash
python3 line_follow/record_base_samples.py
```

Use `record_hard_samples.py` to save selected field-test frames directly into
`LEFT`, `RIGHT`, and `STRAIGHT` class directories:

```bash
python3 line_follow/record_hard_samples.py \
  --output-dir datasets/source_material/hard_examples_YYYYMMDD
```

Neither recorder is required when retraining from an existing dataset.

## Rebuild historical datasets

The scripts below reproduce specific dataset versions. They are provenance and
maintenance tools, not steps that must run before every training session:

```bash
python3 line_follow/scripts/build_v2_base_plus_hard_right.py

python3 line_follow/scripts/build_v3_dedup.py \
  --output datasets/line_follow_v3_rebuilt
```

## Train and export

The default dataset is
`datasets/line_follow_v3_base_plus_hard_right_dedup_20261003/`.

```bash
python3 line_follow/train_float32.py
python3 line_follow/train_qat.py
python3 line_follow/export_tflite.py
```

To reproduce the currently deployed model lineage, explicitly select v2:

```bash
python3 line_follow/train_float32.py \
  --data-dir datasets/line_follow_v2_base_plus_hard_right_20260607
```

Training outputs are written under `trained_models/line_follow/`. Latest-run
pointers use relative paths so they remain valid when the repository is moved or
cloned on another computer.

Both training scripts update latest-run pointers even with a custom
`--output-dir`. For isolated smoke tests, run script copies and a small dataset
outside the checkout so the pointers also stay in the test directory.

Always evaluate the exported uint8 model before deployment. Quantization can
change class behavior even when the float32 model performs well.

## Compile for Coral

The exported `model_int8_uint8.tflite` is a standard TFLite model. Coral requires
an additional Edge TPU compilation step:

```bash
tools/edge_tpu_compiler/compile.sh \
  trained_models/line_follow/mobilenetv2_96_a035_extpre_qat/run_YYYYMMDD_HHMMSS/model_int8_uint8.tflite
```

The wrapper builds and runs the Linux compiler container automatically. The
compiled `_edgetpu.tflite` model and compiler log are written beside the input
model. On Apple Silicon it uses `linux/amd64` by default. The image and platform
can be overridden with `AICAM_EDGETPU_IMAGE` and `AICAM_DOCKER_PLATFORM`.

Docker must be installed and running separately from Conda. The wrapper accepts
any `.tflite` path; Edge TPU mapping depends on the model's operators. After
evaluation, copy the selected compiled model to
`aicam_h618/app/line_follow/tflite_models/`, update its selected path and lineage,
and follow the H618 deployment procedure.

## Validation baseline

The Apple Silicon environment passed loading/inference with existing latest
float and QAT models. An isolated 36-image smoke test completed float head
training and fine-tuning (one epoch each), QAT (one epoch), and float32/uint8
TFLite export and test-set inference. It used `weights=None` in a temporary
script copy because ImageNet downloading was unavailable. This validates the
training/export path, not model quality, ImageNet downloading, hardware sample
collection, or Docker/Edge TPU compilation.

## Storage policy

- Git tracks source code, documentation, configuration examples, and selected
  deployment models.
- DVC tracks `datasets/` and `trained_models/`.
- `var/` is ignored local storage and is not required to restore the active
  workflow.
- The immutable Lenovo snapshot remains the recovery source for material moved
  into `var/archive/`.
