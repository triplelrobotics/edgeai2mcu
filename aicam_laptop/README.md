# AICam laptop workflow

`aicam_laptop/` contains the workstation side of the line-follow workflow: data
collection, dataset preparation, model training, TFLite export, and Coral Edge
TPU compilation. Edge Impulse is not part of the active workflow.

## Layout

```text
aicam_laptop/
├── dataset/                 # Source material and versioned datasets (DVC)
├── line_follow/             # Frequently used collection and model scripts
│   ├── scripts/             # Occasional dataset reconstruction scripts
│   └── trained_line_models/ # Generated training outputs (DVC, Git-ignored)
├── tools/edge_tpu_compiler/ # Platform-specific Coral compiler wrapper
└── var/archive/             # Ignored retired scripts and local history
```

Dataset lineage is documented in `DATASETS.md`. The relationship between
training runs and deployed models is recorded in `MODEL_LINEAGE.json`.

Run the commands below from `aicam_laptop/`.

## Collect data

Use `record_base_samples.py` for continuous synchronized frames and
control labels:

```bash
python3 line_follow/record_base_samples.py
```

Use `record_hard_samples.py` to save selected field-test frames directly into
`LEFT`, `RIGHT`, and `STRAIGHT` class directories:

```bash
python3 line_follow/record_hard_samples.py \
  --output-dir dataset/source_material/hard_examples_YYYYMMDD
```

Neither recorder is required when retraining from an existing dataset.

## Rebuild historical datasets

The scripts below reproduce specific dataset versions. They are provenance and
maintenance tools, not steps that must run before every training session:

```bash
python3 line_follow/scripts/build_v2_base_plus_hard_right.py

python3 line_follow/scripts/build_v3_dedup.py \
  --output dataset/line_follow_v3_rebuilt
```

## Train and export

The default dataset is
`dataset/line_follow_v3_base_plus_hard_right_dedup_20261003/`.

```bash
python3 line_follow/train_float32.py
python3 line_follow/train_qat.py
python3 line_follow/export_tflite.py
```

To reproduce the currently deployed model lineage, explicitly select v2:

```bash
python3 line_follow/train_float32.py \
  --data-dir dataset/line_follow_v2_base_plus_hard_right_20260607
```

Training outputs are written under `line_follow/trained_line_models/`. Latest-run
pointers use relative paths so they remain valid when the repository is moved or
cloned on another computer.

Always evaluate the exported uint8 model before deployment. Quantization can
change class behavior even when the float32 model performs well.

## Compile for Coral

The exported `model_int8_uint8.tflite` is a standard TFLite model. Coral requires
an additional Edge TPU compilation step:

```bash
tools/edge_tpu_compiler/compile.sh \
  line_follow/trained_line_models/mobilenetv2_96_a035_extpre_qat/run_YYYYMMDD_HHMMSS/model_int8_uint8.tflite
```

The wrapper builds and runs the Linux compiler container automatically. The
compiled `_edgetpu.tflite` model and compiler log are written beside the input
model. On Apple Silicon it uses `linux/amd64` by default. The image and platform
can be overridden with `AICAM_EDGETPU_IMAGE` and `AICAM_DOCKER_PLATFORM`.

## Storage policy

- Git tracks source code, documentation, configuration examples, and selected
  deployment models.
- DVC is intended to track `dataset/` and reproducible training artifacts.
- `var/` is ignored local storage and is not required to restore the active
  workflow.
- The immutable Lenovo snapshot remains the recovery source for material moved
  into `var/archive/`.
