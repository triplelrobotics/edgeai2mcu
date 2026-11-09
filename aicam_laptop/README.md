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
conda activate aicam_laptop
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
conda activate aicam_laptop
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

Pass the H618 stream URL and the ESP32 base URL, including its verified port.
On macOS, keyboard capture may require Accessibility permission.

```bash
python3 line_follow/record_base_samples.py \
  --stream-url http://H618_IP:5000/video_feed \
  --motor-url http://ESP32_IP:PORT
```

Each run creates `var/capture/session_YYYYMMDD_HHMMSS_ffffff/` with
`images/`, `labels.csv`, and `session.json` (URLs, timestamps, capture settings).

- **Up:** start/resume fixed-throttle straight driving; releasing has no effect.
- **Left/Right:** hold to turn; release to drive straight along the new heading.
  Repeated taps change heading, not turn strength.
- **Down/Space:** stop; releasing keeps STOP. A new direction press or steering
  release can resume motion. Held-key repeats are ignored, as in the old web UI.
- **Quit:** press `q`. Startup and shutdown send STOP; control failures end capture
  and leave the session pending. If STOP fails, check the robot directly.
- **Review:** choose `k` to keep, `d` to delete the session, or Enter/`p` to leave
  pending. Ctrl+C, errors, or an unavailable prompt also leave it pending.
- **Storage:** kept sessions move to `datasets/source_material/records/`.
  Consecutive STOP samples are skipped after the first saved STOP frame.
  Normal frame availability and sampling rate still apply; no STOP frame is
  forced on exit. Stream outages do not duplicate old frames.
- **Options:** `--capture-dir`, `--output-dir`, `--save-fps`, `--no-preview`.
  Defaults are relative to `aicam_laptop/`; explicit relative paths use the
  working directory.

Review a pending session later, without connecting to either device:

```bash
python3 line_follow/record_base_samples.py \
  --review-session var/capture/session_YYYYMMDD_HHMMSS_ffffff
```

Add `--decision keep|discard|pending` for an explicit decision. Use the same
`--capture-dir` when reviewing captures stored in a custom location.

After review, run `dvc add datasets` and `dvc push`, then commit `datasets.dvc`.
Preparing train/val/test splits from raw sessions remains a separate step.

Use `record_hard_samples.py` to save selected field-test frames directly into
`LEFT`, `RIGHT`, and `STRAIGHT` class directories:

```bash
python3 line_follow/record_hard_samples.py \
  --output-dir datasets/source_material/hard_examples_YYYYMMDD
```

Neither recorder is required when retraining from an existing dataset.

## Review raw recordings and prepare a dataset

`prepare_dataset.py` contains the review page and three commands: `review`,
`check`, and `build`. No handwritten JSON configuration is needed.

Review one session at a time:

```bash
python line_follow/prepare_dataset.py review \
  --source var/capture/session_20261022_085655_070233
```

Open `http://127.0.0.1:8765` (or set `--port`). Ctrl+C stops the server; restarting
with the same source resumes saved decisions. Use another source to review more.

- Click selects one frame; Ctrl/Cmd-click toggles; Shift-click selects a range.
  Select page / Select all filtered support batch decisions across pages.
- Keep (`K`), Exclude (`X`), Unreviewed (`U`), label overrides and reasons save
  automatically. Arrow keys inspect adjacent frames. Undo reverses up to 30 edits
  within the current server session.
- The header shows kept STRAIGHT/RIGHT/LEFT counts for the entire current session,
  using corrected labels and excluding STOP. Counts update after edits and undo.
- Filter by label, decision or integrity issues. Reviewed shows Keep + Exclude
  together; Keep shows only retained samples. Missing/corrupt/invalid rows need
  explicit exclusion. Source hashes detect changes after review.
- Decisions live in `<session>/review.csv`, beside `labels.csv` and `session.json`.
  Raw images and labels are unchanged. Review quality and label correctness now;
  choose training classes later.

For a read-only integrity report, without creating decisions or starting a server:

```bash
python line_follow/prepare_dataset.py check --source PATH_TO_SESSION
```

Retain a pending capture in DVC-managed storage using the existing session command;
its `review.csv` moves with it:

```bash
python line_follow/record_base_samples.py \
  --review-session var/capture/session_20261022_085655_070233 --decision keep
```

After reviewing the desired sessions, pass their directories directly to `build`:

```bash
python line_follow/prepare_dataset.py build \
  --source datasets/source_material/records/session_A \
           datasets/source_material/records/session_B \
  --output datasets/line_follow_v4_reviewed
```

- Pool all kept samples, exclude STOP, and deduplicate identical image content
  before splitting. Conflicting labels for identical content stop the build.
  Only Keep samples enter the dataset. Unreviewed and Exclude samples are skipped
  and counted without changing review decisions. Images absent from the CSV are
  omitted and reported.
- Split each class approximately 70% train / 15% val / 15% test, with fixed
  `--seed 42`. A class with at least three unique samples contributes to all splits;
  one or two samples stay in train. Empty classes are reported and checked when
  training selects its classes. One sufficiently large session is enough.
- Same sources, decisions, seed and script produce identical output contents,
  regardless of source argument order. Adding sources can change the split;
  keep an existing generated version for model comparisons.
- Exact duplicates cannot cross splits. Similar neighboring frames can, so this
  pooled split can overestimate performance on independent runs. No automatic
  near-duplicate removal, class balancing or augmentation is performed.

Raw sessions stay under `datasets/source_material/records/`; generated
`train/val/test/<class>/` folders stay under the separate `--output` directory.
Existing output directories are never overwritten. The only generated JSON is
`dataset.json`: class order, source/review hashes, session metadata, seed, ratios,
counts and builder hash. `manifest.csv` traces each output image to its source,
original/final label and review reason.

DVC versions raw sessions (including reviews) and generated datasets. Run
`dvc add datasets` and `dvc push`, then commit `datasets.dvc` after preparation.
`var/capture/` alone is not backed up by DVC. No extra config/review directories
are required.

Dataset builds retain LEFT/RIGHT/STRAIGHT. Use the same dataset for two- or
three-class training; pass the same ordered `--classes` to training, QAT and export:

```bash
python line_follow/train_float32.py \
  --data-dir datasets/line_follow_v4_reviewed --classes RIGHT STRAIGHT
python line_follow/train_qat.py \
  --data-dir datasets/line_follow_v4_reviewed --classes RIGHT STRAIGHT
python line_follow/export_tflite.py \
  --data-dir datasets/line_follow_v4_reviewed --classes RIGHT STRAIGHT
```

Omit `--classes` to use all dataset classes. Unselected images do not participate
in training, validation, testing or quantization calibration. Old datasets without
class metadata retain LEFT/RIGHT/STRAIGHT order. Models record their selected
labels, and QAT/export reject mismatches. Two-class models need corresponding
inference-side label mapping before deployment; device code is unchanged here.

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

The wrapper reuses `edgetpu_compiler:16.0.384591198`, loading the DVC-managed
archive automatically if the image is missing. It does not build the Dockerfile
or download compiler packages. Restore the archive from the repository root:

```bash
dvc pull aicam_laptop/tools/edge_tpu_compiler/edgetpu-compiler-16.0.384591198-linux-amd64.tar.dvc
```

The archive was exported from the Lenovo environment and contains compiler
`16.0.384591198`, matching the May/June 2026 compilation logs. To check it after
the first compilation:

```bash
docker run --rm --platform linux/amd64 \
  edgetpu_compiler:16.0.384591198 edgetpu_compiler --version
```

The compiled `_edgetpu.tflite` model, compiler report (`_edgetpu.log`), and console
log (`_edgetpu.compile.log`) are written beside the input model.
Apple Silicon uses `linux/amd64` emulation. Override the image and platform
with `AICAM_EDGETPU_IMAGE` and `AICAM_DOCKER_PLATFORM`; custom images must already
be loaded. The old Dockerfile is retained for reference; its Coral APT endpoint
currently returns HTTP 404.

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
