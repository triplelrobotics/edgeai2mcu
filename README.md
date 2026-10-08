# EdgeAI2MCU

Camera-based robot experiments with an H618 board and Coral USB Edge TPU for
vision, an ESP32-S3 running CircuitPython for motor control, and a laptop for
data collection and model training. The repository includes line-following
workflows and earlier camera, classification, and detection demos.

## Module documentation

| Module | Role | Documentation |
| --- | --- | --- |
| `aicam_h618/` | Camera capture, Edge TPU inference, preview, and robot decisions | [H618 setup and deployment](aicam_h618/README.md) |
| `aicam_esp32/` | Wi-Fi HTTP interface and motor control | [ESP32 firmware and deployment](aicam_esp32/README.md) |
| `aicam_laptop/` | Sample collection, datasets, training, export, and compilation | [Laptop development workflow](aicam_laptop/README.md) |

```text
H618 camera -> Coral inference -> H618 lane decision -> ESP32 motor control

Laptop samples -> dataset -> float training -> QAT -> TFLite -> Edge TPU model
```

## Restore a development checkout

Install Git LFS and DVC with Google Drive support. From the cloned repository root:

```bash
git lfs install
git lfs pull
dvc pull
```

Git LFS restores selected deployment models. DVC restores laptop datasets and
training outputs from the configured Google Drive remote. Access to that remote
and local authentication are required; credentials are not stored in Git.

Then follow the [laptop environment instructions](aicam_laptop/README.md#restore-the-python-environment),
the [H618 deployment instructions](aicam_h618/README.md#deployment-from-the-mac),
and the [ESP32 restoration instructions](aicam_esp32/README.md#restore-and-deploy).
The laptop Conda environment targets Apple Silicon; device firmware, the H618
operating system, and Docker are installed separately.

## Development and storage

Make changes in the local checkout, test them on the devices, and commit the
working changes. H618 synchronization maps local `aicam_h618/app/` to device
`/workspace/aicam_coral/app/`. ESP32 runtime files map to the CircuitPython root.

| Storage | Contents |
| --- | --- |
| Git | Code, documentation, configuration examples, and environment metadata |
| Git LFS | Selected H618 deployment models |
| DVC | `aicam_laptop/datasets/` and `aicam_laptop/trained_models/` |
| Ignored local/device files | Credentials, generated files, and `var/` data |

Keep device-generated captures separate from deployment files. Transfer useful
training material into a versioned laptop dataset. Ignored archives are not
restored by cloning.

The [dataset history](aicam_laptop/DATASETS.md) and
[model lineage](aicam_laptop/MODEL_LINEAGE.json) identify deployment training
inputs and models.

## Current verification

The Apple Silicon environment has passed existing float/QAT model loading and
a small training, QAT, and TFLite export smoke test. That test used random
backbone initialization; it did not validate model quality, camera collection,
motor behavior, or Edge TPU compilation. Hardware changes need device testing.
