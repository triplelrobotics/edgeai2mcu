# Local line-follow training

This folder trains and exports line-follow models without Edge Impulse.

The starting architecture mirrors the Edge Impulse transfer-learning setup:

- MobileNetV2
- input size `96x96`
- width multiplier `alpha=0.35`
- pretrained ImageNet backbone
- classifier head: `Dense(16) -> Dropout(0.1) -> Dense(3)`
- class order: `LEFT`, `RIGHT`, `STRAIGHT`

Prepare the dataset first:

```bash
cd aicam_laptop
python prepare_impulse_dataset.py
```

Train the float32 model:

```bash
cd aicam_laptop
python line_training/train_float32.py
```

Export float32 and int8 TFLite models:

```bash
cd aicam_laptop
python line_training/export_tflite.py
```

If PTQ int8 hurts `STRAIGHT`, run quantization-aware training:

```bash
cd aicam_laptop
python line_training/train_qat.py
python line_training/export_tflite.py
```

The default QAT mode is conservative: it freezes most of the model, trains only the classifier head, uses a very small learning rate, and does not apply class weights. If that preserves accuracy but does not improve int8 enough, try:

```bash
python line_training/train_qat.py --trainable last-n --last-n 30 --learning-rate 5e-7 --epochs 20
```

Outputs go to:

```text
aicam_laptop/line_training/trained_line_models/mobilenetv2_96_a035_extpre/run_YYYYMMDD_HHMMSS/
```

The `_extpre` suffix means MobileNetV2 preprocessing is external to the model graph. The Keras/TFLite float32 model input is already preprocessed to `[-1, 1]`. This usually gives cleaner uint8 PTQ input parameters, close to `scale=1/128` and `zero_point=128`, and matches the H618 inference server's `(pixel - 128) / 128` input handling.

`train_float32.py` writes the newest run path to `trained_line_models/latest_run.txt`. `export_tflite.py` uses that latest run by default. To export an older run, pass `--model-dir`.

Important: evaluate the int8 TFLite model, not just the float32 model. The current field issue is that the float32 model can classify obvious `STRAIGHT` frames correctly while the quantized int8/EdgeTPU model biases them toward `LEFT`.
