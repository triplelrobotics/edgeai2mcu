# Laptop datasets

The original Lenovo Legion snapshot is preserved outside this repository at:

```text
/Users/xubo92/Documents/archives/edgeai2mcu_20260923_from_lenovo_legion/aicam_laptop
```

Do not edit that snapshot. The active dataset contains source material followed
by explicitly versioned line-follow training sets.

## Source material

`datasets/source_material/` preserves inputs that cannot be safely reconstructed:

- `recordings/record_02`, `record_03`, and `record_05` contain `images/` and
  `labels.csv`. Their generated `edgeimpulse_dataset/` copies are omitted.
- `recordings/record_04` contains the original labeled images available for that
  recording.
- `curated_final_20260518/` preserves the manually merged FINAL dataset.
- `edge_impulse_export_20260518/` preserves the Edge Impulse export, including
  one image that is not present in FINAL.
- `hard_examples_20260607/` preserves the 83 distinct field-test examples.

Source data may contain duplicate frames and historical labeling issues. It is
preserved for provenance, not used directly for evaluation.

Known source issues found by SHA-256 audit:

- `curated_final_20260518/` contains 182 duplicate copies. Of those, 178 use a
  `(2)` collision suffix and four are consecutive filenames with identical
  content.
- `record_05` contains three identical-frame label conflicts at the action
  boundaries: `002023/002024` (RIGHT/STRAIGHT), `000846/000847`
  (STRAIGHT/RIGHT), and `001611/001612` (STRAIGHT/LEFT).
- Generated classification copies from records 02, 03, and 05 are omitted
  because they can be recreated from `images/` and `labels.csv`.

## Versioned training sets

The training-set names describe their lineage directly:

- `line_follow_v1_base_20260518/` is the original prepared base dataset. It was
  used by float run `20260518_153829` and QAT run `20260518_162849`.
- `line_follow_v2_base_plus_hard_right_20260607/` adds the field-test hard-right
  samples to v1. It was used by float run `20260607_133611` and the currently
  deployed QAT run `20260607_133850`.
- `line_follow_v3_base_plus_hard_right_dedup_20261003/` is generated from v2 by
  removing identical content and split leakage. It is the next training
  candidate and has not trained a deployed model yet.

Versions v1 and v2 intentionally retain duplicate images and split leakage so
their existing training runs can be investigated or reproduced exactly.

The v2 dataset contains 1,265 image files representing 1,086
unique contents. There are 103 hashes that occur in more than one split.

## Building v3

The v3 dataset is generated from v2 with:

```bash
cd aicam_laptop
python3 line_follow/scripts/build_v3_dedup.py
```

Images are grouped by SHA-256 before split assignment. If identical content
appeared in multiple splits, it is assigned to `test` first, then `val`, then
`train`. Identical content with conflicting labels stops the build.

Generated filenames use the image hash. `manifest.csv` records every original
path, and `summary.json` records the resulting class counts and removed copies.

The current clean build contains 1,086 unique images:

| Split | LEFT | RIGHT | STRAIGHT |
| --- | ---: | ---: | ---: |
| train | 144 | 285 | 275 |
| val | 26 | 58 | 58 |
| test | 43 | 90 | 107 |

This is a new evaluation dataset. It must not be used to claim direct
comparability with metrics produced from v1 or v2.

The complete `datasets/` directory is tracked with DVC, not Git.
