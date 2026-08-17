# Modality Degradation Probe Setup

This folder provides a unified launcher for camera degradation probe experiments
for the five selected models:

- `hdmapnet`
- `maptr`
- `admap`
- `gemap`
- `himap`

The setup follows your SatforHDMap-style protocol:

1. Load an existing encoder checkpoint (`camera` or `fusion`)
2. Freeze backbone / encoder
3. Train only a camera decoder probe
4. Evaluate with mIoU and mAP

Two launch styles are supported:

- `satfor_probe` (currently `hdmapnet`): uses explicit probe args like
  `--train_decoder_only`, `--probe_feature_source`, and
  `--probe_encoder_checkpoint`.
- `mmdet_probe` (currently `maptr`, `admap`, `gemap`, `himap`): uses each
  project's `tools/train.py` with camera/fusion config plus
  `--cfg-options load_from=<checkpoint>` and strict decoder-only freeze
  (`--decoder-only` with train allowlist + encoder/fuser freeze denylist).

## Files

- `model_registry.json`: model paths and command templates
- `run_probe.py`: run one probe experiment
- `run_all_probes.py`: batch run all configured experiments

## Quick Start

Run one model + one source:

```bash
python modality_degradation/run_probe.py \
  --model hdmapnet \
  --source camera \
  --checkpoint /abs/path/to/camera_model_last.pt \
  --gpus 0,1
```

Run one model with both `camera` and `fusion` sources:

```bash
python modality_degradation/run_all_probes.py \
  --models hdmapnet \
  --checkpoint-camera /abs/path/to/camera_model_last.pt \
  --checkpoint-fusion /abs/path/to/fusion_model_last.pt \
  --gpus 0,1
```

## Notes

- `ADMap-main.zip` is extracted to `111/ADMap-main`.
- `HIMap(1).zip` is extracted to `111/HIMap`.
- Commands are intentionally explicit and editable to match each codebase.
- Set `--dry-run` to preview full command without execution.
