# SatforHDMap Branch Experiments

## Script Directory

All Satfor branch experiment scripts are stored in:

`SatforHDMap_modified/branch_experiments/`

Each script also has a matching `*_newsplit.sh` variant.

- the original `.sh` scripts run the old split
- the `*_newsplit.sh` scripts append `--is_newsplit`
- the `*_newsplit.sh` scripts also write into separate `_newsplit` output directories, so they do not overwrite old-split runs

## Default GPU And Port Settings

All scripts now default to:

- `CUDA_VISIBLE_DEVICES=0,1`

Training scripts additionally default to:

- `NPROC_PER_NODE=2`
- one unique `MASTER_PORT` per script for tmux parallel runs

Default training ports:

- `train_camera_only.sh` -> `29501`
- `train_sat_only.sh` -> `29502`
- `train_fusion_base.sh` -> `29503`
- `train_fusion_distilled_camera.sh` -> `29504`
- `train_fusion_distilled_satellite.sh` -> `29505`
- `train_camera_decoder_probe_camera_only.sh` -> `29506`
- `train_camera_decoder_probe_fusion_base.sh` -> `29507`
- `train_camera_decoder_probe_fusion_distilled.sh` -> `29508`
- `train_sat_decoder_probe_sat_only.sh` -> `29509`
- `train_sat_decoder_probe_fusion_base.sh` -> `29510`
- `train_sat_decoder_probe_fusion_distilled.sh` -> `29511`

If you launch multiple jobs manually in tmux, you can still override ports, for example:

`MASTER_PORT=29611 bash branch_experiments/train_fusion_base.sh`

The `_newsplit.sh` training scripts use ports `29601` through `29611` by default.

## Output Directory

All experiment outputs are stored under:

`SatforHDMap_modified/branch_runs/<experiment_name>/`

All AP / mAP calculations now default to:

- `MAP_THRESHOLDS=0.2 0.5 1.0`

This applies to training-time final evaluation, standalone `test.py` evaluation scripts, and `eval_vectors_paper_ap.sh`.

Each training experiment writes:

- `results.log`
- `model_last.pt`
- final test vector export: `<experiment>_<branch_mode>_<split>_in_memory_vectors.json`
- final test metrics export: `<experiment>_<branch_mode>_<split>_in_memory_metrics.json`

Each evaluation script writes:

- `results.log`
- vector export: `<experiment>_<branch_mode>_<split>_<checkpoint_name>_vectors.json`
- metrics export: `<experiment>_<branch_mode>_<split>_<checkpoint_name>_metrics.json`

## Camera-Side Experiments

- `train_camera_only.sh` -> `branch_runs/camera_only/`
- `train_fusion_base.sh` -> `branch_runs/fusion_base/`
- `eval_fusion_base_drop_satellite.sh` -> `branch_runs/fusion_base_drop_satellite/`
- `train_fusion_distilled_camera.sh` -> `branch_runs/fusion_distilled_camera/`
- `eval_fusion_distilled_drop_satellite.sh` -> `branch_runs/fusion_distilled_drop_satellite/`
- `train_camera_decoder_probe_camera_only.sh` -> `branch_runs/camera_decoder_probe_camera_only/`
- `train_camera_decoder_probe_fusion_base.sh` -> `branch_runs/camera_decoder_probe_fusion_base/`
- `train_camera_decoder_probe_fusion_distilled.sh` -> `branch_runs/camera_decoder_probe_fusion_distilled/`

## Satellite-Side Experiments

- `train_sat_only.sh` -> `branch_runs/sat_only/`
- `eval_fusion_base_drop_camera.sh` -> `branch_runs/fusion_base_drop_camera/`
- `train_fusion_distilled_satellite.sh` -> `branch_runs/fusion_distilled_satellite/`
- `eval_fusion_distilled_drop_camera.sh` -> `branch_runs/fusion_distilled_drop_camera/`
- `train_sat_decoder_probe_sat_only.sh` -> `branch_runs/sat_decoder_probe_sat_only/`
- `train_sat_decoder_probe_fusion_base.sh` -> `branch_runs/sat_decoder_probe_fusion_base/`
- `train_sat_decoder_probe_fusion_distilled.sh` -> `branch_runs/sat_decoder_probe_fusion_distilled/`

## Decoder-Only Probe Experiments

These six probe runs isolate BEV representation quality from decoder co-adaptation:

- load a trained checkpoint
- freeze the corresponding encoder path
- randomly reinitialize the single-modal decoder
- train only that decoder with the normal segmentation / instance / direction losses

Camera-side probe checkpoints:

- `train_camera_decoder_probe_camera_only.sh` uses `branch_runs/camera_only/model_last.pt`
- `train_camera_decoder_probe_fusion_base.sh` uses `branch_runs/fusion_base/model_last.pt`
- `train_camera_decoder_probe_fusion_distilled.sh` uses `branch_runs/fusion_distilled_camera/model_last.pt`

Satellite-side probe checkpoints:

- `train_sat_decoder_probe_sat_only.sh` uses `branch_runs/sat_only/model_last.pt`
- `train_sat_decoder_probe_fusion_base.sh` uses `branch_runs/fusion_base/model_last.pt`
- `train_sat_decoder_probe_fusion_distilled.sh` uses `branch_runs/fusion_distilled_satellite/model_last.pt`

Interpretation:

- if two probe runs use the same decoder architecture and training recipe, the remaining gap mostly reflects the BEV feature quality produced by the frozen encoder branch
- camera probes should be compared against the original `camera_only` and `fusion_*_drop_satellite` results
- satellite probes should be compared against the original `sat_only` and `fusion_*_drop_camera` results
