#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT="${CHECKPOINT:-./branch_runs/hdmapnet_surr_legacy/model_last.pt}"
DATAROOT="${DATAROOT:-../nuScenes}"
PRIOR_MAP_ROOT="${PRIOR_MAP_ROOT:-../AID4AD_ego_referenced}"
LOGDIR="${LOGDIR:-./branch_runs/hdmapnet_surr_legacy_eval}"
BSZ="${BSZ:-4}"
NWORKERS="${NWORKERS:-10}"
VERSION="${VERSION:-v1.0-trainval}"
MAP_THRESHOLDS="${MAP_THRESHOLDS:-0.2 0.5 1.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python test.py \
  --instance_seg --direction_pred \
  --model HDMapNet_cam_legacy \
  --branch_mode camera_only \
  --experiment_name hdmapnet_surr_legacy \
  --modelf "${CHECKPOINT}" \
  --dataroot "${DATAROOT}" \
  --prior_map_root "${PRIOR_MAP_ROOT}" \
  --version "${VERSION}" \
  --bsz "${BSZ}" \
  --nworkers "${NWORKERS}" \
  --map_thresholds ${MAP_THRESHOLDS} \
  --logdir "${LOGDIR}"
