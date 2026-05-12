#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
LOGDIR="${LOGDIR:-./branch_runs/hdmapnet_surr_legacy}"
DATAROOT="${DATAROOT:-../nuScenes}"
PRIOR_MAP_ROOT="${PRIOR_MAP_ROOT:-../AID4AD_ego_referenced}"
VERSION="${VERSION:-v1.0-trainval}"
BSZ="${BSZ:-4}"
NWORKERS="${NWORKERS:-10}"
LR="${LR:-1e-3}"
MAP_THRESHOLDS="${MAP_THRESHOLDS:-0.2 0.5 1.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
  --multi_gpu false \
  --instance_seg --direction_pred \
  --model HDMapNet_cam_legacy \
  --branch_mode camera_only \
  --experiment_name hdmapnet_surr_legacy \
  --dataroot "${DATAROOT}" \
  --prior_map_root "${PRIOR_MAP_ROOT}" \
  --version "${VERSION}" \
  --bsz "${BSZ}" \
  --nworkers "${NWORKERS}" \
  --lr "${LR}" \
  --map_thresholds ${MAP_THRESHOLDS} \
  --logdir "${LOGDIR}"
