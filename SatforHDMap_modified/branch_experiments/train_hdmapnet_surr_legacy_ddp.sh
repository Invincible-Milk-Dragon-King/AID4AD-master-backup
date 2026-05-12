#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29611}"
LOGDIR="${LOGDIR:-./branch_runs/hdmapnet_surr_legacy_ddp}"
DATAROOT="${DATAROOT:-../nuScenes}"
PRIOR_MAP_ROOT="${PRIOR_MAP_ROOT:-../AID4AD_ego_referenced}"
VERSION="${VERSION:-v1.0-trainval}"
BSZ="${BSZ:-2}"
NWORKERS="${NWORKERS:-16}"
LR="${LR:-1e-3}"
MAP_THRESHOLDS="${MAP_THRESHOLDS:-0.2 0.5 1.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --multi_gpu true \
  --instance_seg --direction_pred \
  --model HDMapNet_cam_legacy \
  --branch_mode camera_only \
  --experiment_name hdmapnet_surr_legacy_ddp \
  --dataroot "${DATAROOT}" \
  --prior_map_root "${PRIOR_MAP_ROOT}" \
  --version "${VERSION}" \
  --bsz "${BSZ}" \
  --nworkers "${NWORKERS}" \
  --lr "${LR}" \
  --map_thresholds ${MAP_THRESHOLDS} \
  --logdir "${LOGDIR}"
