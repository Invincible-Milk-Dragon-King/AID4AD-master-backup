#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT="${CHECKPOINT:-./branch_runs/camera_only/model_last.pt}"
DATAROOT="${DATAROOT:-../nuScenes}"
PRIOR_MAP_ROOT="${PRIOR_MAP_ROOT:-../AID4AD_ego_referenced}"
LOGDIR="${LOGDIR:-./branch_runs/camera_only_eval}"
BSZ="${BSZ:-4}"
NWORKERS="${NWORKERS:-10}"
VERSION="${VERSION:-v1.0-trainval}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python test.py \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode camera_only \
  --experiment_name camera_only \
  --modelf "${CHECKPOINT}" \
  --dataroot "${DATAROOT}" \
  --prior_map_root "${PRIOR_MAP_ROOT}" \
  --version "${VERSION}" \
  --bsz "${BSZ}" \
  --nworkers "${NWORKERS}" \
  --logdir "${LOGDIR}"
