#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOGDIR="${LOGDIR:-./branch_runs/fusion_base_drop_camera}"
MODEL="${MODEL:-./branch_runs/fusion_base/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python test.py \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode drop_camera \
  --experiment_name fusion_base_drop_camera \
  --prior_map_root ../AID4AD_ego_referenced \
  --logdir "${LOGDIR}" \
  --modelf "${MODEL}"
