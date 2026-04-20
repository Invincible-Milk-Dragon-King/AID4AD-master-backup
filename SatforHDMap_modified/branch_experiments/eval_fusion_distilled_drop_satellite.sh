#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOGDIR="${LOGDIR:-./branch_runs/fusion_distilled_drop_satellite}"
MODEL="${MODEL:-./branch_runs/fusion_distilled_camera/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python test.py \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode drop_satellite \
  --experiment_name fusion_distilled_drop_satellite \
  --prior_map_root ../AID4AD_ego_referenced \
  --logdir "${LOGDIR}" \
  --modelf "${MODEL}"
