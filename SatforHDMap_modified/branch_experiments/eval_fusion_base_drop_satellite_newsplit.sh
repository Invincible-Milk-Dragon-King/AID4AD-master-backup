#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOGDIR="${LOGDIR:-./branch_runs/fusion_base_drop_satellite_newsplit}"
MODEL="${MODEL:-./branch_runs/fusion_base_newsplit/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python test.py \
  --is_newsplit \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode drop_satellite \
  --experiment_name fusion_base_drop_satellite_newsplit \
  --prior_map_root ../AID4AD_ego_referenced \
  --logdir "${LOGDIR}" \
  --modelf "${MODEL}"
