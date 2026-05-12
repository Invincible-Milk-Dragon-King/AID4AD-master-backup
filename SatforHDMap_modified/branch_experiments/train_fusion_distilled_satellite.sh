#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29505}"
LOGDIR="${LOGDIR:-./branch_runs/fusion_distilled_satellite}"
TEACHER="${TEACHER:-./branch_runs/sat_only/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode fusion \
  --experiment_name fusion_distilled_satellite \
  --prior_map_root ../AID4AD_ego_referenced \
  --teacher_model_root "${TEACHER}" \
  --teacher_branch_mode sat_only \
  --distill_feature satellite_branch_feature \
  --distill_loss mse \
  --distill_weight 1.0 \
  --logdir "${LOGDIR}"
