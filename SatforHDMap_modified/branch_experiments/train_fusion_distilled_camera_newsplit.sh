#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29604}"
LOGDIR="${LOGDIR:-./branch_runs/fusion_distilled_camera_newsplit}"
TEACHER="${TEACHER:-./branch_runs/camera_only_newsplit/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --is_newsplit \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode fusion \
  --experiment_name fusion_distilled_camera_newsplit \
  --prior_map_root ../AID4AD_ego_referenced \
  --teacher_model_root "${TEACHER}" \
  --teacher_branch_mode camera_only \
  --distill_feature camera_branch_feature \
  --distill_loss mse \
  --distill_weight 1.0 \
  --logdir "${LOGDIR}"
