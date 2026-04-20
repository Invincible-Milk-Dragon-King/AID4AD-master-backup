#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29506}"
LOGDIR="${LOGDIR:-./branch_runs/camera_decoder_probe_camera_only}"
CHECKPOINT="${CHECKPOINT:-./branch_runs/camera_only/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode camera_only \
  --experiment_name camera_decoder_probe_camera_only \
  --prior_map_root ../AID4AD_ego_referenced \
  --train_decoder_only \
  --probe_feature_source camera_only_camera \
  --probe_encoder_checkpoint "${CHECKPOINT}" \
  --logdir "${LOGDIR}"
