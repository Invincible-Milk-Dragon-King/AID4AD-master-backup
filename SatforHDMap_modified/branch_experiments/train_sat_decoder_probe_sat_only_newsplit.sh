#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29609}"
LOGDIR="${LOGDIR:-./branch_runs/sat_decoder_probe_sat_only_newsplit}"
CHECKPOINT="${CHECKPOINT:-./branch_runs/sat_only_newsplit/model_last.pt}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --is_newsplit \
  --instance_seg --direction_pred \
  --fusion_mode seg-masked-atten --align_fusion \
  --branch_mode sat_only \
  --experiment_name sat_decoder_probe_sat_only_newsplit \
  --prior_map_root ../AID4AD_ego_referenced \
  --train_decoder_only \
  --probe_feature_source sat_only_sat \
  --probe_encoder_checkpoint "${CHECKPOINT}" \
  --logdir "${LOGDIR}"
