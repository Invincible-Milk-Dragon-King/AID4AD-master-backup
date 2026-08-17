#!/usr/bin/env bash
set -euo pipefail

# Train probe model from a pretrained baseline checkpoint.
#
# Usage:
#   bash 30_train_probe.sh <model> <source> <checkpoint>
#
# source:
#   - camera: use camera baseline checkpoint
#   - fusion: use fusion baseline checkpoint
#
# For mmdet models, set visible GPUs separately, e.g.:
#   CUDA_VISIBLE_DEVICES=6,7 GPUS=6,7 NPROC=2 bash 30_train_probe.sh maptr camera <ckpt>

MODEL="${1:?model required}"
SOURCE="${2:?source required (camera|fusion)}"
CKPT="${3:?checkpoint required}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
PORT="${PORT:-29566}"
# ADMap camera baseline already ~34GB @ samples_per_gpu=8; probe with the same
# batch hit CUDA illegal memory access on first iter. Use a smaller default.
if [[ -z "${GLOBAL_BATCH:-}" ]]; then
  case "${MODEL}:${SOURCE}" in
    admap:fusion) GLOBAL_BATCH=4 ;;
    admap:*) GLOBAL_BATCH=8 ;;
    *) GLOBAL_BATCH=16 ;;
  esac
fi
# GeMap baseline config is 110e; probe defaults to 24e via run_probe.py.
# Override with: PROBE_EPOCHS=24 bash 30_train_probe.sh ...
export PROBE_EPOCHS="${PROBE_EPOCHS:-24}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${GPU_LIST}  nproc_per_node=${NPROC}  global_batch=${GLOBAL_BATCH}  probe_epochs=${PROBE_EPOCHS}"

python "${MD_ROOT}/run_probe.py" \
  --model "${MODEL}" \
  --source "${SOURCE}" \
  --checkpoint "${CKPT}" \
  --gpus "${GPU_LIST}" \
  --nproc-per-node "${NPROC}" \
  --master-port "${PORT}" \
  --global-batch "${GLOBAL_BATCH}"
