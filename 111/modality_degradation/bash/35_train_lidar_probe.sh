#!/usr/bin/env bash
set -euo pipefail

# Train lidar-branch probe from a pretrained baseline checkpoint.
#
# Usage:
#   bash 35_train_lidar_probe.sh <model> <source> <checkpoint>
#
# source:
#   - lidar:        use Full-L baseline checkpoint  -> Probe(L→L)
#   - fusion_lidar: use Full-F baseline checkpoint  -> Probe(F→L)
#
# Supported models (protocol 1): hdmapnet, himap
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 GPUS=1 bash 35_train_lidar_probe.sh hdmapnet lidar \
#     ../../HDMapNet/runs/hdmapnet_lidar/model0.pt
#   CUDA_VISIBLE_DEVICES=6,7 GPUS=6,7 NPROC=2 bash 35_train_lidar_probe.sh himap fusion_lidar \
#     ../../HIMap/work_dirs/himap_tiny_fusion_24e/epoch_24.pth

MODEL="${1:?model required (hdmapnet|himap)}"
SOURCE="${2:?source required (lidar|fusion_lidar)}"
CKPT="${3:?checkpoint required}"

case "${MODEL}" in
  hdmapnet|himap) ;;
  *)
    echo "[ERROR] LiDAR probe only supports hdmapnet|himap (got: ${MODEL})"
    exit 2
    ;;
esac
case "${SOURCE}" in
  lidar|fusion_lidar) ;;
  *)
    echo "[ERROR] source must be lidar|fusion_lidar (got: ${SOURCE})"
    exit 2
    ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
PORT="${PORT:-29576}"
if [[ -z "${GLOBAL_BATCH:-}" ]]; then
  GLOBAL_BATCH=16
fi
export PROBE_EPOCHS="${PROBE_EPOCHS:-24}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${GPU_LIST}  nproc_per_node=${NPROC}  global_batch=${GLOBAL_BATCH}  probe_epochs=${PROBE_EPOCHS}"

# HIMap auto-resumes latest.pth under work_dir; clear probe dirs before retrain.
if [[ "${MODEL}" == "himap" ]]; then
  PROBE_DIR="${ROOT}/HIMap/branch_runs/lidar_decoder_probe_himap_${SOURCE}"
  if [[ -d "${PROBE_DIR}" ]]; then
    BACKUP="${PROBE_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "[WARN] Moving existing HIMap probe dir -> ${BACKUP}"
    mv "${PROBE_DIR}" "${BACKUP}"
  fi
fi

python "${MD_ROOT}/run_probe.py" \
  --model "${MODEL}" \
  --source "${SOURCE}" \
  --checkpoint "${CKPT}" \
  --gpus "${GPU_LIST}" \
  --nproc-per-node "${NPROC}" \
  --master-port "${PORT}" \
  --global-batch "${GLOBAL_BATCH}"
