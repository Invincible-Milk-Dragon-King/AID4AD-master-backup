#!/usr/bin/env bash
set -euo pipefail

# End-to-end LiDAR modality-degradation recipe for HDMapNet + HIMap (protocol 1).
# Edit CKPT paths after Full-L / Full-F finishes, then run sections.
#
# Protocol:
#   Full-L  -> 10_train_baselines.sh <model> lidar  (24e)
#   Probe(L→L) / Probe(F→L) -> freeze lidar-side encoder, train lidar decoder
#   Δ_lidar = Probe(F→L) - Probe(L→L)
#
# Prerequisites (same containers/envs as camera probes):
#   hdmapnet: conda activate hdmapnet
#   himap:    conda activate himap
#
# Usage (from modality_degradation/bash):
#   bash run_lidar_degradation_hdmapnet_himap.sh

ROOT_BASH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_BASH}/_env.sh"
mkdir -p "${ROOT}/exp_results/hdmapnet" "${ROOT}/exp_results/himap"

########################################
# 0) GPU assignment (edit me)
########################################
# Example dual-GPU:
#   export CUDA_VISIBLE_DEVICES=0,1
#   export GPUS=0,1   # _env.sh normalizes to process count
: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES
export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES}}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-24}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-16}"

########################################
# 1) Train Full-L baselines (24e)
########################################
# Uncomment when ready:
# bash "${ROOT_BASH}/10_train_baselines.sh" hdmapnet lidar \
#   2>&1 | tee "${ROOT}/exp_results/hdmapnet/train_l.log"
# bash "${ROOT_BASH}/10_train_baselines.sh" himap lidar \
#   2>&1 | tee "${ROOT}/exp_results/himap/train_l.log"

########################################
# 2) Fill checkpoints after training
########################################
# HDMapNet: train.py saves model{epoch}.pt under logdir; pick best or last.
HDMAPNET_L_CKPT="${HDMAPNET_L_CKPT:-${ROOT}/HDMapNet/runs/hdmapnet_lidar/model23.pt}"
HDMAPNET_F_CKPT="${HDMAPNET_F_CKPT:-${ROOT}/HDMapNet/runs/hdmapnet_fusion/model29.pt}"

HIMAP_L_CKPT="${HIMAP_L_CKPT:-${ROOT}/HIMap/work_dirs/himap_tiny_lidar_24e/epoch_24.pth}"
HIMAP_F_CKPT="${HIMAP_F_CKPT:-${ROOT}/HIMap/work_dirs/himap_tiny_fusion_24e/epoch_24.pth}"

########################################
# 3) Test Full-L (and optionally Full-F again)
########################################
# bash "${ROOT_BASH}/20_test_baselines.sh" hdmapnet lidar "${HDMAPNET_L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/hdmapnet/baseline_l.log"
# bash "${ROOT_BASH}/20_test_baselines.sh" himap lidar "${HIMAP_L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/himap/baseline_l.log"

########################################
# 4) Train probes
########################################
# HDMapNet Probe(L→L)
# bash "${ROOT_BASH}/35_train_lidar_probe.sh" hdmapnet lidar "${HDMAPNET_L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/hdmapnet/probe_l2l_train.log"
# HDMapNet Probe(F→L)
# bash "${ROOT_BASH}/35_train_lidar_probe.sh" hdmapnet fusion_lidar "${HDMAPNET_F_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/hdmapnet/probe_f2l_train.log"

# HIMap Probe(L→L)  — script auto-backs up old branch_runs to avoid auto-resume
# bash "${ROOT_BASH}/35_train_lidar_probe.sh" himap lidar "${HIMAP_L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/himap/probe_l2l_train.log"
# HIMap Probe(F→L)
# bash "${ROOT_BASH}/35_train_lidar_probe.sh" himap fusion_lidar "${HIMAP_F_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/himap/probe_f2l_train.log"

########################################
# 5) Test probes (fill probe ckpt paths after train)
########################################
# HDMAPNET_L2L_CKPT="${ROOT}/HDMapNet/branch_runs/lidar_decoder_probe_hdmapnet_lidar/model23.pt"
# HDMAPNET_F2L_CKPT="${ROOT}/HDMapNet/branch_runs/lidar_decoder_probe_hdmapnet_fusion_lidar/model23.pt"
# HIMAP_L2L_CKPT="${ROOT}/HIMap/branch_runs/lidar_decoder_probe_himap_lidar/latest.pth"
# HIMAP_F2L_CKPT="${ROOT}/HIMap/branch_runs/lidar_decoder_probe_himap_fusion_lidar/latest.pth"
#
# bash "${ROOT_BASH}/45_test_lidar_probe.sh" hdmapnet lidar "${HDMAPNET_L2L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/hdmapnet/probe_l2l.log"
# bash "${ROOT_BASH}/45_test_lidar_probe.sh" hdmapnet fusion_lidar "${HDMAPNET_F2L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/hdmapnet/probe_f2l.log"
# bash "${ROOT_BASH}/45_test_lidar_probe.sh" himap lidar "${HIMAP_L2L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/himap/probe_l2l.log"
# bash "${ROOT_BASH}/45_test_lidar_probe.sh" himap fusion_lidar "${HIMAP_F2L_CKPT}" \
#   2>&1 | tee "${ROOT}/exp_results/himap/probe_f2l.log"

echo "[INFO] Template ready. Uncomment sections after setting GPU/ckpt paths."
echo "[INFO] HDMAPNET_L_CKPT=${HDMAPNET_L_CKPT}"
echo "[INFO] HDMAPNET_F_CKPT=${HDMAPNET_F_CKPT}"
echo "[INFO] HIMAP_L_CKPT=${HIMAP_L_CKPT}"
echo "[INFO] HIMAP_F_CKPT=${HIMAP_F_CKPT}"
