#!/usr/bin/env bash
# Run AFTER probe_f2c train finishes. Pass epoch ckpt as $1 or set PROBE_CKPT.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptr
cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export CUDA_VISIBLE_DEVICES=6,7
export GPUS=2
export NPROC=2
export PORT=29572
export FORCE_CAMERA_ONLY=1
CKPT="${1:-${PROBE_CKPT:-}}"
if [[ -z "${CKPT}" ]]; then
  CKPT=$(ls -1t /workspace/AID4AD-master/111/MapTR/branch_runs/camera_decoder_probe_maptr_fusion/epoch_*.pth 2>/dev/null | head -1 || true)
fi
[[ -n "${CKPT}" && -f "${CKPT}" ]] || { echo "Need probe ckpt. Usage: $0 <epoch_xx.pth>"; exit 1; }
mkdir -p ../../exp_results/maptr
echo "[START] Probe(F→C) test  ckpt=${CKPT}"
bash 40_test_probe.sh maptr fusion "${CKPT}" 2>&1 | tee ../../exp_results/maptr/probe_f2c.log
echo "[DONE] Probe(F→C) test"
exec bash
