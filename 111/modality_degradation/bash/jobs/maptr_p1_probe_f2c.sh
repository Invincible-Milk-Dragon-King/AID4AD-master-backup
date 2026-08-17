#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptr
cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export CUDA_VISIBLE_DEVICES=2,3
export GPUS=2
export NPROC=2
export PORT=29562
mkdir -p ../../exp_results/maptr
echo "[START] Probe(F→C) train on GPUs ${CUDA_VISIBLE_DEVICES} (force_camera_only)"
bash 30_train_probe.sh maptr fusion \
  /workspace/AID4AD-master/111/checkpoints/maptr/maptr_tiny_fusion_24e.pth \
  2>&1 | tee ../../exp_results/maptr/probe_f2c_train.log
echo "[DONE] Probe(F→C) train"
exec bash
