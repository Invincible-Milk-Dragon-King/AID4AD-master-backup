#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maptr
cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export CUDA_VISIBLE_DEVICES=0,1
export GPUS=2
export NPROC=2
export PORT=29561
mkdir -p ../../exp_results/maptr
echo "[START] Probe(C→C) train on GPUs ${CUDA_VISIBLE_DEVICES}"
bash 30_train_probe.sh maptr camera \
  /workspace/AID4AD-master/111/checkpoints/maptr/maptr_tiny_r50_24e.pth \
  2>&1 | tee ../../exp_results/maptr/probe_c2c_train.log
echo "[DONE] Probe(C→C) train"
exec bash
