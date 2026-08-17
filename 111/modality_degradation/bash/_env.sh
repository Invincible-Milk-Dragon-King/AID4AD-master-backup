#!/usr/bin/env bash
# Shared path resolution for host (/data2/...) and container (/workspace/...).
# Source from other scripts:  source "$(dirname "$0")/_env.sh"
#
# GPU usage (dual-GPU example):
#   export CUDA_VISIBLE_DEVICES=0,1
#   export GPUS=2
# Or shorthand (auto-normalized):
#   export GPUS=0,1          # -> CUDA_VISIBLE_DEVICES=0,1 and GPUS=2

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# bash/ -> modality_degradation/ -> 111/
ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
MD_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

# Prefer container mount if present, else host path, else env override.
if [[ -d /workspace/datasets/nuscenes ]]; then
  : "${NUSCENES_ROOT:=/workspace/datasets/nuscenes}"
elif [[ -d /data1/data/nuscenes ]]; then
  : "${NUSCENES_ROOT:=/data1/data/nuscenes}"
else
  : "${NUSCENES_ROOT:=/data1/data/nuscenes}"
fi

: "${GPUS:=1}"
: "${NPROC:=}"
: "${PORT:=29566}"

# If GPUS looks like a device list ("0,1"), treat it as CUDA_VISIBLE_DEVICES
# and convert GPUS to process count for dist_train / dist_test.
if [[ "${GPUS}" == *","* ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  IFS=',' read -r -a _GPU_IDS <<< "${GPUS}"
  GPUS="${#_GPU_IDS[@]}"
  unset _GPU_IDS
fi

# NPROC defaults to GPUS (process count) for probe launcher
if [[ -z "${NPROC}" ]]; then
  NPROC="${GPUS}"
fi

# Device list for run_probe.py --gpus (CUDA_VISIBLE_DEVICES style)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="${CUDA_VISIBLE_DEVICES}"
else
  # e.g. GPUS=2 -> 0,1
  GPU_LIST="$(seq -s, 0 $((GPUS - 1)))"
fi

mkdir -p "${ROOT}/exp_results/maptr" \
         "${ROOT}/exp_results/gemap" \
         "${ROOT}/exp_results/hdmapnet" \
         "${ROOT}/exp_results/admap" \
         "${ROOT}/exp_results/himap" \
         "${ROOT}/checkpoints/maptr" \
         "${ROOT}/checkpoints/gemap"

link_nuscenes() {
  local repo="$1"
  mkdir -p "${repo}/data"
  ln -sfn "${NUSCENES_ROOT}" "${repo}/data/nuscenes"
}

# Share ImageNet backbone weights already present under MapTR/ckpts.
# Use relative symlinks so they resolve both on host (/data2/...) and in
# container (/workspace/...), which mount different absolute prefixes.
link_maptr_backbone_ckpts() {
  local repo="$1"
  mkdir -p "${repo}/ckpts"
  local rel_maptr
  rel_maptr="$(realpath --relative-to="${repo}/ckpts" "${ROOT}/MapTR/ckpts" 2>/dev/null || python3 -c "import os; print(os.path.relpath('${ROOT}/MapTR/ckpts', '${repo}/ckpts'))")"
  if [[ -f "${ROOT}/MapTR/ckpts/resnet50-19c8e357.pth" ]]; then
    ln -sfn "${rel_maptr}/resnet50-19c8e357.pth" "${repo}/ckpts/resnet50-19c8e357.pth"
  fi
  if [[ -f "${ROOT}/MapTR/ckpts/resnet18-f37072fd.pth" ]]; then
    ln -sfn "${rel_maptr}/resnet18-f37072fd.pth" "${repo}/ckpts/resnet18-f37072fd.pth"
  fi
}

# GeMap expects data/hdmap/{nuscenes,can_bus} (see GeMap/docs/prepare_dataset.md).
link_gemap_hdmap() {
  local repo="$1"
  mkdir -p "${repo}/data/hdmap"
  ln -sfn "${NUSCENES_ROOT}" "${repo}/data/hdmap/nuscenes"
  # Official unzip is often nested as can_bus/can_bus/*.json
  if [[ -d "${NUSCENES_ROOT}/can_bus/can_bus" ]]; then
    ln -sfn "${NUSCENES_ROOT}/can_bus/can_bus" "${repo}/data/hdmap/can_bus"
  elif [[ -d "${NUSCENES_ROOT}/can_bus" ]]; then
    ln -sfn "${NUSCENES_ROOT}/can_bus" "${repo}/data/hdmap/can_bus"
  fi
}
