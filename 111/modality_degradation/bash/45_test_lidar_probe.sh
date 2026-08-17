#!/usr/bin/env bash
set -euo pipefail

# LiDAR probe test: always evaluate lidar-branch capability (no camera / no fuser).
#
# Usage:
#   bash 45_test_lidar_probe.sh <model> <source> <probe_checkpoint>
#
# source:
#   lidar        -> Probe(L→L)   (test modality=lidar)
#   fusion_lidar -> Probe(F→L)   (test modality=fusion + force_lidar_only)

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

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

export FORCE_LIDAR_ONLY=1
unset FORCE_CAMERA_ONLY || true

case "${SOURCE}" in
  lidar)
    TEST_MODALITY="lidar"
    ;;
  fusion_lidar)
    # Keep fusion config/weights; force lidar-only forward via FORCE_LIDAR_ONLY.
    TEST_MODALITY="fusion"
    ;;
  *)
    echo "[ERROR] source must be lidar|fusion_lidar (got: ${SOURCE})"
    exit 2
    ;;
esac

bash "${_SCRIPT_DIR}/20_test_baselines.sh" "${MODEL}" "${TEST_MODALITY}" "${CKPT}"
