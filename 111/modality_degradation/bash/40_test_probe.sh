#!/usr/bin/env bash
set -euo pipefail

# Probe test: always evaluate camera-branch capability (no LiDAR / no fuser).
#
# Usage:
#   bash 40_test_probe.sh <model> <source> <probe_checkpoint>
#
# source:
#   camera -> Probe(C→C)
#   fusion -> Probe(F→C)  (fusion structure/weights, camera-only forward)

MODEL="${1:?model required}"
SOURCE="${2:?source required (camera|fusion)}"
CKPT="${3:?checkpoint required}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

# Both probe columns measure camera-branch detection; never feed LiDAR.
export FORCE_CAMERA_ONLY=1

bash "${_SCRIPT_DIR}/20_test_baselines.sh" "${MODEL}" "${SOURCE}" "${CKPT}"
