#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-aid4ad-modality:cu118}"

docker build -f "${SCRIPT_DIR}/Dockerfile.modality" -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
echo "[DONE] Built image: ${IMAGE_NAME}"
