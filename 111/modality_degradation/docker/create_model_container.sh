#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash create_model_container.sh <model>
# model: hdmapnet | maptr | admap | gemap | himap

MODEL="${1:?model required}"
IMAGE_NAME="${IMAGE_NAME:-aid4ad-modality:cu118}"
ROOT_HOST="${ROOT_HOST:-/data2/file_swap/la_space/aid4ad/AID4AD-master}"
NUSCENES_HOST="${NUSCENES_HOST:-/data1/data/nuscenes}"

case "${MODEL}" in
  hdmapnet|maptr|admap|gemap|himap) ;;
  *) echo "Unsupported model: ${MODEL}" && exit 2 ;;
esac

CONTAINER_NAME="aid4ad-${MODEL}-exp"
ENV_CACHE_HOST="${ROOT_HOST}/docker_cache/conda_${MODEL}"
mkdir -p "${ENV_CACHE_HOST}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}" >/dev/null
  docker exec -it "${CONTAINER_NAME}" /bin/bash -l
  exit 0
fi

docker run --gpus all -it --name "${CONTAINER_NAME}" \
  --shm-size=32g \
  -v "${ROOT_HOST}:/workspace/AID4AD-master" \
  -v "${NUSCENES_HOST}:/workspace/datasets/nuscenes:ro" \
  -v "${ENV_CACHE_HOST}:/opt/conda/envs/${MODEL}" \
  -w /workspace/AID4AD-master/111 \
  "${IMAGE_NAME}" /bin/bash -l
