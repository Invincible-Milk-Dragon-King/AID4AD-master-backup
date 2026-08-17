#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash 20_test_baselines.sh <model> <modality> <checkpoint_path>
#
# For camera-branch probe evaluation (skip LiDAR / fuser):
#   FORCE_CAMERA_ONLY=1 bash 20_test_baselines.sh maptr fusion <probe_ckpt>
#   (also set automatically by 40_test_probe.sh)
#
# For lidar-branch probe evaluation (skip camera / fuser):
#   FORCE_LIDAR_ONLY=1 bash 20_test_baselines.sh himap fusion <probe_ckpt>
#   (also set automatically by 45_test_lidar_probe.sh)

MODEL="${1:?model required}"
MODALITY="${2:?modality required}"
CKPT="${3:?checkpoint path required}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  nproc_per_node=${GPUS}  FORCE_CAMERA_ONLY=${FORCE_CAMERA_ONLY:-0}  FORCE_LIDAR_ONLY=${FORCE_LIDAR_ONLY:-0}"

if [[ "${FORCE_CAMERA_ONLY:-0}" == "1" && "${FORCE_LIDAR_ONLY:-0}" == "1" ]]; then
  echo "[ERROR] FORCE_CAMERA_ONLY and FORCE_LIDAR_ONLY cannot both be 1"
  exit 2
fi

MMDET_EXTRA=()
if [[ "${FORCE_CAMERA_ONLY:-0}" == "1" ]]; then
  MMDET_EXTRA+=(--cfg-options model.force_camera_only=True)
fi
if [[ "${FORCE_LIDAR_ONLY:-0}" == "1" ]]; then
  MMDET_EXTRA+=(--cfg-options model.force_lidar_only=True)
fi

case "${MODEL}:${MODALITY}" in
  hdmapnet:camera|hdmapnet:fusion|hdmapnet:lidar)
    cd "${ROOT}/HDMapNet"
    mkdir -p dataset outputs
    ln -sfn "${NUSCENES_ROOT}" dataset/nuScenes
    if [[ "${MODALITY}" == "camera" ]]; then M="HDMapNet_cam"; fi
    if [[ "${MODALITY}" == "fusion" ]]; then M="HDMapNet_fusion"; fi
    if [[ "${MODALITY}" == "lidar" ]]; then M="HDMapNet_lidar"; fi
    BRANCH_ARGS=()
    if [[ "${FORCE_CAMERA_ONLY:-0}" == "1" ]]; then
      # Probe eval: keep fusion weights when source=fusion, but camera-only forward.
      BRANCH_ARGS+=(--branch_mode camera_only)
      if [[ "${MODALITY}" == "fusion" ]]; then
        M="HDMapNet_fusion"
      fi
    fi
    if [[ "${FORCE_LIDAR_ONLY:-0}" == "1" ]]; then
      BRANCH_ARGS+=(--branch_mode lidar_only)
      if [[ "${MODALITY}" == "fusion" ]]; then
        M="HDMapNet_fusion"
      else
        M="HDMapNet_lidar"
      fi
    elif [[ "${MODALITY}" == "lidar" ]]; then
      # Full-L baseline uses isomorphic lidar branch.
      BRANCH_ARGS+=(--branch_mode lidar_only)
    fi
    OUT_JSON="outputs/${M}_${MODALITY}_c${FORCE_CAMERA_ONLY:-0}_l${FORCE_LIDAR_ONLY:-0}_pred.json"
    # HDMapNet eval is single-process; prefer first visible GPU. Retry once on
    # intermittent torch `_C` import failures under multi-job GPU load.
    run_hdmapnet_export() {
      python -u export_pred_to_json.py \
        --model "${M}" \
        --version v1.0-trainval \
        --dataroot dataset/nuScenes \
        --modelf "${CKPT}" \
        --output "${OUT_JSON}" \
        "${BRANCH_ARGS[@]}"
    }
    if ! run_hdmapnet_export; then
      echo "[WARN] export_pred_to_json failed once; retrying..."
      sleep 2
      run_hdmapnet_export
    fi
    python -u evaluate_json.py --version v1.0-trainval --dataroot dataset/nuScenes --eval_set val --result_path "${OUT_JSON}"
    ;;

  maptr:camera)
    cd "${ROOT}/MapTR"
    link_nuscenes "${ROOT}/MapTR"
    bash tools/dist_test_map.sh projects/configs/maptr/maptr_tiny_r50_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;
  maptr:fusion)
    cd "${ROOT}/MapTR"
    link_nuscenes "${ROOT}/MapTR"
    bash tools/dist_test_map.sh projects/configs/maptr/maptr_tiny_fusion_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;

  admap:camera)
    cd "${ROOT}/ADMap-main"
    link_nuscenes "${ROOT}/ADMap-main"
    bash tools/dist_test_map.sh configs/ADMap_cam_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;
  admap:fusion)
    cd "${ROOT}/ADMap-main"
    link_nuscenes "${ROOT}/ADMap-main"
    bash tools/dist_test_map.sh configs/ADMap_fusion_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;

  gemap:camera)
    # Fair modality pair: simple-camera (not gemap_full).
    cd "${ROOT}/GeMap"
    link_gemap_hdmap "${ROOT}/GeMap"
    bash tools/dist_test_map.sh projects/configs/gemap/gemap_simple_r50_110ep.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;
  gemap:fusion)
    cd "${ROOT}/GeMap"
    link_gemap_hdmap "${ROOT}/GeMap"
    bash tools/dist_test_map.sh projects/configs/gemap/gemap_simple_r50_sec_110ep.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;
  gemap:full_camera)
    # Optional SOTA camera row only; do NOT compare Δ with simple-fusion.
    cd "${ROOT}/GeMap"
    link_gemap_hdmap "${ROOT}/GeMap"
    bash tools/dist_test_map.sh projects/configs/gemap/gemap_full_r50_110ep.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;

  himap:camera)
    cd "${ROOT}/HIMap"
    link_nuscenes "${ROOT}/HIMap"
    link_maptr_backbone_ckpts "${ROOT}/HIMap"
    bash tools/dist_test_map.sh projects/configs/sensor_missing/himap_tiny_cam_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;
  himap:fusion)
    cd "${ROOT}/HIMap"
    link_nuscenes "${ROOT}/HIMap"
    link_maptr_backbone_ckpts "${ROOT}/HIMap"
    bash tools/dist_test_map.sh projects/configs/sensor_missing/himap_tiny_fusion_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;
  himap:lidar)
    cd "${ROOT}/HIMap"
    link_nuscenes "${ROOT}/HIMap"
    link_maptr_backbone_ckpts "${ROOT}/HIMap"
    bash tools/dist_test_map.sh projects/configs/sensor_missing/himap_tiny_lidar_24e.py "${CKPT}" "${GPUS}" "${MMDET_EXTRA[@]}"
    ;;

  *)
    echo "Unsupported pair: ${MODEL}:${MODALITY}"
    exit 2
    ;;
esac
