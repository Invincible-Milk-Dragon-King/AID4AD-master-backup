#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash 10_train_baselines.sh hdmapnet camera
#   bash 10_train_baselines.sh hdmapnet fusion
#   bash 10_train_baselines.sh hdmapnet lidar
#   bash 10_train_baselines.sh maptr camera
#   bash 10_train_baselines.sh maptr fusion
#   bash 10_train_baselines.sh admap camera
#   bash 10_train_baselines.sh admap fusion
#   bash 10_train_baselines.sh gemap camera
#   bash 10_train_baselines.sh gemap fusion
#   bash 10_train_baselines.sh himap camera
#   bash 10_train_baselines.sh himap fusion
#   bash 10_train_baselines.sh himap lidar

MODEL="${1:?model required}"
MODALITY="${2:?modality required}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  nproc_per_node=${GPUS}"

case "${MODEL}:${MODALITY}" in
  hdmapnet:camera)
    cd "${ROOT}/HDMapNet"
    mkdir -p dataset
    ln -sfn "${NUSCENES_ROOT}" dataset/nuScenes
    python train.py --instance_seg --direction_pred --model HDMapNet_cam --version v1.0-trainval --dataroot dataset/nuScenes --logdir runs/hdmapnet_cam
    ;;
  hdmapnet:fusion)
    cd "${ROOT}/HDMapNet"
    mkdir -p dataset
    ln -sfn "${NUSCENES_ROOT}" dataset/nuScenes
    python train.py --instance_seg --direction_pred --model HDMapNet_fusion --version v1.0-trainval --dataroot dataset/nuScenes --logdir runs/hdmapnet_fusion
    ;;
  hdmapnet:lidar)
    cd "${ROOT}/HDMapNet"
    mkdir -p dataset
    ln -sfn "${NUSCENES_ROOT}" dataset/nuScenes
    # Protocol-1 Full-L: isomorphic lidar branch (pp + lidar_bevencode), 24e.
    python train.py --instance_seg --direction_pred --model HDMapNet_lidar \
      --branch_mode lidar_only --nepochs 24 \
      --version v1.0-trainval --dataroot dataset/nuScenes --logdir runs/hdmapnet_lidar
    ;;

  maptr:camera)
    cd "${ROOT}/MapTR"
    link_nuscenes "${ROOT}/MapTR"
    bash tools/dist_train.sh projects/configs/maptr/maptr_tiny_r50_24e.py "${GPUS}" --work-dir work_dirs/maptr_tiny_r50_24e_camera
    ;;
  maptr:fusion)
    cd "${ROOT}/MapTR"
    link_nuscenes "${ROOT}/MapTR"
    bash tools/dist_train.sh projects/configs/maptr/maptr_tiny_fusion_24e.py "${GPUS}" --work-dir work_dirs/maptr_tiny_fusion_24e
    ;;

  admap:camera)
    cd "${ROOT}/ADMap-main"
    link_nuscenes "${ROOT}/ADMap-main"
    link_maptr_backbone_ckpts "${ROOT}/ADMap-main"
    bash tools/dist_train.sh configs/ADMap_cam_24e.py "${GPUS}" --work-dir work_dirs/admap_cam_24e
    ;;
  admap:fusion)
    cd "${ROOT}/ADMap-main"
    link_nuscenes "${ROOT}/ADMap-main"
    link_maptr_backbone_ckpts "${ROOT}/ADMap-main"
    bash tools/dist_train.sh configs/ADMap_fusion_24e.py "${GPUS}" --work-dir work_dirs/admap_fusion_24e
    ;;

  gemap:camera)
    # Fair modality pair: simple-camera (not gemap_full).
    cd "${ROOT}/GeMap"
    link_gemap_hdmap "${ROOT}/GeMap"
    bash tools/dist_train.sh projects/configs/gemap/gemap_simple_r50_110ep.py "${GPUS}" --work-dir work_dirs/gemap_simple_r50_110ep_camera
    ;;
  gemap:fusion)
    cd "${ROOT}/GeMap"
    link_gemap_hdmap "${ROOT}/GeMap"
    bash tools/dist_train.sh projects/configs/gemap/gemap_simple_r50_sec_110ep.py "${GPUS}" --work-dir work_dirs/gemap_simple_r50_sec_110ep_fusion
    ;;
  gemap:full_camera)
    # Optional SOTA camera baseline; not used for modality Δ vs fusion.
    cd "${ROOT}/GeMap"
    link_gemap_hdmap "${ROOT}/GeMap"
    bash tools/dist_train.sh projects/configs/gemap/gemap_full_r50_110ep.py "${GPUS}" --work-dir work_dirs/gemap_full_r50_110ep_camera
    ;;

  himap:camera)
    cd "${ROOT}/HIMap"
    link_nuscenes "${ROOT}/HIMap"
    link_maptr_backbone_ckpts "${ROOT}/HIMap"
    bash tools/dist_train.sh projects/configs/sensor_missing/himap_tiny_cam_24e.py "${GPUS}" --work-dir work_dirs/himap_tiny_cam_24e
    ;;
  himap:fusion)
    cd "${ROOT}/HIMap"
    link_nuscenes "${ROOT}/HIMap"
    link_maptr_backbone_ckpts "${ROOT}/HIMap"
    bash tools/dist_train.sh projects/configs/sensor_missing/himap_tiny_fusion_24e.py "${GPUS}" --work-dir work_dirs/himap_tiny_fusion_24e
    ;;
  himap:lidar)
    cd "${ROOT}/HIMap"
    link_nuscenes "${ROOT}/HIMap"
    link_maptr_backbone_ckpts "${ROOT}/HIMap"
    bash tools/dist_train.sh projects/configs/sensor_missing/himap_tiny_lidar_24e.py "${GPUS}" --work-dir work_dirs/himap_tiny_lidar_24e
    ;;

  *)
    echo "Unsupported pair: ${MODEL}:${MODALITY}"
    exit 2
    ;;
esac
