# 按模型划分的实验命令清单

以下命令默认在：

```bash
cd /data2/file_swap/la_space/aid4ad/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/data1/data/nuscenes
```

---

## 1) HDMapNet

### 训练 baseline
```bash
bash 10_train_baselines.sh hdmapnet camera
bash 10_train_baselines.sh hdmapnet fusion
bash 10_train_baselines.sh hdmapnet lidar
```

### 测试 baseline
```bash
bash 20_test_baselines.sh hdmapnet camera <HDMAPNET_CAM_CKPT>
bash 20_test_baselines.sh hdmapnet fusion <HDMAPNET_FUSION_CKPT>
bash 20_test_baselines.sh hdmapnet lidar <HDMAPNET_LIDAR_CKPT>
```

### 训练 probe
```bash
bash 30_train_probe.sh hdmapnet camera <HDMAPNET_CAM_CKPT>
bash 30_train_probe.sh hdmapnet fusion <HDMAPNET_FUSION_CKPT>
# LiDAR-branch probes (protocol 1; Full-L must be trained with new isomorphic branch)
bash 35_train_lidar_probe.sh hdmapnet lidar <HDMAPNET_LIDAR_CKPT>
bash 35_train_lidar_probe.sh hdmapnet fusion_lidar <HDMAPNET_FUSION_CKPT>
```

### 测试 probe
```bash
bash 40_test_probe.sh hdmapnet camera <HDMAPNET_PROBE_FROM_CAM_CKPT>
bash 40_test_probe.sh hdmapnet fusion <HDMAPNET_PROBE_FROM_FUSION_CKPT>
bash 45_test_lidar_probe.sh hdmapnet lidar <HDMAPNET_PROBE_L2L_CKPT>
bash 45_test_lidar_probe.sh hdmapnet fusion_lidar <HDMAPNET_PROBE_F2L_CKPT>
```

---

## 2) MapTR

### 下载官方权重（一次）
```bash
bash 00_download_official_ckpts.sh
```

### （可选）训练 baseline
```bash
bash 10_train_baselines.sh maptr camera
bash 10_train_baselines.sh maptr fusion
```

### 测试 baseline（推荐先跑官方）
```bash
bash 20_test_baselines.sh maptr camera /data2/file_swap/la_space/aid4ad/AID4AD-master/111/checkpoints/maptr/maptr_tiny_r50_24e_camera.pth
bash 20_test_baselines.sh maptr fusion /data2/file_swap/la_space/aid4ad/AID4AD-master/111/checkpoints/maptr/maptr_tiny_fusion_24e.pth
```

### 训练 probe（已自动启用 decoder-only）
```bash
bash 30_train_probe.sh maptr camera /data2/file_swap/la_space/aid4ad/AID4AD-master/111/checkpoints/maptr/maptr_tiny_r50_24e_camera.pth
bash 30_train_probe.sh maptr fusion /data2/file_swap/la_space/aid4ad/AID4AD-master/111/checkpoints/maptr/maptr_tiny_fusion_24e.pth
```

### 测试 probe
```bash
bash 40_test_probe.sh maptr camera <MAPTR_PROBE_FROM_CAM_CKPT>
bash 40_test_probe.sh maptr fusion <MAPTR_PROBE_FROM_FUSION_CKPT>
```

---

## 3) ADMap

路径约定（与 MapTR 一致）：
- **权重 / mmcv 日志**：`ADMap-main/work_dirs/admap_{cam,fusion}_24e/`
- **控制台日志（tee）**：`111/exp_results/admap/*.log`

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate admap
cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export CUDA_VISIBLE_DEVICES=4,5
export GPUS=2 NPROC=2 PORT=29563
mkdir -p ../../exp_results/admap

# baseline ckpt（自训已完成）
ADMAP_CAM_CKPT=/workspace/AID4AD-master/111/ADMap-main/work_dirs/admap_cam_24e/epoch_24.pth
ADMAP_FUSION_CKPT=/workspace/AID4AD-master/111/ADMap-main/work_dirs/admap_fusion_24e/epoch_24.pth
# probe ckpt（30_train_probe 训完后生成；logdir=branch_runs/camera_decoder_probe_admap_{camera,fusion}）
ADMAP_PROBE_C_CKPT=/workspace/AID4AD-master/111/ADMap-main/branch_runs/camera_decoder_probe_admap_camera/epoch_24.pth
ADMAP_PROBE_F_CKPT=/workspace/AID4AD-master/111/ADMap-main/branch_runs/camera_decoder_probe_admap_fusion/epoch_24.pth

# 1) 自训 Full-C / Full-F（已完成可跳过）
# bash 10_train_baselines.sh admap camera 2>&1 | tee ../../exp_results/admap/train_c.log
# bash 10_train_baselines.sh admap fusion 2>&1 | tee ../../exp_results/admap/train_f.log

# 2) 测 baseline
bash 20_test_baselines.sh admap camera "$ADMAP_CAM_CKPT" 2>&1 | tee ../../exp_results/admap/baseline_c.log
bash 20_test_baselines.sh admap fusion "$ADMAP_FUSION_CKPT" 2>&1 | tee ../../exp_results/admap/baseline_f.log

# 3) Probe 训 + 测
# ADMap defaults: camera GLOBAL_BATCH=8, fusion GLOBAL_BATCH=4（避免首 iter CUDA illegal access）
bash 30_train_probe.sh admap camera "$ADMAP_CAM_CKPT" 2>&1 | tee ../../exp_results/admap/probe_c2c_train.log
bash 30_train_probe.sh admap fusion "$ADMAP_FUSION_CKPT" 2>&1 | tee ../../exp_results/admap/probe_f2c_train.log
bash 40_test_probe.sh admap camera "$ADMAP_PROBE_C_CKPT" 2>&1 | tee ../../exp_results/admap/probe_c2c.log
bash 40_test_probe.sh admap fusion "$ADMAP_PROBE_F_CKPT" 2>&1 | tee ../../exp_results/admap/probe_f2c.log
```

---

## 4) GeMap

### 下载官方权重（一次）
```bash
bash 00_download_official_ckpts.sh
```

### （可选）训练 baseline
```bash
bash 10_train_baselines.sh gemap camera
bash 10_train_baselines.sh gemap fusion
```

### 测试 baseline（推荐先跑官方）
```bash
bash 20_test_baselines.sh gemap camera /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_110ep.pth
bash 20_test_baselines.sh gemap fusion /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_sec_110ep.pth
# optional SOTA camera (not for Δ):
# bash 20_test_baselines.sh gemap full_camera .../gemap_full_r50_110ep.pth
```

### 训练 probe（已自动启用 decoder-only）
```bash
bash 30_train_probe.sh gemap camera /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_110ep.pth
bash 30_train_probe.sh gemap fusion /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_sec_110ep.pth
```

### 测试 probe
```bash
bash 40_test_probe.sh gemap camera <GEMAP_PROBE_FROM_CAM_CKPT>
bash 40_test_probe.sh gemap fusion <GEMAP_PROBE_FROM_FUSION_CKPT>
```

---

## 5) HIMap

路径约定（与 ADMap / MapTR 一致）：
- **权重 / mmcv 日志**：`HIMap/work_dirs/himap_tiny_{cam,fusion,lidar}_24e/`
- **Probe 权重**：`HIMap/branch_runs/camera_decoder_probe_himap_{camera,fusion}/` 与 `lidar_decoder_probe_himap_{lidar,fusion_lidar}/`
- **控制台日志（tee）**：`111/exp_results/himap/*.log`

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate himap
cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export CUDA_VISIBLE_DEVICES=4,5
export GPUS=2 NPROC=2 PORT=29672
mkdir -p ../../exp_results/himap

# baseline ckpt（自训已完成；fusion 重训后 epoch24 mAP≈0.71）
HIMAP_CAM_CKPT=/workspace/AID4AD-master/111/HIMap/work_dirs/himap_tiny_cam_24e/epoch_24.pth
HIMAP_FUSION_CKPT=/workspace/AID4AD-master/111/HIMap/work_dirs/himap_tiny_fusion_24e/epoch_24.pth
# probe ckpt（30_train_probe 训完后生成）
HIMAP_PROBE_C_CKPT=/workspace/AID4AD-master/111/HIMap/branch_runs/camera_decoder_probe_himap_camera/epoch_24.pth
HIMAP_PROBE_F_CKPT=/workspace/AID4AD-master/111/HIMap/branch_runs/camera_decoder_probe_himap_fusion/epoch_24.pth

# 1) 自训 Full-C / Full-F（已完成可跳过）
# bash 10_train_baselines.sh himap camera 2>&1 | tee ../../exp_results/himap/train_c.log
# bash 10_train_baselines.sh himap fusion 2>&1 | tee ../../exp_results/himap/train_f.log

# 2) 测 baseline
bash 20_test_baselines.sh himap camera "$HIMAP_CAM_CKPT" 2>&1 | tee ../../exp_results/himap/baseline_c.log
bash 20_test_baselines.sh himap fusion "$HIMAP_FUSION_CKPT" 2>&1 | tee ../../exp_results/himap/baseline_f.log

# 3) Probe 训 + 测（decoder-only 默认开启；fusion probe 会 FORCE_CAMERA_ONLY）
bash 30_train_probe.sh himap camera "$HIMAP_CAM_CKPT" 2>&1 | tee ../../exp_results/himap/probe_c2c_train.log
bash 30_train_probe.sh himap fusion "$HIMAP_FUSION_CKPT" 2>&1 | tee ../../exp_results/himap/probe_f2c_train.log
bash 40_test_probe.sh himap camera "$HIMAP_PROBE_C_CKPT" 2>&1 | tee ../../exp_results/himap/probe_c2c.log
bash 40_test_probe.sh himap fusion "$HIMAP_PROBE_F_CKPT" 2>&1 | tee ../../exp_results/himap/probe_f2c.log

# LiDAR-branch probes (protocol 1)
bash 10_train_baselines.sh himap lidar 2>&1 | tee ../../exp_results/himap/train_l.log
bash 20_test_baselines.sh himap lidar "$HIMAP_LIDAR_CKPT" 2>&1 | tee ../../exp_results/himap/baseline_l.log
bash 35_train_lidar_probe.sh himap lidar "$HIMAP_LIDAR_CKPT" 2>&1 | tee ../../exp_results/himap/probe_l2l_train.log
bash 35_train_lidar_probe.sh himap fusion_lidar "$HIMAP_FUSION_CKPT" 2>&1 | tee ../../exp_results/himap/probe_f2l_train.log
bash 45_test_lidar_probe.sh himap lidar "$HIMAP_PROBE_L2L_CKPT" 2>&1 | tee ../../exp_results/himap/probe_l2l.log
bash 45_test_lidar_probe.sh himap fusion_lidar "$HIMAP_PROBE_F2L_CKPT" 2>&1 | tee ../../exp_results/himap/probe_f2l.log
```
