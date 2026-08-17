# 五模型模态退化实验方案（按模型）

## 0. 实验目的（统一）

验证：Camera-LiDAR **融合训练之后**，**相机分支**的建图/检测能力是否相对纯 Camera 训练退化。

Probe 协议（五模型统一）：

| Probe | 权重来源 | 结构 | Forward |
|:---|:---|:---|:---|
| **Probe(C→C)** | Camera ckpt | 冻结 C-side encoder，只训 camera decoder | 仅 Camera |
| **Probe(F→C)** | Fusion ckpt 的 **camera 侧 encoder** | 与 C→C **同构 decoder**；冻 encoder，跳过 LiDAR/fuser | 仅 Camera |

MapTR / GeMap / ADMap / HIMap：
- F→C：`model.force_camera_only=True`（`lidar_feat=None` → 不走 ConvFuser）
- decoder-only：**只训** `transformer.decoder` + query/cls/reg；**冻结** BEV encoder / fuser / `bev_embedding` / `can_bus_mlp` 等（见 `decoder_freeze_keywords`）

HDMapNet：`--branch_mode camera_only` + 独立 `camera_bevencode`（Satfor 协议，本身正确）。

> 旧版把整个 `pts_bbox_head`（含 BEV encoder）解冻，**不符合**“只训 camera decoder”；mmdet 四模型 probe 需按新 freeze 重跑。

对每个模型，比较以下设置（指标：**mIoU / mAP**）：

| 代号 | 含义 | 操作 |
|:---|:---|:---|
| **Full-C** | Camera-only 原模型完整评测 | 用 C ckpt + C 输入测试 |
| **Full-F** | Fusion 原模型完整评测 | 用 F ckpt + C+L 输入测试 |
| **Full-L** | LiDAR-only 原模型（若有） | 用 L ckpt + L 输入测试 |
| **Probe(C→C)** | C 骨干冻结，只训 camera decoder | 加载 C ckpt，decoder-only，camera 输入 |
| **Probe(F→C)** | F 骨干冻结，去掉 LiDAR，只训 camera decoder | 加载 F ckpt，decoder-only，camera 输入 |

退化相关差值（论文主表可报）：

- \(\Delta_{\mathrm{deg}} = \mathrm{Probe}(F\rightarrow C) - \mathrm{Full\text{-}F}\)：融合模型退化到仅 Camera 的损失
- \(\Delta_{\mathrm{cam}} = \mathrm{Probe}(F\rightarrow C) - \mathrm{Probe}(C\rightarrow C)\)：融合训练对 Camera 分支的影响

---

## 1. 统一前置（每个模型容器都做）

```bash
# 宿主机进入对应容器（示例 maptr）
docker start -ai aid4ad-maptr-exp
# 或首次创建：见 modality_degradation/docker/README.md

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <model>   # hdmapnet / maptr / admap / gemap / himap

cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export CUDA_VISIBLE_DEVICES=0,1
export GPUS=2
export NPROC=2
export PORT=29566
```

### 1.1 tmux 四路双卡并行（MapTR，需 8 卡 0–7）

在**宿主机**（不要在已占用的交互 docker 里抢卡）：

```bash
cd /data2/file_swap/la_space/aid4ad/AID4AD-master/111/modality_degradation/bash
bash tmux_4x2gpu_maptr.sh
tmux attach -t maptr4
```

默认四窗：Probe(C→C)训 / Probe(F→C)训 / Full-C测 / Full-F测。  
probe 训完后在空闲窗跑 `jobs/maptr_p2_probe_c2c_test.sh` 与 `jobs/maptr_p3_probe_f2c_test.sh`。

结果建议统一落盘：

```text
111/exp_results/<model>/
  baseline_c.log
  baseline_f.log
  baseline_l.log          # 若有
  probe_c2c.log
  probe_f2c.log
  metrics_summary.md
```

---

## 2. MapTR（优先跑：有官方 C/F 权重）

**容器**：`aid4ad-maptr-exp`  
**环境**：`conda activate maptr`  
**Config**：C=`maptr_tiny_r50_24e.py`，F=`maptr_tiny_fusion_24e.py`  
**权重来源**：官方 Google Drive（无官方 L）

### Step A — 准备权重
```bash
bash 00_download_official_ckpts.sh
# 得到：
#   ../../checkpoints/maptr/maptr_tiny_r50_24e.pth
#   ../../checkpoints/maptr/maptr_tiny_fusion_24e.pth
```

### Step B — Baseline 测试（Full-C / Full-F）
```bash
# GPUS = 进程数（整数）。指定物理卡用 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
export GPUS=1

bash 20_test_baselines.sh maptr camera \
  /workspace/AID4AD-master/111/checkpoints/maptr/maptr_tiny_r50_24e.pth \
  2>&1 | tee ../../exp_results/maptr/baseline_c.log

bash 20_test_baselines.sh maptr fusion \
  /workspace/AID4AD-master/111/checkpoints/maptr/maptr_tiny_fusion_24e.pth \
  2>&1 | tee ../../exp_results/maptr/baseline_f.log
```

### Step C — Probe 训练（decoder-only；fusion 源会强制 camera-only forward）
```bash
# Probe(C→C): camera config + camera ckpt，仅 Camera
bash 30_train_probe.sh maptr camera \
  /workspace/AID4AD-master/111/checkpoints/maptr/maptr_tiny_r50_24e.pth

# Probe(F→C): 保留 fusion 结构/权重，但 model.force_camera_only=True
# （跳过 LiDAR 提取与 ConvFuser，只走相机 BEV）
bash 30_train_probe.sh maptr fusion \
  /workspace/AID4AD-master/111/checkpoints/maptr/maptr_tiny_fusion_24e.pth
```
Probe ckpt 默认在 `MapTR/branch_runs/camera_decoder_probe_maptr_*`。

> 若此前已跑过 **未** 开启 `force_camera_only` 的 fusion probe，结果无效，需用新脚本重训。

### Step D — Probe 测试（`40_test_probe` 会自动 `FORCE_CAMERA_ONLY=1`）
```bash
bash 40_test_probe.sh maptr camera <MAPTR_PROBE_C2C_CKPT> 2>&1 | tee ../../exp_results/maptr/probe_c2c.log
bash 40_test_probe.sh maptr fusion <MAPTR_PROBE_F2C_CKPT> 2>&1 | tee ../../exp_results/maptr/probe_f2c.log
```

### Step E — 填表
| Full-C | Full-F | Probe(C→C) | Probe(F→C) | Δ_deg | Δ_cam |
记录 mIoU / mAP。

**不跑**：LiDAR-only（官方无 L 建图 config/权重）。

---

## 3. GeMap（优先跑：有官方 C/F 权重）

**容器**：`aid4ad-gemap-exp`  
**环境**：`conda activate gemap`  
**模态公平配对（同一 simple objective）**：  
- C=`gemap_simple_r50_110ep.py` + `gemap_simple_r50_110ep.pth`（官方 ~62.7）  
- F=`gemap_simple_r50_sec_110ep.py` + `gemap_simple_r50_sec_110ep.pth`（官方 ~66.5）  

> 勿用 `gemap_full_*`（~69.4）与 simple-fusion 做 Δ；full 仅作可选 SOTA camera 行。

### Step A — 准备权重
```bash
# 本地下载后放到（服务器常无法访问 Google Drive）：
#   checkpoints/gemap/gemap_simple_r50_110ep.pth
#   checkpoints/gemap/gemap_simple_r50_sec_110ep.pth
# 链接见 00_download_official_ckpts.sh
```

### Step B — Baseline 测试
```bash
bash 20_test_baselines.sh gemap camera \
  /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_110ep.pth \
  2>&1 | tee ../../exp_results/gemap/baseline_c_simple.log

bash 20_test_baselines.sh gemap fusion \
  /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_sec_110ep.pth \
  2>&1 | tee ../../exp_results/gemap/baseline_f.log

# 可选：SOTA camera（不参与 Δ）
# bash 20_test_baselines.sh gemap full_camera \
#   .../gemap_full_r50_110ep.pth | tee ../../exp_results/gemap/baseline_c_full.log
```

### Step C — Probe 训练
```bash
# 必须用 simple 权重；目录里若只有 gemap_full_* 会错配（日志会出现大量 missing/unexpected keys）。
# Probe 默认只训 24 epoch（覆盖 config 里的 110），与 MapTR/ADMap/HIMap 对齐。
export PROBE_EPOCHS=24
export PORT=29572   # 若报 Address already in use，换端口

bash 30_train_probe.sh gemap camera \
  /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_110ep.pth
bash 30_train_probe.sh gemap fusion \
  /workspace/AID4AD-master/111/checkpoints/gemap/gemap_simple_r50_sec_110ep.pth
```

### Step D — Probe 测试
```bash
bash 40_test_probe.sh gemap camera <GEMAP_PROBE_C2C_CKPT> 2>&1 | tee ../../exp_results/gemap/probe_c2c.log
bash 40_test_probe.sh gemap fusion <GEMAP_PROBE_F2C_CKPT> 2>&1 | tee ../../exp_results/gemap/probe_f2c.log
```

### Step E — 填表
同 MapTR 四列 + 两个 Δ（仅 simple-C vs simple-F）。

---

## 4. HDMapNet（需自训 C/F/L）

**容器**：`aid4ad-hdmapnet-exp`  
**环境**：`conda activate hdmapnet`  
**入口**：`train.py --model HDMapNet_{cam,lidar,fusion}`  
**权重**：无官方，必须自训  
**Probe 风格**：已移植到 `111/HDMapNet`（`decoder_probe.py` + `camera_bevencode` + `--branch_mode camera_only`）。  
注意：此处 fusion = **Camera+LiDAR**（不是 Satfor 的 Camera+卫星图）。

### Step A — 训练 baseline
```bash
bash 10_train_baselines.sh hdmapnet camera   # -> HDMapNet/runs/hdmapnet_cam/
bash 10_train_baselines.sh hdmapnet fusion   # -> .../hdmapnet_fusion/
bash 10_train_baselines.sh hdmapnet lidar    # -> .../hdmapnet_lidar/
```
取每个目录最后一轮 `model*.pt` 作为 ckpt。

### Step B — Baseline 测试（Full-C / Full-F / Full-L）
```bash
bash 20_test_baselines.sh hdmapnet camera <HDMAPNET_CAM_CKPT>   2>&1 | tee ../../exp_results/hdmapnet/baseline_c.log
bash 20_test_baselines.sh hdmapnet fusion <HDMAPNET_FUSION_CKPT> 2>&1 | tee ../../exp_results/hdmapnet/baseline_f.log
bash 20_test_baselines.sh hdmapnet lidar  <HDMAPNET_LIDAR_CKPT>  2>&1 | tee ../../exp_results/hdmapnet/baseline_l.log
```
（导出 JSON + `evaluate_json.py`，得到 IoU / Chamfer AP）

### Step C — Probe 训练（仅 C 与 F 源）
```bash
bash 30_train_probe.sh hdmapnet camera <HDMAPNET_CAM_CKPT>
bash 30_train_probe.sh hdmapnet fusion <HDMAPNET_FUSION_CKPT>
```
LiDAR 源不做 Camera-decoder probe（与“C 模态退化”目的不一致）；Full-L 只作基线对照。

### Step D — Probe 测试
```bash
bash 40_test_probe.sh hdmapnet camera <PROBE_C2C_CKPT> 2>&1 | tee ../../exp_results/hdmapnet/probe_c2c.log
bash 40_test_probe.sh hdmapnet fusion <PROBE_F2C_CKPT> 2>&1 | tee ../../exp_results/hdmapnet/probe_f2c.log
```

### Step E — 填表
| Full-C | Full-F | Full-L | Probe(C→C) | Probe(F→C) | Δ_deg | Δ_cam |

---

## 5. ADMap（需自训 C/F）

**容器**：`aid4ad-admap-exp`  
**环境**：`conda activate admap`  
**Config**：`configs/ADMap_cam_24e.py` / `configs/ADMap_fusion_24e.py`  
**权重**：无官方，自训  
**Probe**：`--decoder-only` 已默认启用

### Step A — 训练 baseline
```bash
bash 10_train_baselines.sh admap camera   # work_dirs/admap_cam_24e
bash 10_train_baselines.sh admap fusion   # work_dirs/admap_fusion_24e
```

### Step B — Baseline 测试
```bash
bash 20_test_baselines.sh admap camera <ADMAP_CAM_CKPT>   2>&1 | tee ../../exp_results/admap/baseline_c.log
bash 20_test_baselines.sh admap fusion <ADMAP_FUSION_CKPT> 2>&1 | tee ../../exp_results/admap/baseline_f.log
```

### Step C — Probe 训练
```bash
bash 30_train_probe.sh admap camera <ADMAP_CAM_CKPT>
bash 30_train_probe.sh admap fusion <ADMAP_FUSION_CKPT>
```

### Step D — Probe 测试
```bash
bash 40_test_probe.sh admap camera <ADMAP_PROBE_C2C_CKPT> 2>&1 | tee ../../exp_results/admap/probe_c2c.log
bash 40_test_probe.sh admap fusion <ADMAP_PROBE_F2C_CKPT> 2>&1 | tee ../../exp_results/admap/probe_f2c.log
```

### Step E — 填表
同 MapTR（无 L 列）。

---

## 6. HIMap（需自训 C/F/L）

**容器**：`aid4ad-himap-exp`  
**环境**：`conda activate himap`  
**Config**：
- C=`himap_tiny_cam_24e.py`
- F=`himap_tiny_fusion_24e.py`
- L=`himap_tiny_lidar_24e.py`
**权重**：无官方，自训  
**Probe**：decoder-only 默认开启；L 只作 Full-L 对照

### Step A — 训练 baseline
```bash
bash 10_train_baselines.sh himap camera
bash 10_train_baselines.sh himap fusion
bash 10_train_baselines.sh himap lidar
```

### Step B — Baseline 测试
```bash
bash 20_test_baselines.sh himap camera <HIMAP_CAM_CKPT>   2>&1 | tee ../../exp_results/himap/baseline_c.log
bash 20_test_baselines.sh himap fusion <HIMAP_FUSION_CKPT> 2>&1 | tee ../../exp_results/himap/baseline_f.log
bash 20_test_baselines.sh himap lidar  <HIMAP_LIDAR_CKPT>  2>&1 | tee ../../exp_results/himap/baseline_l.log
```

### Step C — Probe 训练（仅 C/F）
```bash
bash 30_train_probe.sh himap camera <HIMAP_CAM_CKPT>
bash 30_train_probe.sh himap fusion <HIMAP_FUSION_CKPT>
```

### Step D — Probe 测试
```bash
bash 40_test_probe.sh himap camera <HIMAP_PROBE_C2C_CKPT> 2>&1 | tee ../../exp_results/himap/probe_c2c.log
bash 40_test_probe.sh himap fusion <HIMAP_PROBE_F2C_CKPT> 2>&1 | tee ../../exp_results/himap/probe_f2c.log
```

### Step E — 填表
| Full-C | Full-F | Full-L | Probe(C→C) | Probe(F→C) | Δ_deg | Δ_cam |

---

## 7. 推荐执行顺序（省时间、先出结果）

1. **MapTR**（官方权重，最快验证整条链路）  
2. **GeMap**（官方权重，第二条完整链路）  
3. **HDMapNet**（栅格基线，自训 C/F/L）  
4. **ADMap**（自训 C/F）  
5. **HIMap**（自训 C/F/L，最重）

每个模型内部固定顺序：

```text
Baseline 测试（或先训练再测） → Probe 训练 → Probe 测试 → 填表
```

---

## 8. 论文主表目标形态（汇总）

每个格子填 `mIoU / mAP`：

| Method | Full-C | Full-F | Full-L | Probe(C→C) | Probe(F→C) | Δ_deg | Δ_cam |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HDMapNet | | | | | | | |
| MapTR | | | — | | | | |
| ADMap | | | — | | | | |
| GeMap | | | — | | | | |
| HIMap | | | | | | | |

---

## 9. 每个模型跑完后的自检清单

- [ ] Baseline log 中有可复现的 mIoU / mAP（或 IoU / AP）
- [ ] Probe 日志出现 `Decoder-only mode enabled`（mmdet 系）或 `train_decoder_only`（HDMapNet/Satfor）
- [ ] Trainable params 远小于全量参数
- [ ] Probe(F→C) 推理时为 **camera-only 输入**（无 LiDAR）
- [ ] 结果已写入 `exp_results/<model>/`

---

## 10. 已知注意点

1. **Fusion probe 必须 camera-only forward**：训练（`30_train_probe`）与测试（`40_test_probe`）都已强制；Full-F baseline 测试不要设 `FORCE_CAMERA_ONLY`。  
2. **MapTR/ADMap/GeMap/HIMap**：`--decoder-only` + `decoder_freeze_keywords` 冻 BEV encoder/fuser，只训 decoder；fusion 源额外 `force_camera_only`。  
3. **HDMapNet**：Satfor 式 `train_decoder_only`；fusion probe 用 `HDMapNet_fusion` 结构加载 fusion ckpt，只训/评 `camera_bevencode`（无需改 freeze）。  
4. **指标口径**：矢量系主报 mAP；HDMapNet 主报 IoU + Chamfer AP。  
5. **GPU**：`CUDA_VISIBLE_DEVICES=0,1` + `GPUS=2`；HDMapNet probe 当前为单进程。
