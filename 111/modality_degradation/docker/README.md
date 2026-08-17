# Docker + Conda（按模型容器化）

目标：一个基础镜像，五个模型独立容器，容器内独立 conda 环境，分别执行各模型实验。

---

## 1) 构建基础镜像（一次）
## 1) 构建基础镜像（一次）

```bash
cd /data2/file_swap/la_space/aid4ad/AID4AD-master/111/modality_degradation/docker
bash build_image.sh
```

默认镜像名：`aid4ad-modality:cu118`

---

## 2) 为某个模型创建/进入容器

```bash
bash create_model_container.sh <model>
```

`<model>` 支持：
- `hdmapnet`
- `maptr`
- `admap`
- `gemap`
- `himap`

容器命名：
- `aid4ad-hdmapnet-exp`
- `aid4ad-maptr-exp`
- `aid4ad-admap-exp`
- `aid4ad-gemap-exp`
- `aid4ad-himap-exp`

---

## 3) 在容器内创建该模型 conda 环境（一次）

进入容器后执行：

```bash
cd /workspace/AID4AD-master/111
bash modality_degradation/docker/setup_model_conda_env.sh <model>
```

环境名与模型同名（如 `maptr`、`gemap`）。

---

## 4) 在模型容器内执行实验

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <model>

cd /workspace/AID4AD-master/111/modality_degradation/bash
export NUSCENES_ROOT=/workspace/datasets/nuscenes
export GPUS=6,7
export NPROC=2
export PORT=29566
```

然后按：

`model_commands_by_model.md`

中的对应模型命令执行。

---

## 5) 宿主机路径映射（默认）

| 宿主机 | 容器内 |
|---|---|
| `/data2/file_swap/la_space/aid4ad/AID4AD-master` | `/workspace/AID4AD-master` |
| `/data1/data/nuscenes` | `/workspace/datasets/nuscenes` |
| `AID4AD-master/docker_cache/conda_<model>` | `/opt/conda/envs/<model>` |

