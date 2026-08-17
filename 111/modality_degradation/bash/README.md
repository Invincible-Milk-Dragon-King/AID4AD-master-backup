# Modality Degradation Bash Scripts

All scripts here are prepared for:

- Models: `hdmapnet`, `maptr`, `admap`, `gemap`, `himap`
- Dataset root: `NUSCENES_ROOT=/data1/data/nuscenes`
- Workflow: official checkpoint download -> baseline train/test -> probe train/test

## 1) Download official checkpoints (MapTR, GeMap)

```bash
bash 00_download_official_ckpts.sh
```

Downloaded files:

- `checkpoints/maptr/maptr_tiny_r50_24e.pth`
- `checkpoints/maptr/maptr_tiny_fusion_24e.pth`
- `checkpoints/gemap/gemap_simple_r50_110ep.pth`（模态公平 Camera）
- `checkpoints/gemap/gemap_simple_r50_sec_110ep.pth`（Fusion）
- `checkpoints/gemap/gemap_full_r50_110ep.pth`（可选 SOTA Camera，不参与 Δ）

## 2) Train baseline

```bash
# generic
bash 10_train_baselines.sh <model> <modality>
```

Supported modality:

- `hdmapnet`: `camera`, `fusion`, `lidar`
- `maptr`: `camera`, `fusion`
- `admap`: `camera`, `fusion`
- `gemap`: `camera`, `fusion`
- `himap`: `camera`, `fusion`, `lidar`

## 3) Test baseline

```bash
bash 20_test_baselines.sh <model> <modality> <checkpoint_path>
```

## 4) Train probe

```bash
bash 30_train_probe.sh <model> <source> <checkpoint_path>
```

- `<source>` is `camera` or `fusion`.
- For `maptr/admap/gemap/himap`, decoder-only is enabled by default:
  backbone/encoder/transformer are frozen by strict allowlist matching and
  only map head/decoder-related params are trainable.

## 5) Test probe

```bash
bash 40_test_probe.sh <model> <source> <probe_checkpoint_path>
```

## Runtime knobs

Set these before running:

```bash
export NUSCENES_ROOT=/data1/data/nuscenes
export GPUS=6,7
export NPROC=2
export PORT=29566
```
