# SatforHDMap Branch Runs 总结

本文档汇总当前 `SatforHDMap_modified/branch_runs/` 下已经产出的 old split / trainval 实验结果。

## 当前实验状态

| 实验目录 | 状态 | 说明 |
| --- | --- | --- |
| `camera_only/` | 已完成 | 有 `results.log`、`model_last.pt`、最终 metrics / vectors |
| `sat_only/` | 已完成 | 有 `results.log`、`model_last.pt`、最终 metrics / vectors |
| `fusion_base/` | 已完成 | 有 `results.log`、`model_last.pt`、最终 metrics / vectors |
| `fusion_distilled_satellite/` | 已完成 | 有 `results.log`、`model_last.pt`、最终 metrics / vectors |
| `fusion_distilled_camera/` | 训练完成但最终测试缺失 | 有 `EVAL[29]` 和 `model_last.pt`，但当前没有最终 `*_metrics.json` |
| `sat_decoder_probe_sat_only/` | 已完成 | 有 `results.log`、最终 metrics / vectors |
| `sat_decoder_probe_fusion_base/` | 已完成 | 有 `results.log`、最终 metrics / vectors |

当前还没有看到以下结果目录或最终 metrics：

- `fusion_base_drop_satellite/`
- `fusion_base_drop_camera/`
- `fusion_distilled_drop_satellite/`
- `fusion_distilled_drop_camera/`
- `camera_decoder_probe_camera_only/`
- `camera_decoder_probe_fusion_base/`
- `camera_decoder_probe_fusion_distilled/`
- `sat_decoder_probe_fusion_distilled/`

## 最终测试指标

以下结果来自各实验目录下的 `*_metrics.json`。

| 实验 | branch mode | mIoU | mAP | raster IoU |
| --- | --- | ---: | ---: | ---: |
| `camera_only` | `camera_only` | 0.2832 | 0.4505 | 0.2658 |
| `sat_only` | `sat_only` | 0.6026 | 0.6636 | 0.5323 |
| `fusion_base` | `fusion` | 0.6076 | 0.6706 | 0.5317 |
| `fusion_distilled_satellite` | `fusion` | 0.6076 | 0.6513 | 0.5290 |
| `sat_decoder_probe_sat_only` | `sat_only` | 0.6166 | 0.6625 | 0.5343 |
| `sat_decoder_probe_fusion_base` | `sat_only` | 0.6015 | 0.6509 | 0.5255 |

## 训练末轮验证集 IoU

以下结果来自各实验 `results.log` 中的 `EVAL[29]`。

| 实验 | `EVAL[29]` per-class IoU |
| --- | --- |
| `camera_only` | `[0.355, 0.153, 0.345]` |
| `sat_only` | `[0.607, 0.628, 0.574]` |
| `fusion_base` | `[0.587, 0.645, 0.590]` |
| `fusion_distilled_satellite` | `[0.608, 0.638, 0.579]` |
| `fusion_distilled_camera` | `[0.604, 0.669, 0.613]` |
| `sat_decoder_probe_sat_only` | `[0.620, 0.647, 0.586]` |
| `sat_decoder_probe_fusion_base` | `[0.593, 0.640, 0.569]` |

注意：`fusion_distilled_camera` 当前只有训练末轮验证集结果和 `model_last.pt`，没有最终测试 metrics。因此它暂时不应该和上表的最终测试指标直接混用。

## 初步结论

### 1. `camera_only` 明显弱于其它设置

当前最终测试结果中：

- `camera_only` mIoU = 0.2832
- `fusion_base` mIoU = 0.6076
- `sat_only` mIoU = 0.6026

这说明当前 SatforHDMap 设置下，纯 camera 分支表现显著低于 satellite-only 和 fusion。这一点会直接影响以 `camera_only` 作为 teacher 的 camera-side 蒸馏实验解释。

### 2. `fusion_base` 相比 `sat_only` 的增益很小

按最终测试指标计算：

- mIoU: `fusion_base - sat_only = +0.0050`
- mAP: `fusion_base - sat_only = +0.0070`
- raster IoU: `fusion_base - sat_only = -0.0006`

这说明当前结果里，fusion 的整体收益主要不明显，甚至 raster IoU 与 `sat_only` 基本持平。

### 3. satellite-side 蒸馏没有带来最终性能提升

按最终测试指标计算：

- mIoU: `fusion_distilled_satellite - fusion_base = -0.0000`
- mAP: `fusion_distilled_satellite - fusion_base = -0.0193`
- raster IoU: `fusion_distilled_satellite - fusion_base = -0.0027`

当前 satellite-side feature distillation 没有提升最终融合性能，mAP 还有下降。这个结果需要和 `drop_camera` 评估一起看，才能判断是否改善了 fusion 内部 satellite branch。

### 4. satellite decoder-only probe 支持“sat_only 表征更可解码”

decoder-only probe 中，两组都使用 `satellite_bevencode` 作为随机初始化 decoder，并冻结已有 encoder：

- `sat_decoder_probe_sat_only` mIoU = 0.6166
- `sat_decoder_probe_fusion_base` mIoU = 0.6015

差值：

- mIoU: `+0.0151`
- mAP: `+0.0116`
- raster IoU: `+0.0088`

这说明在当前 probe 设定下，`sat_only` 训练出来的 satellite BEV feature 比 `fusion_base` 内部的 satellite feature 更容易被单模态 decoder 解码。

## 仍需补跑的关键实验

为了完整支持 branch-level diagnosis，建议下一步补齐：

1. `eval_fusion_base_drop_satellite.sh`
2. `eval_fusion_base_drop_camera.sh`
3. `eval_fusion_distilled_drop_satellite.sh`
4. `eval_fusion_distilled_drop_camera.sh`
5. `train_camera_decoder_probe_camera_only.sh`
6. `train_camera_decoder_probe_fusion_base.sh`
7. `train_camera_decoder_probe_fusion_distilled.sh`
8. `train_sat_decoder_probe_fusion_distilled.sh`

此外，`fusion_distilled_camera/` 已经有 `model_last.pt`，但缺少最终测试 metrics。建议先补一次测试或检查为什么训练结束后的 final evaluation 没有产出 `*_metrics.json`。

## 写论文时的注意事项

- 当前 `camera_only` 结果明显偏低，不能简单拿它作为“强 camera teacher”来支撑 camera-side repair 结论。
- satellite-side 的 probe 结果更清晰：`sat_only` 表征优于 fusion 内部 satellite 表征。
- satellite-side distillation 的最终融合指标没有提升，因此论文叙事中不能只说 repair 有效；需要结合 drop-branch 结果确认是否只改善 branch，还是没有改善。
- 目前最关键缺口是 `drop_satellite/drop_camera` 两组评估，它们决定能否计算 `CameraDegenerationGap`、`RecoveryGain` 等诊断指标。
