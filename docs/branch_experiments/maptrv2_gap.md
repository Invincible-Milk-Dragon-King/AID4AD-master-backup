# MapTRv2 Branch-Level Gap

## Current Decision

`MapTRv2_modified` is excluded from the first round of branch-level diagnosis and repair experiments.

## Why It Is Excluded

The current `MapTRv2_modified` tree does not contain a satellite or aerial branch that matches the `camera_only / fusion / drop_satellite / distilled` protocol used by `SatforHDMap` and `StreamMapNet`.

The codebase currently provides:

- camera-to-BEV encoding in `MapBEVPrediction_modified/MapTRv2_modified/projects/mmdet3d_plugin/maptr/modules/encoder.py`
- camera BEV processing and optional LiDAR fusion in `MapBEVPrediction_modified/MapTRv2_modified/projects/mmdet3d_plugin/maptr/modules/transformer.py`
- detector integration in `MapBEVPrediction_modified/MapTRv2_modified/projects/mmdet3d_plugin/maptr/detectors/maptrv2.py`

It does not currently provide:

- an aerial image loader in the active MapTRv2 data path
- a dedicated satellite encoder branch
- a camera-plus-satellite fusion module
- a stable branch feature API that exposes a satellite branch alongside the camera branch

## Implication For Phase 1

Phase 1 branch-level experiments should use:

- `SatforHDMap_modified`
- `MapBEVPrediction_modified/StreamMapNet_modified`

These two models now share the branch-level protocol implementation and can produce the paper metrics directly.

## Next Step If MapTRv2 Must Be Added

To bring `MapTRv2` into the same protocol, add the following in order:

1. Add aerial image loading to the active MapTRv2 dataset pipeline.
2. Introduce a satellite/aerial BEV branch parallel to the camera BEV path.
3. Fuse camera and satellite BEV features before the detection head.
4. Expose `camera_branch_feature`, `satellite_branch_feature`, and `fusion_feature`.
5. Reuse the same branch repair recipe by distilling the camera branch from a strong camera-only teacher.
