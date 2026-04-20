# Camera-Only HDMapNet Alignment Design

## Goal
Align the `camera_only` experiment in `SatforHDMap_modified` with the upstream `HDMapNet` camera-to-BEV path while keeping the current experiment shell unchanged. The training entrypoint, dataset selection, batch settings, log directory behavior, and loss configuration entrypoints must remain as they are today.

## Current Problem
The current `camera_only` path does not run the same model implementation as upstream `HDMapNet`.

- `branch_experiments/train_camera_only.sh` launches `train.py` with `--branch_mode camera_only`.
- In `model/__init__.py`, that branch mode currently selects `HDMapNetCameraBaseline`.
- `HDMapNetCameraBaseline` is a forked camera-only implementation rather than the shared `HDMapNet` model used upstream.
- A key behavioral difference is in `get_Ks_RTs_and_post_RTs()`: the baseline writes `intrins` into `Ks`, while upstream `HDMapNet` uses identity `Ks`.

The result is that `camera_only` is not a faithful reproduction of upstream `HDMapNet`, and the existing training log shows evaluation IoU staying near zero across epochs.

## Requirements
The implementation must satisfy all of the following:

1. Keep `branch_experiments/train_camera_only.sh` unchanged.
2. Keep current experiment arguments and defaults for batch size, dataset paths, log directories, and loss configuration entrypoints unchanged.
3. Change `camera_only` model selection so it no longer instantiates `HDMapNetCameraBaseline`.
4. Make the `camera_only` image-to-BEV path match upstream `HDMapNet-main/model/hdmapnet.py` for:
   - `CamEncode`
   - `ViewTransformation`
   - `IPM`
   - `Upsample`
   - `BevEncode`
5. Restore upstream `Ks` behavior for the camera path by using identity `Ks` instead of copying `intrins` into `Ks`.
6. Preserve the current training framework interfaces, including:
   - `branch_mode`
   - `return_branch_features`
   - branch feature dictionary fields used by distillation and decoder probe code
7. Avoid regressions in `sat_only`, `fusion`, `drop_satellite`, `drop_camera`, teacher, and decoder-probe flows.

## Recommended Approach
Use the existing shared `HDMapNet` implementation in `model/hdmapnet.py` for `camera_only`, instead of maintaining a separate camera-only model class.

This is the smallest change that satisfies the reproduction goal:

- It removes the forked model selection path.
- It reuses the same camera encoder, view transform, IPM, upsampling, and BEV decoder stack already used in the shared model.
- It keeps branch-aware return values and downstream compatibility because `model/hdmapnet.py` already supports `branch_mode` and `return_branch_features`.

## Design
### Model Selection
Update `model/__init__.py` so `HDMapNet_cam` always instantiates `HDMapNet`, regardless of whether `args.branch_mode == "camera_only"`.

`camera_only` behavior should then be driven by `HDMapNet.forward(..., branch_mode="camera_only")`, not by a separate class.

### Camera Path Alignment
Do not change the camera branch stages inside `model/hdmapnet.py` beyond what is needed for upstream parity. The existing shared implementation already matches the upstream stage ordering:

1. `CamEncode`
2. `ViewTransformation`
3. `IPM`
4. `Upsample`
5. `camera_bevencode`

The alignment-sensitive item is `get_Ks_RTs_and_post_RTs()`. This function must continue to behave like upstream `HDMapNet`, where `Ks` is initialized as identity matrices and `post_RTs` remains `None`.

### Compatibility Contract
The implementation must keep these observable behaviors stable for the rest of the repo:

- `train.py` and `test.py` still pass `branch_mode` and `return_branch_features`.
- `branch_mode="camera_only"` still returns:
  - `semantic`
  - `embedding`
  - `direction`
  - `branch_mode`
  - `camera_branch_feature`
  - `satellite_branch_feature`
  - `fusion_feature`
- decoder-probe helpers still expose the camera encoder stack expected by probe code.
- teacher/distillation setup still works because the teacher model also comes through `get_model(...)`.

### Test Strategy
Add a regression test before the production change.

The test coverage should lock in:

1. `camera_only` no longer selects `HDMapNetCameraBaseline` in `model/__init__.py`.
2. `camera_only` uses the shared `HDMapNet` path.
3. The shared `HDMapNet` camera path keeps identity `Ks` behavior instead of writing `intrins` into `Ks`.

These tests should be lightweight source- or behavior-level checks that fit the current test style in `tests/test_branch_experiment_args.py`.

## Files Expected To Change
- Modify `model/__init__.py`
- Modify `tests/test_branch_experiment_args.py`

Possible but not required:
- Keep `model/camera_baseline.py` untouched if it is no longer referenced by active model selection.
- Remove the import from `model/__init__.py` if it becomes unused.

## Non-Goals
- Do not change `train_camera_only.sh`.
- Do not change dataset loading or split behavior.
- Do not tune learning rate, batch size, weight decay, or other training defaults.
- Do not refactor fusion or satellite-only architectures beyond what is required for `camera_only` selection compatibility.
- Do not redesign losses or evaluation logic.

## Risks And Mitigations
### Risk: decoder probe or distillation logic depended on the baseline class
Mitigation: preserve `return_branch_features` outputs and confirm probe helper accessors still exist on the shared `HDMapNet` class.

### Risk: tests overfit to string matching
Mitigation: keep the regression tests focused on stable selection behavior and the identity-`Ks` contract, not incidental formatting.

### Risk: removing the baseline path affects future experiments that still import it manually
Mitigation: do not delete `model/camera_baseline.py` in this change unless follow-up cleanup is explicitly requested.

## Validation
Implementation is complete when:

- `camera_only` trains through the shared `HDMapNet` model path.
- no test relies on `HDMapNetCameraBaseline` as the selected `camera_only` model anymore.
- regression tests pass.
- existing branch argument tests still pass.
