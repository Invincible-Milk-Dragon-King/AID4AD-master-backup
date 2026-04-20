# Camera-Only HDMapNet Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `camera_only` use the shared `HDMapNet` implementation instead of `HDMapNetCameraBaseline`, while preserving the current experiment shell and branch-feature compatibility.

**Architecture:** Route `HDMapNet_cam` through `model/hdmapnet.py` for all branch modes, keep `camera_only` behavior driven by `branch_mode="camera_only"` inside `HDMapNet.forward()`, and lock the change down with regression tests that assert shared-model selection and identity-`Ks` behavior.

**Tech Stack:** Python, PyTorch, pytest, existing branch experiment framework

---

### Task 1: Add failing regression tests for camera-only model selection

**Files:**
- Modify: `tests/test_branch_experiment_args.py`
- Test: `tests/test_branch_experiment_args.py`

- [ ] **Step 1: Write the failing test**

```python
def test_camera_only_uses_shared_hdmapnet_model():
    source = (ROOT / "model/__init__.py").read_text()

    assert "args.branch_mode == 'camera_only'" not in source
    assert "HDMapNetCameraBaseline(" not in source
    assert "model = HDMapNet(data_conf, args" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_branch_experiment_args.py -k "camera_only_uses_shared_hdmapnet_model" -v`
Expected: FAIL because `model/__init__.py` still contains the `camera_only` special-case baseline selection.

- [ ] **Step 3: Add a second failing test for identity-Ks behavior**

```python
def test_shared_hdmapnet_keeps_identity_k_matrices():
    source = (ROOT / "model/hdmapnet.py").read_text()

    assert "Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)" in source
    assert "Ks[:, :, :3, :3] = intrins" not in source
```

- [ ] **Step 4: Run test to verify it passes before implementation**

Run: `pytest tests/test_branch_experiment_args.py -k "shared_hdmapnet_keeps_identity_k_matrices" -v`
Expected: PASS because the shared `HDMapNet` implementation already uses identity `Ks`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_branch_experiment_args.py
git commit -m "test: lock camera-only HDMapNet selection"
```

### Task 2: Route camera-only through shared HDMapNet

**Files:**
- Modify: `model/__init__.py`
- Test: `tests/test_branch_experiment_args.py`

- [ ] **Step 1: Write the failing test for selection change**

```python
def test_camera_only_uses_shared_hdmapnet_model():
    source = (ROOT / "model/__init__.py").read_text()

    assert "args.branch_mode == 'camera_only'" not in source
    assert "HDMapNetCameraBaseline(" not in source
    assert "model = HDMapNet(data_conf, args" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_branch_experiment_args.py -k "camera_only_uses_shared_hdmapnet_model" -v`
Expected: FAIL because the baseline branch still exists.

- [ ] **Step 3: Write minimal implementation**

```python
from project_paths import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from .hdmapnet import HDMapNet
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar


def get_model(method, data_conf, args, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    else:
        raise NotImplementedError

    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_branch_experiment_args.py -k "camera_only_uses_shared_hdmapnet_model or shared_hdmapnet_keeps_identity_k_matrices" -v`
Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add model/__init__.py tests/test_branch_experiment_args.py
git commit -m "fix: align camera-only path with shared HDMapNet"
```

### Task 3: Update existing assertions and run focused verification

**Files:**
- Modify: `tests/test_branch_experiment_args.py`
- Test: `tests/test_branch_experiment_args.py`

- [ ] **Step 1: Replace outdated baseline-specific assertions**

```python
def test_camera_only_uses_shared_hdmapnet_model():
    source = (ROOT / "model/__init__.py").read_text()

    assert "HDMapNetCameraBaseline" not in source
    assert "args.branch_mode == 'camera_only'" not in source
    assert "model = HDMapNet(data_conf, args" in source


def test_shared_hdmapnet_keeps_identity_k_matrices():
    source = (ROOT / "model/hdmapnet.py").read_text()

    assert "Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)" in source
    assert "Ks[:, :, :3, :3] = intrins" not in source
```

- [ ] **Step 2: Run the focused test file**

Run: `pytest tests/test_branch_experiment_args.py -v`
Expected: PASS, including branch-argument coverage, teacher override coverage, decoder-probe argument coverage, and the new camera-only shared-model regression checks.

- [ ] **Step 3: Run lints for touched files**

Run: use the workspace lint reader on:
- `model/__init__.py`
- `tests/test_branch_experiment_args.py`

Expected: no new diagnostics introduced by this change.

- [ ] **Step 4: Commit**

```bash
git add model/__init__.py tests/test_branch_experiment_args.py
git commit -m "test: refresh camera-only regression coverage"
```
