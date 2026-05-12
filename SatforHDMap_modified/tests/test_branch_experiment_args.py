import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _option_strings(script_name: str) -> list[list[str]]:
    source = (ROOT / script_name).read_text()
    tree = ast.parse(source)
    options = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        option_strings = [
            arg.value for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if option_strings:
            options.append(option_strings)

    return options


def _flatten(script_name: str) -> set[str]:
    return {option for options in _option_strings(script_name) for option in options}


@pytest.mark.parametrize("script_name", ["train.py", "test.py"])
def test_satfor_scripts_accept_branch_mode_arguments(script_name):
    options = _flatten(script_name)

    assert "--branch_mode" in options
    assert "--experiment_name" in options
    assert "--return_branch_features" in options


def test_satfor_train_accepts_distillation_arguments():
    options = _flatten("train.py")

    assert "--teacher_model_root" in options
    assert "--teacher_branch_mode" in options
    assert "--distill_feature" in options
    assert "--distill_loss" in options
    assert "--distill_weight" in options


def test_satfor_scripts_define_symmetric_branch_modes():
    source = (ROOT / "train.py").read_text()

    assert "sat_only" in source
    assert "drop_camera" in source


def test_camera_only_uses_shared_hdmapnet_model():
    source = (ROOT / "model/__init__.py").read_text()

    assert "args.branch_mode == 'camera_only'" not in source
    assert "HDMapNetCameraBaseline(" not in source
    assert "model = HDMapNet(data_conf, args" in source


def test_shared_hdmapnet_keeps_identity_k_matrices():
    source = (ROOT / "model/hdmapnet.py").read_text()

    assert "Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)" in source
    assert "Ks[:, :, :3, :3] = intrins" not in source


def test_teacher_model_uses_teacher_branch_mode_override():
    source = (ROOT / "train.py").read_text()

    assert "copy.deepcopy(args)" in source
    assert "teacher_args.branch_mode = args.teacher_branch_mode" in source


def test_distillation_loss_uses_teacher_scale_normalization():
    source = (ROOT / "train.py").read_text()

    assert "teacher_scale = teacher_feature.pow(2).mean" in source
    assert "student_output[feature_name] / teacher_scale" in source
    assert "teacher_feature = teacher_feature / teacher_scale" in source


def test_model_package_no_longer_imports_camera_baseline():
    source = (ROOT / "model/__init__.py").read_text()

    assert "camera_baseline" not in source


def test_satfor_train_accepts_decoder_probe_arguments():
    options = _flatten("train.py")

    assert "--train_decoder_only" in options
    assert "--probe_feature_source" in options
    assert "--probe_encoder_checkpoint" in options


def test_map_threshold_defaults_use_paper_ap_values():
    train_source = (ROOT / "train.py").read_text()
    test_source = (ROOT / "test.py").read_text()
    eval_source = (ROOT / "evaluate_json.py").read_text()
    vector_eval_script = (ROOT / "branch_experiments/eval_vectors_paper_ap.sh").read_text()

    assert "PAPER_MAP_THRESHOLDS = [0.2, 0.5, 1.0]" in train_source
    assert "default=PAPER_MAP_THRESHOLDS" in train_source
    assert "PAPER_MAP_THRESHOLDS = [0.2, 0.5, 1.0]" in test_source
    assert "default=PAPER_MAP_THRESHOLDS" in test_source
    assert "THRESHOLDS = [0.2, 0.5, 1.0]" in eval_source
    assert 'MAP_THRESHOLDS="${MAP_THRESHOLDS:-0.2 0.5 1.0}"' in vector_eval_script


def test_camera_only_newsplit_uses_hdmapnet_legacy_config():
    script = (ROOT / "branch_experiments/train_camera_only_newsplit.sh").read_text()

    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"' in script
    assert 'DATAROOT="${DATAROOT:-../nuScenes}"' in script
    assert 'PRIOR_MAP_ROOT="${PRIOR_MAP_ROOT:-../AID4AD_ego_referenced}"' in script
    assert 'VERSION="${VERSION:-v1.0-trainval}"' in script
    assert 'BSZ="${BSZ:-4}"' in script
    assert 'NWORKERS="${NWORKERS:-10}"' in script
    assert 'LR="${LR:-1e-3}"' in script
    assert 'MAP_THRESHOLDS="${MAP_THRESHOLDS:-0.2 0.5 1.0}"' in script
    assert "python train.py" in script
    assert "--multi_gpu false" in script
    assert "--model HDMapNet_cam_legacy" in script
    assert "--is_newsplit" in script
