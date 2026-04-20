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


def test_camera_only_uses_dedicated_camera_baseline_model():
    source = (ROOT / "model/__init__.py").read_text()

    assert "HDMapNetCameraBaseline" in source
    assert "args.branch_mode == 'camera_only'" in source


def test_teacher_model_uses_teacher_branch_mode_override():
    source = (ROOT / "train.py").read_text()

    assert "copy.deepcopy(args)" in source
    assert "teacher_args.branch_mode = args.teacher_branch_mode" in source


def test_camera_baseline_uses_current_ipm_signature():
    source = (ROOT / "model/camera_baseline.py").read_text()

    assert "from .ipm_net import IPMNet" not in source
    assert "IPM(args," in source


def test_satfor_train_accepts_decoder_probe_arguments():
    options = _flatten("train.py")

    assert "--train_decoder_only" in options
    assert "--probe_feature_source" in options
    assert "--probe_encoder_checkpoint" in options
