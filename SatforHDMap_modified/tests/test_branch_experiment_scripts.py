from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "branch_experiments"


def test_satfor_branch_experiment_shell_scripts_exist():
    base_expected = {
        "train_camera_only.sh",
        "train_sat_only.sh",
        "train_fusion_base.sh",
        "eval_fusion_base_drop_satellite.sh",
        "eval_fusion_base_drop_camera.sh",
        "train_fusion_distilled_camera.sh",
        "train_fusion_distilled_satellite.sh",
        "eval_fusion_distilled_drop_satellite.sh",
        "eval_fusion_distilled_drop_camera.sh",
        "train_camera_decoder_probe_camera_only.sh",
        "train_camera_decoder_probe_fusion_base.sh",
        "train_camera_decoder_probe_fusion_distilled.sh",
        "train_sat_decoder_probe_sat_only.sh",
        "train_sat_decoder_probe_fusion_base.sh",
        "train_sat_decoder_probe_fusion_distilled.sh",
    }
    expected = base_expected | {
        filename.replace(".sh", "_newsplit.sh")
        for filename in base_expected
    }

    existing = {path.name for path in SCRIPTS_DIR.glob("*.sh")}

    assert expected.issubset(existing)
