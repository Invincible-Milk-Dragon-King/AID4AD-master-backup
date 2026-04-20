import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from branch_experiments.protocol import (  # noqa: E402
    BRANCH_MODE_CAMERA_ONLY,
    BRANCH_MODE_DROP_CAMERA,
    BRANCH_MODE_DROP_SATELLITE,
    BRANCH_MODE_FUSION,
    BRANCH_MODE_SAT_ONLY,
    BranchMetricsInput,
    compute_branch_metrics,
    uses_camera_branch,
    uses_fusion_head,
    uses_satellite_branch,
)
from branch_experiments.report import summarize_branch_metrics, write_metrics_csv  # noqa: E402


def test_branch_mode_helpers_distinguish_camera_and_drop_satellite():
    assert not uses_satellite_branch(BRANCH_MODE_CAMERA_ONLY)
    assert not uses_satellite_branch(BRANCH_MODE_DROP_SATELLITE)
    assert uses_satellite_branch(BRANCH_MODE_FUSION)
    assert uses_satellite_branch(BRANCH_MODE_SAT_ONLY)
    assert uses_satellite_branch(BRANCH_MODE_DROP_CAMERA)

    assert uses_camera_branch(BRANCH_MODE_CAMERA_ONLY)
    assert not uses_camera_branch(BRANCH_MODE_SAT_ONLY)
    assert not uses_camera_branch(BRANCH_MODE_DROP_CAMERA)
    assert uses_camera_branch(BRANCH_MODE_FUSION)

    assert not uses_fusion_head(BRANCH_MODE_CAMERA_ONLY)
    assert not uses_fusion_head(BRANCH_MODE_SAT_ONLY)
    assert uses_fusion_head(BRANCH_MODE_DROP_SATELLITE)
    assert uses_fusion_head(BRANCH_MODE_DROP_CAMERA)
    assert uses_fusion_head(BRANCH_MODE_FUSION)


def test_compute_branch_metrics_matches_paper_definition():
    metrics = compute_branch_metrics(
        BranchMetricsInput(
            camera_only=0.35,
            fusion_base=0.41,
            fusion_base_drop_satellite=0.29,
            fusion_distilled=0.45,
            fusion_distilled_drop_satellite=0.34,
        )
    )

    assert metrics["FusionGain"] == pytest.approx(0.06)
    assert metrics["CameraDegenerationGap"] == pytest.approx(0.06)
    assert metrics["RecoveryGain"] == pytest.approx(0.05)
    assert metrics["FinalPerformanceGain"] == pytest.approx(0.04)


def test_summarize_branch_metrics_preserves_raw_scores_and_derived_metrics(tmp_path):
    rows = summarize_branch_metrics([
        {
            "model": "SatforHDMap",
            "metric": "iou",
            "camera_only": 0.35,
            "fusion_base": 0.41,
            "fusion_base_drop_satellite": 0.29,
            "fusion_distilled": 0.45,
            "fusion_distilled_drop_satellite": 0.34,
        }
    ])

    assert rows == [
        {
            "model": "SatforHDMap",
            "metric": "iou",
            "camera_only": 0.35,
            "fusion_base": 0.41,
            "fusion_base_drop_satellite": 0.29,
            "fusion_distilled": 0.45,
            "fusion_distilled_drop_satellite": 0.34,
            "FusionGain": pytest.approx(0.06),
            "CameraDegenerationGap": pytest.approx(0.06),
            "RecoveryGain": pytest.approx(0.05),
            "FinalPerformanceGain": pytest.approx(0.04),
        }
    ]

    output_path = tmp_path / "branch_metrics.csv"
    write_metrics_csv(output_path, rows)
    content = output_path.read_text()

    assert "model,metric,camera_only,fusion_base" in content
    assert "SatforHDMap,iou,0.35,0.41,0.29,0.45,0.34,0.06,0.06,0.05,0.04" in content
