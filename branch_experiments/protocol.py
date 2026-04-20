from dataclasses import dataclass


BRANCH_MODE_CAMERA_ONLY = "camera_only"
BRANCH_MODE_SAT_ONLY = "sat_only"
BRANCH_MODE_FUSION = "fusion"
BRANCH_MODE_DROP_SATELLITE = "drop_satellite"
BRANCH_MODE_DROP_CAMERA = "drop_camera"

BRANCH_MODES = (
    BRANCH_MODE_CAMERA_ONLY,
    BRANCH_MODE_SAT_ONLY,
    BRANCH_MODE_FUSION,
    BRANCH_MODE_DROP_SATELLITE,
    BRANCH_MODE_DROP_CAMERA,
)


@dataclass(frozen=True)
class BranchMetricsInput:
    camera_only: float
    fusion_base: float
    fusion_base_drop_satellite: float
    fusion_distilled: float
    fusion_distilled_drop_satellite: float


def uses_satellite_branch(branch_mode: str) -> bool:
    return branch_mode in (BRANCH_MODE_FUSION, BRANCH_MODE_SAT_ONLY, BRANCH_MODE_DROP_CAMERA)


def uses_camera_branch(branch_mode: str) -> bool:
    return branch_mode in (BRANCH_MODE_CAMERA_ONLY, BRANCH_MODE_FUSION, BRANCH_MODE_DROP_SATELLITE)


def uses_fusion_head(branch_mode: str) -> bool:
    return branch_mode in (BRANCH_MODE_FUSION, BRANCH_MODE_DROP_SATELLITE, BRANCH_MODE_DROP_CAMERA)


def compute_branch_metrics(metrics: BranchMetricsInput) -> dict[str, float]:
    return {
        "FusionGain": metrics.fusion_base - metrics.camera_only,
        "CameraDegenerationGap": metrics.camera_only - metrics.fusion_base_drop_satellite,
        "RecoveryGain": metrics.fusion_distilled_drop_satellite - metrics.fusion_base_drop_satellite,
        "FinalPerformanceGain": metrics.fusion_distilled - metrics.fusion_base,
    }
