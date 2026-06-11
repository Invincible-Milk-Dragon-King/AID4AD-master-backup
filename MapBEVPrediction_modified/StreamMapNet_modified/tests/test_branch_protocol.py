import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    'stream_branch_protocol',
    ROOT / 'plugin/models/mapers/branch_protocol.py',
)
branch_protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(branch_protocol)

BRANCH_MODE_CAMERA_ONLY = branch_protocol.BRANCH_MODE_CAMERA_ONLY
BRANCH_MODE_DROP_CAMERA = branch_protocol.BRANCH_MODE_DROP_CAMERA
BRANCH_MODE_DROP_SATELLITE = branch_protocol.BRANCH_MODE_DROP_SATELLITE
BRANCH_MODE_FUSION = branch_protocol.BRANCH_MODE_FUSION
BRANCH_MODE_SATELLITE_ONLY = branch_protocol.BRANCH_MODE_SATELLITE_ONLY
BRANCH_MODE_SAT_ONLY = branch_protocol.BRANCH_MODE_SAT_ONLY
resolve_branch_mode = branch_protocol.resolve_branch_mode
uses_aerial_encoder = branch_protocol.uses_aerial_encoder
uses_aerial_fuser = branch_protocol.uses_aerial_fuser
uses_camera_encoder = branch_protocol.uses_camera_encoder
uses_fusion_head = branch_protocol.uses_fusion_head


def test_stream_branch_mode_resolution_supports_legacy_and_explicit_modes():
    assert resolve_branch_mode(False, False, None) == BRANCH_MODE_CAMERA_ONLY
    assert resolve_branch_mode(True, False, None) == BRANCH_MODE_FUSION
    assert resolve_branch_mode(True, True, None) == BRANCH_MODE_SATELLITE_ONLY
    assert resolve_branch_mode(True, False, BRANCH_MODE_DROP_SATELLITE) == BRANCH_MODE_DROP_SATELLITE
    assert resolve_branch_mode(True, False, BRANCH_MODE_SAT_ONLY) == BRANCH_MODE_SATELLITE_ONLY


def test_stream_branch_helpers_match_expected_encoder_usage():
    assert not uses_aerial_encoder(BRANCH_MODE_CAMERA_ONLY)
    assert uses_aerial_encoder(BRANCH_MODE_FUSION)
    assert not uses_aerial_encoder(BRANCH_MODE_DROP_SATELLITE)
    assert uses_aerial_encoder(BRANCH_MODE_DROP_CAMERA)
    assert uses_aerial_encoder(BRANCH_MODE_SATELLITE_ONLY)

    assert not uses_camera_encoder(BRANCH_MODE_SATELLITE_ONLY)
    assert uses_camera_encoder(BRANCH_MODE_FUSION)
    assert uses_camera_encoder(BRANCH_MODE_DROP_SATELLITE)
    assert not uses_camera_encoder(BRANCH_MODE_DROP_CAMERA)

    assert not uses_aerial_fuser(BRANCH_MODE_CAMERA_ONLY)
    assert uses_aerial_fuser(BRANCH_MODE_FUSION)
    assert uses_aerial_fuser(BRANCH_MODE_DROP_SATELLITE)
    assert uses_aerial_fuser(BRANCH_MODE_DROP_CAMERA)
    assert uses_fusion_head(BRANCH_MODE_DROP_CAMERA)


def test_stream_mapper_exposes_branch_feature_extractor():
    source = (ROOT / 'plugin/models/mapers/StreamMapNet.py').read_text()
    assert 'def extract_branch_features' in source
    assert 'def get_probe_decoder' in source
    assert 'def get_probe_encoder_modules' in source


def test_decoder_probe_sources_are_defined():
    from decoder_probe import DECODER_PROBE_SOURCES

    assert 'fusion_base_camera' in DECODER_PROBE_SOURCES
    assert 'fusion_distilled_sat' in DECODER_PROBE_SOURCES


def test_branch_experiment_scripts_exist():
    script_dir = ROOT / 'branch_experiments'
    expected = [
        'train_camera_only_newsplit.sh',
        'train_sat_only_newsplit.sh',
        'train_fusion_base_newsplit.sh',
        'train_fusion_distilled_camera_newsplit.sh',
        'train_fusion_distilled_satellite_newsplit.sh',
        'eval_fusion_base_drop_satellite_newsplit.sh',
        'eval_fusion_base_drop_camera_newsplit.sh',
        'eval_fusion_distilled_drop_satellite_newsplit.sh',
        'eval_fusion_distilled_drop_camera_newsplit.sh',
        'train_camera_decoder_probe_camera_only_newsplit.sh',
        'train_camera_decoder_probe_fusion_base_newsplit.sh',
        'train_camera_decoder_probe_fusion_distilled_newsplit.sh',
        'train_sat_decoder_probe_sat_only_newsplit.sh',
        'train_sat_decoder_probe_fusion_base_newsplit.sh',
        'train_sat_decoder_probe_fusion_distilled_newsplit.sh',
    ]
    for name in expected:
        assert (script_dir / name).is_file(), name


def test_distillation_loss_is_normalized_in_mapper():
    source = (ROOT / 'plugin/models/mapers/StreamMapNet.py').read_text()
    assert 'teacher_feature.pow(2).mean().sqrt()' in source


def test_distillation_loss_supports_gt_non_mask():
    source = (ROOT / 'plugin/models/mapers/StreamMapNet.py').read_text()
    assert 'def _build_distill_non_mask' in source
    assert 'self.distill_use_mask' in source
    assert 'non_mask=non_mask' in source


def test_distilled_configs_enable_masked_distillation():
    for config_name in [
        'plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD_distilled.py',
        'plugin/configs/nusc_newsplit_480_60x30_24e_AID4AD_distilled_satellite.py',
    ]:
        source = (ROOT / config_name).read_text()
        assert 'use_mask=True' in source
