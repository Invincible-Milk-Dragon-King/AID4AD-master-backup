import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPEC = importlib.util.spec_from_file_location(
    "stream_branch_protocol",
    ROOT / "plugin/models/mapers/branch_protocol.py",
)
branch_protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(branch_protocol)

BRANCH_MODE_CAMERA_ONLY = branch_protocol.BRANCH_MODE_CAMERA_ONLY
BRANCH_MODE_DROP_SATELLITE = branch_protocol.BRANCH_MODE_DROP_SATELLITE
BRANCH_MODE_FUSION = branch_protocol.BRANCH_MODE_FUSION
BRANCH_MODE_SATELLITE_ONLY = branch_protocol.BRANCH_MODE_SATELLITE_ONLY
resolve_branch_mode = branch_protocol.resolve_branch_mode
uses_aerial_encoder = branch_protocol.uses_aerial_encoder
uses_aerial_fuser = branch_protocol.uses_aerial_fuser


def test_stream_branch_mode_resolution_supports_legacy_and_explicit_modes():
    assert resolve_branch_mode(False, False, None) == BRANCH_MODE_CAMERA_ONLY
    assert resolve_branch_mode(True, False, None) == BRANCH_MODE_FUSION
    assert resolve_branch_mode(True, True, None) == BRANCH_MODE_SATELLITE_ONLY
    assert resolve_branch_mode(True, False, BRANCH_MODE_DROP_SATELLITE) == BRANCH_MODE_DROP_SATELLITE


def test_stream_branch_helpers_match_expected_encoder_usage():
    assert not uses_aerial_encoder(BRANCH_MODE_CAMERA_ONLY)
    assert uses_aerial_encoder(BRANCH_MODE_FUSION)
    assert not uses_aerial_encoder(BRANCH_MODE_DROP_SATELLITE)
    assert uses_aerial_encoder(BRANCH_MODE_SATELLITE_ONLY)

    assert not uses_aerial_fuser(BRANCH_MODE_CAMERA_ONLY)
    assert uses_aerial_fuser(BRANCH_MODE_FUSION)
    assert not uses_aerial_fuser(BRANCH_MODE_DROP_SATELLITE)
    assert not uses_aerial_fuser(BRANCH_MODE_SATELLITE_ONLY)


def test_stream_mapper_exposes_branch_feature_extractor():
    source = (ROOT / "plugin/models/mapers/StreamMapNet.py").read_text()
    tree = ast.parse(source)

    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StreamMapNet"
    )
    method_names = {
        node.name for node in class_node.body
        if isinstance(node, ast.FunctionDef)
    }

    assert "extract_branch_features" in method_names
