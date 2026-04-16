import importlib.util
import sys
import threading
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "02_export_frames.py"
EXPORT_SH_PATH = ROOT / "export_frames.sh"


def _load_export_frames_module():
    fake_nuscenes = types.ModuleType("nuscenes")
    fake_eval = types.ModuleType("nuscenes.eval")
    fake_common = types.ModuleType("nuscenes.eval.common")
    fake_utils = types.ModuleType("nuscenes.eval.common.utils")

    class FakeQuaternion:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        @property
        def q(self):
            return [1.0, 0.0, 0.0, 0.0]

        @property
        def rotation_matrix(self):
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    fake_utils.Quaternion = FakeQuaternion

    fake_helpers = types.ModuleType("helpers")
    fake_offset = types.ModuleType("helpers.offset_grid_helpers")
    fake_offset.load_offset_grids = lambda *args, **kwargs: {}
    fake_offset.get_offset_for_coordinate = lambda *args, **kwargs: (0.0, 0.0)

    injected_modules = {
        "nuscenes": fake_nuscenes,
        "nuscenes.eval": fake_eval,
        "nuscenes.eval.common": fake_common,
        "nuscenes.eval.common.utils": fake_utils,
        "helpers": fake_helpers,
        "helpers.offset_grid_helpers": fake_offset,
    }

    previous_modules = {name: sys.modules.get(name) for name in injected_modules}
    sys.modules.update(injected_modules)
    try:
        spec = importlib.util.spec_from_file_location("export_frames_module", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_process_frame_uses_satellite_images_for_main_output(tmp_path):
    module = _load_export_frames_module()
    generator = module.SatImageGenerator.__new__(module.SatImageGenerator)
    generator.args = types.SimpleNamespace(
        reference_frame="ego",
        img_scaling=0.15,
        dry_run=False,
        crop_basemap=False,
        create_map_overlay=False,
        final_image_size_pixels=None,
    )
    generator.basemap_images = {}
    generator.satmap_images = {"singapore-onenorth": object()}
    generator.offset_grids = None
    generator.frame_counter_per_map = {"singapore-onenorth": 0}
    generator.frame_counter_lock = threading.Lock()
    generator._valid_final_pixels = None
    generator._calc_lidar_to_basemap_offset = lambda frame, ref_frame=None: ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    generator._quaternion_to_yaw = lambda quat: 0.0

    saved_paths = []

    class FakeImage:
        def resize(self, *args, **kwargs):
            return self

        def save(self, path):
            saved_paths.append(Path(path))

    generator._crop_then_rotate = lambda *args, **kwargs: FakeImage()
    generator.args.per_frame_output_path = str(tmp_path)

    frame = {
        "map_location": "singapore-onenorth",
        "token": "sample-token",
    }

    generator._process_frame([frame], 0)

    assert saved_paths == [tmp_path / "singapore-onenorth" / "0_sample-token.png"]


def test_export_script_enables_basemap_and_overlay_outputs():
    lines = [line.strip() for line in EXPORT_SH_PATH.read_text().splitlines()]

    assert "--crop_basemap \\" in lines
    assert "--create_map_overlay" in lines
