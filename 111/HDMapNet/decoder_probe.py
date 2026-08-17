"""Decoder-probe helpers for stock HDMapNet (camera / LiDAR / camera+LiDAR).

Semantic note:
  SatforHDMap's fusion is camera+satellite; here fusion means camera+LiDAR.
  Probe names keep launcher compatibility:
    camera_only_camera / fusion_base_camera  -> camera-branch probes
    lidar_only_lidar   / fusion_base_lidar   -> lidar-branch probes (protocol 1)
"""
from dataclasses import dataclass

BRANCH_MODE_CAMERA_ONLY = "camera_only"
BRANCH_MODE_LIDAR_ONLY = "lidar_only"

CAMERA_DECODER_PROBE_SOURCES = (
    "camera_only_camera",
    "fusion_base_camera",
)

LIDAR_DECODER_PROBE_SOURCES = (
    "lidar_only_lidar",
    "fusion_base_lidar",
)

DECODER_PROBE_SOURCES = CAMERA_DECODER_PROBE_SOURCES + LIDAR_DECODER_PROBE_SOURCES

# Keys that may be missing when loading a fusion/cam/lidar ckpt into a probe model.
CAMERA_PROBE_ALLOWED_MISSING_PREFIXES = (
    "camera_bevencode.",  # freshly created camera decoder on fusion model
    "pp.",                # LiDAR pillar encoder unused in camera-only forward
    "bevencode.",         # fusion decoder unused when probing camera branch
    "lidar_bevencode.",   # lidar decoder unused when probing camera branch
)

LIDAR_PROBE_ALLOWED_MISSING_PREFIXES = (
    "lidar_bevencode.",   # freshly created lidar decoder on older fusion ckpts
    "camera_bevencode.",  # camera decoder unused in lidar-only forward
    "bevencode.",         # fusion decoder unused when probing lidar branch
    "camencode.",
    "view_fusion.",
    "ipm.",
    "up_sampler.",
)


@dataclass(frozen=True)
class DecoderProbeConfig:
    feature_source: str
    decoder_type: str
    branch_mode: str
    load_strict: bool


def resolve_decoder_probe_config(feature_source: str) -> DecoderProbeConfig:
    if feature_source == "camera_only_camera":
        return DecoderProbeConfig(
            feature_source, "camera", BRANCH_MODE_CAMERA_ONLY, True
        )
    if feature_source == "fusion_base_camera":
        return DecoderProbeConfig(
            feature_source, "camera", BRANCH_MODE_CAMERA_ONLY, True
        )
    if feature_source == "lidar_only_lidar":
        return DecoderProbeConfig(
            feature_source, "lidar", BRANCH_MODE_LIDAR_ONLY, True
        )
    if feature_source == "fusion_base_lidar":
        return DecoderProbeConfig(
            feature_source, "lidar", BRANCH_MODE_LIDAR_ONLY, True
        )
    raise ValueError(f"Unsupported decoder probe source: {feature_source}")


def normalize_legacy_camera_state_dict(state_dict, model_state_dict):
    """Map cam-only ckpt `bevencode.*` -> `camera_bevencode.*` when needed."""
    legacy_bev_weight = state_dict.get("bevencode.conv1.weight")
    camera_bev_weight = model_state_dict.get("camera_bevencode.conv1.weight")
    fusion_bev_weight = model_state_dict.get("bevencode.conv1.weight")
    is_legacy_camera = (
        legacy_bev_weight is not None
        and camera_bev_weight is not None
        and fusion_bev_weight is not None
        and tuple(legacy_bev_weight.shape) == tuple(camera_bev_weight.shape)
        and tuple(legacy_bev_weight.shape) != tuple(fusion_bev_weight.shape)
    )
    if not is_legacy_camera:
        return state_dict, False

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("bevencode."):
            normalized[f"camera_bevencode.{key[len('bevencode.'):]}"] = value
        else:
            normalized[key] = value
    return normalized, True


def validate_decoder_probe_load_result(config: DecoderProbeConfig, missing_keys, unexpected_keys):
    allowed = (
        CAMERA_PROBE_ALLOWED_MISSING_PREFIXES
        if config.decoder_type == "camera"
        else LIDAR_PROBE_ALLOWED_MISSING_PREFIXES
    )
    missing_keys = [
        key
        for key in missing_keys
        if not key.startswith(allowed)
    ]
    if missing_keys:
        raise ValueError(
            f"Missing keys while loading decoder probe checkpoint for "
            f"{config.feature_source}: {missing_keys}"
        )
    # Unexpected keys (e.g. optimizer leftovers) are ignored for stock ckpts.
    _ = unexpected_keys


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _reset_if_supported(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def configure_decoder_probe_model(model, decoder_type: str):
    base_model = _unwrap_model(model)
    decoder = base_model.get_probe_decoder(decoder_type)
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    for parameter in decoder.parameters():
        parameter.requires_grad = True
    return decoder


def reset_decoder_probe_parameters(model, decoder_type: str):
    decoder = _unwrap_model(model).get_probe_decoder(decoder_type)
    decoder.apply(_reset_if_supported)
    return decoder


def set_decoder_probe_train_mode(model, decoder_type: str):
    base_model = _unwrap_model(model)
    base_model.train()
    decoder = base_model.get_probe_decoder(decoder_type)
    for module in base_model.get_probe_encoder_modules(decoder_type):
        module.eval()
    decoder.train()
    return decoder


def get_decoder_probe_trainable_parameters(model, decoder_type: str):
    decoder = _unwrap_model(model).get_probe_decoder(decoder_type)
    return [parameter for parameter in decoder.parameters() if parameter.requires_grad]
