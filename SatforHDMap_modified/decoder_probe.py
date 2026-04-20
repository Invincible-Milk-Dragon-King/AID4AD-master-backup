from dataclasses import dataclass

BRANCH_MODE_CAMERA_ONLY = "camera_only"
BRANCH_MODE_SAT_ONLY = "sat_only"


CAMERA_DECODER_PROBE_SOURCES = (
    "camera_only_camera",
    "fusion_base_camera",
    "fusion_distilled_camera",
)

SATELLITE_DECODER_PROBE_SOURCES = (
    "sat_only_sat",
    "fusion_base_sat",
    "fusion_distilled_sat",
)

DECODER_PROBE_SOURCES = CAMERA_DECODER_PROBE_SOURCES + SATELLITE_DECODER_PROBE_SOURCES


@dataclass(frozen=True)
class DecoderProbeConfig:
    feature_source: str
    decoder_type: str
    branch_mode: str
    load_strict: bool


def resolve_decoder_probe_config(feature_source: str) -> DecoderProbeConfig:
    if feature_source == "camera_only_camera":
        return DecoderProbeConfig(feature_source, "camera", BRANCH_MODE_CAMERA_ONLY, True)
    if feature_source in ("fusion_base_camera", "fusion_distilled_camera"):
        return DecoderProbeConfig(feature_source, "camera", BRANCH_MODE_CAMERA_ONLY, False)
    if feature_source in SATELLITE_DECODER_PROBE_SOURCES:
        return DecoderProbeConfig(feature_source, "satellite", BRANCH_MODE_SAT_ONLY, True)
    raise ValueError(f"Unsupported decoder probe source: {feature_source}")


def validate_decoder_probe_load_result(config: DecoderProbeConfig, missing_keys, unexpected_keys):
    if missing_keys:
        raise ValueError(
            f"Missing keys while loading decoder probe checkpoint for {config.feature_source}: {missing_keys}"
        )
    if config.load_strict and unexpected_keys:
        raise ValueError(
            f"Unexpected keys while loading strict decoder probe checkpoint for {config.feature_source}: {unexpected_keys}"
        )


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
