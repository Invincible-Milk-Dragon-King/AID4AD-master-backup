from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "decoder_probe.py"


def _load_decoder_probe_module():
    spec = importlib.util.spec_from_file_location("decoder_probe", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResettableDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.reset_calls = 0

    def reset_parameters(self):
        self.reset_calls += 1
        self.layer.reset_parameters()


class DummyProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.decoder = ResettableDecoder()

    def get_probe_encoder_modules(self, decoder_type: str):
        assert decoder_type == "camera"
        return [self.encoder]

    def get_probe_decoder(self, decoder_type: str):
        assert decoder_type == "camera"
        return self.decoder


def test_resolve_decoder_probe_config_maps_feature_sources():
    decoder_probe = _load_decoder_probe_module()

    camera_only = decoder_probe.resolve_decoder_probe_config("camera_only_camera")
    fusion_camera = decoder_probe.resolve_decoder_probe_config("fusion_base_camera")
    fusion_sat = decoder_probe.resolve_decoder_probe_config("fusion_distilled_sat")

    assert camera_only.decoder_type == "camera"
    assert camera_only.branch_mode == "camera_only"
    assert camera_only.load_strict is True
    assert fusion_camera.decoder_type == "camera"
    assert fusion_camera.branch_mode == "camera_only"
    assert fusion_camera.load_strict is False
    assert fusion_sat.decoder_type == "satellite"
    assert fusion_sat.branch_mode == "sat_only"
    assert fusion_sat.load_strict is True


def test_configure_decoder_probe_freezes_only_encoder_modules():
    decoder_probe = _load_decoder_probe_module()
    model = DummyProbeModel()

    decoder_probe.configure_decoder_probe_model(model, "camera")

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())


def test_reset_decoder_probe_parameters_reinitializes_decoder():
    decoder_probe = _load_decoder_probe_module()
    model = DummyProbeModel()
    initial_weight = model.decoder.layer.weight.detach().clone()

    decoder_probe.reset_decoder_probe_parameters(model, "camera")

    assert model.decoder.reset_calls == 1
    assert not torch.equal(model.decoder.layer.weight.detach(), initial_weight)


def test_set_decoder_probe_train_mode_keeps_encoder_eval_and_decoder_train():
    decoder_probe = _load_decoder_probe_module()
    model = DummyProbeModel()

    model.train()
    decoder_probe.set_decoder_probe_train_mode(model, "camera")

    assert model.training is True
    assert model.encoder.training is False
    assert model.decoder.training is True


def test_validate_decoder_probe_load_result_rejects_missing_keys():
    decoder_probe = _load_decoder_probe_module()
    config = decoder_probe.resolve_decoder_probe_config("fusion_base_camera")

    try:
        decoder_probe.validate_decoder_probe_load_result(
            config,
            missing_keys=["camera_bevencode.layer1.weight"],
            unexpected_keys=["fusion.weight"],
        )
    except ValueError as exc:
        assert "Missing keys" in str(exc)
    else:
        raise AssertionError("Expected missing probe keys to be rejected.")
