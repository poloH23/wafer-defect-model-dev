from typing import get_args

import pytest
import torch

from src.common.config_schema import ModelConfig
from src.models.factory import SUPPORTED_MODELS, build_model, build_model_from_config
from src.models.profiling import count_flops, count_parameters


@pytest.mark.parametrize("small_input_stem", [True, False])
@pytest.mark.parametrize("in_channels", [1, 3])
def test_output_shape(small_input_stem, in_channels):
    model = build_model("resnet18", num_classes=8, in_channels=in_channels, small_input_stem=small_input_stem)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, in_channels, 52, 52))
    assert out.shape == (2, 8)


def test_small_stem_keeps_more_spatial_detail():
    # Feature map entering the global pooling layer: 7x7 with the small stem, 2x2 with the ImageNet stem
    sizes = {}
    for small in (True, False):
        model = build_model("resnet18", num_classes=8, in_channels=3, small_input_stem=small)
        captured = {}
        model.avgpool.register_forward_hook(lambda _m, inputs, _o, store=captured: store.update(shape=inputs[0].shape))
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 52, 52))
        sizes[small] = captured["shape"][-1]
    assert sizes == {True: 7, False: 2}


def test_unsupported_model_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        build_model("vgg16", num_classes=8, in_channels=3)


# ========== every supported architecture ==========


def final_feature_size(model: torch.nn.Module, in_channels: int) -> int:
    """Spatial size of the last convolution's output, i.e. the feature map the classifier pools over."""
    last_conv = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    captured = {}
    last_conv.register_forward_hook(lambda _m, _inputs, output: captured.update(size=output.shape[-1]))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, 52, 52))
    return captured["size"]


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_architecture_keeps_spatial_detail_with_the_small_input_stem(name):
    model = build_model(name, num_classes=8, in_channels=3, small_input_stem=True)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 52, 52)).shape == (2, 8)
    assert final_feature_size(model, 3) == 7


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_standard_stem_downsamples_32x(name):
    assert final_feature_size(build_model(name, num_classes=8, in_channels=3, small_input_stem=False), 3) == 2


@pytest.mark.parametrize("name", [n for n in SUPPORTED_MODELS if not n.startswith("resnet")])
def test_single_channel_input(name):
    model = build_model(name, num_classes=8, in_channels=1)
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 1, 52, 52)).shape == (2, 8)


def test_config_schema_lists_the_same_architectures():
    assert set(get_args(ModelConfig.model_fields["name"].annotation)) == set(SUPPORTED_MODELS)


# ========== profiling ==========


def test_count_flops_counts_a_multiply_accumulate_as_two():
    assert count_flops(torch.nn.Linear(4, 3, bias=False), (1, 4)) == 2 * 4 * 3


def test_count_flops_restores_the_training_mode():
    model = build_model("mobilenet_v3_small", num_classes=8, in_channels=3)
    model.train()
    count_flops(model, (1, 3, 52, 52))
    assert model.training


def test_count_parameters():
    assert count_parameters(torch.nn.Linear(4, 3)) == 4 * 3 + 3


def test_mobile_architectures_are_cheaper_than_resnet18():
    flops = {name: count_flops(build_model(name, 8, 3), (1, 3, 52, 52)) for name in ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0")}
    assert flops["mobilenet_v3_small"] < flops["resnet18"]
    assert flops["shufflenet_v2_x1_0"] < flops["resnet18"]


# ========== slim_resnet ==========


def test_slim_resnet_with_resnet18_settings_is_resnet18():
    reference = build_model("resnet18", num_classes=8, in_channels=3)
    slim = build_model("slim_resnet", num_classes=8, in_channels=3, width=64, depths=(2, 2, 2, 2))
    assert count_parameters(slim) == count_parameters(reference)
    assert count_flops(slim, (1, 3, 52, 52)) == count_flops(reference, (1, 3, 52, 52))


def test_slim_resnet_gets_cheaper_when_narrower_or_shallower():
    def flops(width: int, depth: int) -> int:
        return count_flops(build_model("slim_resnet", 8, 3, width=width, depths=(depth,) * 4), (1, 3, 52, 52))

    assert flops(16, 2) < flops(32, 2) < flops(64, 2)
    assert flops(32, 1) < flops(32, 2)


def test_slim_resnet_has_no_pretrained_weights():
    with pytest.raises(ValueError, match="no pretrained weights"):
        build_model("slim_resnet", 8, 3, pretrained=True)


def test_build_model_from_config_honours_every_option():
    # A slim config must not silently become the default width-64 ResNet-18
    cfg = ModelConfig(name="slim_resnet", width=16, depths=[1, 2, 1, 1], small_input_stem=False)
    from_config = build_model_from_config(cfg, num_classes=8, in_channels=3)
    direct = build_model("slim_resnet", 8, 3, small_input_stem=False, width=16, depths=(1, 2, 1, 1))
    assert count_parameters(from_config) == count_parameters(direct) < count_parameters(build_model("resnet18", 8, 3))
