import pytest
import torch

from src.models.factory import build_model


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
