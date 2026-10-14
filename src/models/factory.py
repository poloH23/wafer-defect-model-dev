"""
Build classification backbones adapted to this task's input size, channels and number of labels.
"""

from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn

SUPPORTED_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
)

# Every supported architecture downsamples its input 32x in total, in five stride-2 steps, a design
# made for 224x224 photos. Removing the first two steps keeps a 52x52 map at 7x7 before the
# classifier instead of 2x2, the same as the small-input ResNet stem.
SMALL_INPUT_REMOVED_DOWNSAMPLING = 2


def build_model(name: str, num_classes: int, in_channels: int, pretrained: bool = False, small_input_stem: bool = True) -> nn.Module:
    """
    A torchvision architecture with its input layer, early downsampling and classifier head
    adapted to the task.

    small_input_stem:
        ResNet: the ImageNet stem (7x7 conv stride 2 + max pool) is replaced by a 3x3 stride-1 conv
        without max pool (the CIFAR-style stem). Other architectures: their first two stride-2
        operations become stride 1. Either way a 52x52 input reaches the classifier as 7x7.

    pretrained:
        Loads ImageNet weights. Layers that are replaced (the first conv when the number of input
        channels changes, the ResNet stem, the classifier head) are re-initialized.

    The head outputs raw logits (no sigmoid): BCEWithLogitsLoss applies the sigmoid itself,
    in a numerically stable way.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{name}', choose from {SUPPORTED_MODELS}")

    model = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        # Kept exactly as before, so checkpoints of earlier runs still load into the same architecture
        _adapt_resnet_stem(model, name, in_channels, small_input_stem)
    else:
        if small_input_stem:
            _remove_first_downsampling(model, SMALL_INPUT_REMOVED_DOWNSAMPLING)
        _set_input_channels(model, in_channels)
    _replace_last_linear(model, num_classes)
    return model


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the submodule at a dotted path such as "features.0.0"."""
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _adapt_resnet_stem(model: nn.Module, name: str, in_channels: int, small_input_stem: bool) -> None:
    """get_model is typed to return a plain nn.Module, so type checkers cannot know conv1 exists."""
    conv1 = model.conv1
    if not isinstance(conv1, nn.Conv2d):
        raise TypeError(f"'{name}' does not have the expected ResNet stem (conv1)")
    if small_input_stem or in_channels != 3:
        kernel_size, stride, padding = (3, 1, 1) if small_input_stem else (7, 2, 3)
        model.conv1 = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if small_input_stem:
        model.maxpool = nn.Identity()


def _remove_first_downsampling(model: nn.Module, count: int) -> None:
    """
    Make the first 'count' downsampling operations stride 1, in the order the modules are defined,
    which for the supported architectures is the forward order. A stride-2 convolution keeps its
    padding, so it now preserves the spatial size; a stride-2 max pool is removed.

    Only safe for the stem and first stage: deeper stages of some architectures (e.g. the two
    parallel branches of a ShuffleNet block) must be changed together to keep shapes consistent.
    """
    targets = []
    for name, module in model.named_modules():
        if len(targets) == count:
            break
        if _is_downsampling(module):
            targets.append((name, module))
    if len(targets) < count:
        raise ValueError(f"Found only {len(targets)} downsampling operations, expected at least {count}")

    for name, module in targets:
        if isinstance(module, nn.Conv2d):
            module.stride = (1, 1)
        else:
            _set_module(model, name, nn.Identity())


def _is_downsampling(module: nn.Module) -> bool:
    """A stride-2 convolution or max pool, i.e. a step that halves the spatial size."""
    if isinstance(module, nn.Conv2d):
        return module.stride == (2, 2)
    if isinstance(module, nn.MaxPool2d):
        return module.stride in (2, (2, 2))
    return False


def _set_input_channels(model: nn.Module, in_channels: int) -> None:
    """Replace the first convolution when the number of input channels differs; everything else is kept."""
    name, conv = next((n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d))
    if conv.in_channels == in_channels:
        return
    _set_module(
        model,
        name,
        nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        ),
    )


def _replace_last_linear(model: nn.Module, num_classes: int) -> None:
    """Replace the final fully connected layer (the classifier head) with one producing `num_classes` logits."""
    name, linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    _set_module(model, name, nn.Linear(linear.in_features, num_classes))


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """
    Rebuild a model from a checkpoint saved by scripts/train.py, in evaluation mode.

    The checkpoint stores everything needed to rebuild the architecture, so evaluation never depends
    on config files that may have changed since training. pretrained=False because every weight is
    restored from the checkpoint anyway, so nothing needs to be downloaded.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_cfg = checkpoint["model_config"]
    model = build_model(model_cfg["name"], checkpoint["num_classes"], checkpoint["in_channels"], pretrained=False, small_input_stem=model_cfg["small_input_stem"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
